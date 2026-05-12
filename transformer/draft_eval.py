"""Evaluation helpers for baseline ART and draft ART."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import DecisionTransformerConfig

from optimization.rpod_scenario import state_rtn_target
from transformer.art import AutonomousRendezvousTransformer
from transformer.draft_art import DraftARTConfig, PhysicsInformedDraftART
from transformer.draft_losses import rollout_dynamics
from utils.metrics import fuel_cost, koz_violation, terminal_errors
from utils.timing import synchronize_if_cuda


def load_repository_tensors(dataset_dir: str, seed: int) -> Tuple[Dict[str, torch.Tensor], str]:
    """Load ART dataset tensors, falling back to reproducible synthetic data.

    Args:
        dataset_dir: Repository dataset directory.
        seed: Seed for synthetic fallback.

    Returns:
        Tuple of tensor dictionary and dataset label. Shapes are
        ``states=float32[D,N,6]``, ``actions=float32[D,N,3]``, ``rtgs=float32[D,N,1]``,
        and ``ctgs=float32[D,N,1]``. RTN state units are ``[m, m/s]`` for repository
        data and synthetic fallback.
    """
    required = [
        "torch_states_rtn_scp.pth",
        "torch_states_rtn_cvx.pth",
        "torch_actions_scp.pth",
        "torch_actions_cvx.pth",
        "torch_rtgs_scp.pth",
        "torch_rtgs_cvx.pth",
        "torch_ctgs_scp.pth",
        "torch_ctgs_cvx.pth",
    ]
    if all(os.path.exists(os.path.join(dataset_dir, name)) for name in required):
        states = torch.cat((torch.load(os.path.join(dataset_dir, required[0])), torch.load(os.path.join(dataset_dir, required[1]))), dim=0)
        actions = torch.cat((torch.load(os.path.join(dataset_dir, required[2])), torch.load(os.path.join(dataset_dir, required[3]))), dim=0)
        rtgs = torch.cat((torch.load(os.path.join(dataset_dir, required[4])), torch.load(os.path.join(dataset_dir, required[5]))), dim=0)
        ctgs = torch.cat((torch.load(os.path.join(dataset_dir, required[6])), torch.load(os.path.join(dataset_dir, required[7]))), dim=0)
        return {"states": states.float(), "actions": actions.float(), "rtgs": rtgs.float(), "ctgs": ctgs.float()}, dataset_dir
    generator = torch.Generator().manual_seed(seed)
    num_data, num_steps = 32, 24
    initial_pos = torch.randn(num_data, 3, generator=generator) * 40.0 + torch.tensor([0.0, -120.0, 0.0])
    target = torch.tensor(state_rtn_target, dtype=torch.float32)
    states = torch.zeros(num_data, num_steps, 6)
    actions = torch.zeros(num_data, num_steps, 3)
    for i in range(num_data):
        for k in range(num_steps):
            alpha = k / max(num_steps - 1, 1)
            pos = (1 - alpha) * initial_pos[i] + alpha * target[:3]
            vel = (target[:3] - initial_pos[i]) / num_steps
            states[i, k, :3] = pos + 0.5 * torch.randn(3, generator=generator)
            states[i, k, 3:] = vel
            actions[i, k] = -0.02 * states[i, k, 3:] + 0.002 * torch.randn(3, generator=generator)
    fuel_remaining = torch.flip(torch.cumsum(torch.flip(torch.linalg.norm(actions, dim=-1, keepdim=True), dims=[1]), dim=1), dims=[1])
    rtgs = fuel_remaining
    ctgs = torch.zeros(num_data, num_steps, 1)
    return {"states": states, "actions": actions, "rtgs": rtgs, "ctgs": ctgs}, "synthetic_fallback_no_repository_dataset"


def normalize_split(tensors: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
    """Normalize repository tensors using existing ART convention.

    Args:
        tensors: Raw tensor dictionary.

    Returns:
        Normalized data dictionary, stats dictionary, and train split index.
    """
    states = tensors["states"]
    actions = tensors["actions"]
    stats = {
        "states_mean": states.mean(dim=0),
        "states_std": states.std(dim=0) + 1e-6,
        "actions_mean": actions.mean(dim=0),
        "actions_std": actions.std(dim=0) + 1e-6,
    }
    data = {
        "states": (states - stats["states_mean"]) / stats["states_std"],
        "actions": (actions - stats["actions_mean"]) / stats["actions_std"],
        "rtgs": tensors["rtgs"],
        "ctgs": tensors["ctgs"],
        "raw_states": states,
        "raw_actions": actions,
    }
    split = int(0.9 * states.shape[0])
    return data, stats, split


def build_full_art_model(state_dim: int, action_dim: int, device: torch.device, checkpoint: Optional[str] = None) -> AutonomousRendezvousTransformer:
    """Build the existing repository ART baseline model without changing it.

    Args:
        state_dim: State dimension, usually 6.
        action_dim: Control dimension, usually 3.
        device: Torch device.
        checkpoint: Optional checkpoint directory or file. Directory checkpoints
            saved by Accelerate are not required for smoke-test evaluation.

    Returns:
        Existing :class:`AutonomousRendezvousTransformer` in eval mode.
    """
    config = DecisionTransformerConfig(
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=384,
        max_ep_len=128,
        vocab_size=1,
        action_tanh=False,
        n_positions=1024,
        n_layer=6,
        n_head=6,
    )
    model = AutonomousRendezvousTransformer(config).to(device)
    if checkpoint and os.path.isfile(checkpoint):
        payload = torch.load(checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
    model.eval()
    return model


def evaluate_full_art_baseline(
    model: AutonomousRendezvousTransformer,
    data: Dict[str, torch.Tensor],
    stats: Dict[str, torch.Tensor],
    split: int,
    num_eval_samples: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate existing ART baseline by teacher-forced one-shot inference.

    Args:
        model: Existing repository ART model.
        data: Normalized tensors with raw counterparts.
        stats: Normalization stats.
        split: Evaluation starts at this sample index.
        num_eval_samples: Number of samples.
        device: CPU or CUDA.

    Returns:
        Metrics dictionary plus representative states and controls.
    """
    runtimes = []
    records = []
    max_samples = min(num_eval_samples, data["states"].shape[0] - split)
    rep_states = data["raw_states"][split].numpy()
    rep_controls = data["raw_actions"][split].numpy()
    for i in range(max_samples):
        idx = split + i
        states = data["states"][idx : idx + 1].to(device)
        actions = torch.zeros_like(data["actions"][idx : idx + 1]).to(device)
        rtgs = data["rtgs"][idx : idx + 1].to(device)
        ctgs = data["ctgs"][idx : idx + 1].to(device)
        timesteps = torch.arange(states.shape[1], device=device).view(1, -1).long()
        attention = torch.ones(1, states.shape[1], device=device).long()
        synchronize_if_cuda(device)
        start = time.perf_counter()
        with torch.no_grad():
            _, pred = model(states=states, actions=actions, returns_to_go=rtgs, constraints_to_go=ctgs, timesteps=timesteps, attention_mask=attention, return_dict=False)
        synchronize_if_cuda(device)
        runtimes.append(time.perf_counter() - start)
        control = (pred.cpu()[0] * stats["actions_std"] + stats["actions_mean"]).numpy()
        raw_state = data["raw_states"][idx].numpy()
        records.append({"fuel_cost_m_s": fuel_cost(control), **terminal_errors(raw_state), **koz_violation(raw_state)})
        if i == 0:
            rep_controls = control
    return {
        "full_art_inference_time_mean_s": float(np.mean(runtimes)),
        "full_art_inference_time_std_s": float(np.std(runtimes)),
        "fuel_cost_m_s": float(np.mean([r["fuel_cost_m_s"] for r in records])),
        "terminal_position_error_m": float(np.mean([r["terminal_position_error_m"] for r in records])),
        "terminal_velocity_error_m_s": float(np.mean([r["terminal_velocity_error_m_s"] for r in records])),
        "koz_max_violation": float(np.mean([r["koz_max_violation"] for r in records])),
    }, rep_states, rep_controls


def evaluate_draft_model(
    model: PhysicsInformedDraftART,
    data: Dict[str, torch.Tensor],
    stats: Dict[str, torch.Tensor],
    split: int,
    num_eval_samples: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate draft-only chunk inference.

    Args:
        model: Draft model.
        data: Normalized tensors.
        stats: Normalization stats.
        split: Evaluation split index.
        num_eval_samples: Number of samples.
        device: CPU or CUDA.

    Returns:
        Metrics dictionary plus representative raw states and denormalized controls.
    """
    model.eval()
    runtimes = []
    records = []
    max_samples = min(num_eval_samples, data["states"].shape[0] - split)
    rep_states = data["raw_states"][split].numpy()
    rep_controls = data["raw_actions"][split].numpy()
    for i in range(max_samples):
        idx = split + i
        states = data["states"][idx : idx + 1].to(device)
        actions = torch.zeros(1, 1, 3, device=device)
        fuel = data["rtgs"][idx : idx + 1, :1].to(device)
        ctg = data["ctgs"][idx : idx + 1, :1].to(device)
        synchronize_if_cuda(device)
        start = time.perf_counter()
        with torch.no_grad():
            chunks = []
            k = 0
            while k < states.shape[1]:
                hist = states[:, : k + 1]
                action_hist = torch.zeros(1, k + 1, 3, device=device)
                if k:
                    action_hist[:, 1:] = torch.cat(chunks, dim=1)[:, :k]
                pred = model(hist, action_hist, fuel.expand(-1, k + 1, -1), ctg.expand(-1, k + 1, -1), horizon=min(model.config.draft_chunk_length, states.shape[1] - k))
                chunks.append(pred)
                k += pred.shape[1]
            pred_all = torch.cat(chunks, dim=1)[:, : states.shape[1]]
        synchronize_if_cuda(device)
        runtimes.append(time.perf_counter() - start)
        control = (pred_all.cpu()[0] * stats["actions_std"] + stats["actions_mean"]).numpy()
        raw_state = data["raw_states"][idx].numpy()
        records.append({"fuel_cost_m_s": fuel_cost(control), **terminal_errors(raw_state), **koz_violation(raw_state)})
        if i == 0:
            rep_controls = control
    return {
        "draft_inference_time_mean_s": float(np.mean(runtimes)),
        "draft_inference_time_std_s": float(np.std(runtimes)),
        "fuel_cost_m_s": float(np.mean([r["fuel_cost_m_s"] for r in records])),
        "terminal_position_error_m": float(np.mean([r["terminal_position_error_m"] for r in records])),
        "terminal_velocity_error_m_s": float(np.mean([r["terminal_velocity_error_m_s"] for r in records])),
        "koz_max_violation": float(np.mean([r["koz_max_violation"] for r in records])),
    }, rep_states, rep_controls


def load_draft_checkpoint(path: str, device: torch.device) -> PhysicsInformedDraftART:
    """Load a draft checkpoint saved by ``train_draft_model``."""
    payload = torch.load(path, map_location=device)
    config = DraftARTConfig(**payload["config"])
    model = PhysicsInformedDraftART(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
