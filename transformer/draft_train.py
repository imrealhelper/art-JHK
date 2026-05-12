"""Training utilities for PhysicsInformedDraftART."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from transformer.draft_art import DraftARTConfig, PhysicsInformedDraftART
from transformer.draft_losses import DraftLossConfig, total_draft_loss
from utils.timing import timed_block


class DraftChunkDataset(Dataset):
    """Chunked dataset wrapper for ART draft training.

    Items contain normalized tensors from repository ART data or a seeded synthetic
    fallback. Shapes: ``state_context=float32[L+1,6]``,
    ``action_context=float32[L+1,3]``, ``target_control=float32[K,3]`` in
    normalized delta-v units, ``fuel_to_go=float32[L+1,1]`` in positive fuel units,
    ``constraint_to_go=float32[L+1,1]``, and ``loss_mask=bool[K]``.
    """

    def __init__(self, states: torch.Tensor, actions: torch.Tensor, rtgs: torch.Tensor, ctgs: torch.Tensor, chunk_length: int):
        self.states = states.float()
        self.actions = actions.float()
        self.rtgs = rtgs.float()
        self.ctgs = ctgs.float()
        self.chunk_length = chunk_length
        self.num_data, self.num_steps, _ = self.states.shape

    def __len__(self) -> int:
        return self.num_data * self.num_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = idx % self.num_data
        step = (idx // self.num_data) % self.num_steps
        end = min(step + self.chunk_length, self.num_steps)
        state_context = self.states[sample, : step + 1]
        action_context = torch.zeros(step + 1, self.actions.shape[-1])
        if step > 0:
            action_context[1:] = self.actions[sample, :step]
        rtg_context = self.rtgs[sample, : step + 1]
        fuel_context = -rtg_context if torch.nanmean(rtg_context) < 0 else rtg_context
        ctg_context = self.ctgs[sample, : step + 1]
        target = torch.zeros(self.chunk_length, self.actions.shape[-1])
        target[: end - step] = self.actions[sample, step:end]
        mask = torch.zeros(self.chunk_length, dtype=torch.bool)
        mask[: end - step] = True
        return {
            "state_context": state_context,
            "action_context": action_context,
            "fuel_context": fuel_context,
            "constraint_context": ctg_context,
            "target_control": target,
            "loss_mask": mask,
            "fuel_now": fuel_context[-1:].view(1),
        }


def collate_chunks(batch):
    """Right-pad variable history contexts for draft training batches."""
    max_len = max(item["state_context"].shape[0] for item in batch)
    batch_size = len(batch)
    state = torch.zeros(batch_size, max_len, 6)
    action = torch.zeros(batch_size, max_len, 3)
    fuel = torch.zeros(batch_size, max_len, 1)
    ctg = torch.zeros(batch_size, max_len, 1)
    attention = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        length = item["state_context"].shape[0]
        state[i, :length] = item["state_context"]
        action[i, :length] = item["action_context"]
        fuel[i, :length] = item["fuel_context"]
        ctg[i, :length] = item["constraint_context"]
        attention[i, :length] = True
    return {
        "state": state,
        "action": action,
        "fuel": fuel,
        "ctg": ctg,
        "attention_mask": attention,
        "target_control": torch.stack([item["target_control"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "fuel_now": torch.stack([item["fuel_now"] for item in batch]),
    }


@dataclass
class TrainResult:
    """Draft training result."""

    model: PhysicsInformedDraftART
    history: Dict[str, float]
    training_time_s: float
    checkpoint_path: str


def train_draft_model(
    *,
    train_dataset: DraftChunkDataset,
    config: DraftARTConfig,
    loss_config: DraftLossConfig,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    output_dir: str,
    use_distillation: bool,
    use_physics_loss: bool,
    use_acceptance_loss: bool,
    teacher_model: Optional[torch.nn.Module] = None,
) -> TrainResult:
    """Train a small draft model.

    Args:
        train_dataset: Chunk dataset with normalized ART tensors.
        config: Draft model config.
        loss_config: Loss config and weights.
        device: CPU or CUDA device.
        batch_size: Mini-batch size.
        epochs: Number of epochs. CPU smoke tests may use 1.
        learning_rate: AdamW learning rate.
        output_dir: Directory for ``draft_art.pt``.
        use_distillation: Enables teacher MSE when teacher predictions are supplied.
        use_physics_loss: Enables control/fuel physics-informed losses.
        use_acceptance_loss: Enables verifier acceptance proxy.
        teacher_model: Optional frozen ART teacher; currently used only by callers
            that precompute teacher chunks.

    Returns:
        :class:`TrainResult` with checkpoint path and scalar history.
    """
    del teacher_model
    model = PhysicsInformedDraftART(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_chunks)
    max_batches = max(1, min(len(loader), 20))
    history: Dict[str, float] = {}
    timing: Dict[str, float] = {}
    with timed_block("training", device, timing):
        for _epoch in range(max(1, epochs)):
            model.train()
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                pred = model(
                    batch["state"].to(device),
                    batch["action"].to(device),
                    batch["fuel"].to(device),
                    batch["ctg"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                losses = total_draft_loss(
                    pred,
                    batch["target_control"].to(device),
                    loss_config,
                    loss_mask=batch["loss_mask"].to(device),
                    teacher_control_dv=batch["target_control"].to(device) if use_distillation or use_acceptance_loss else None,
                    fuel_to_go=batch["fuel_now"].to(device),
                    use_distillation=use_distillation,
                    use_physics_loss=use_physics_loss,
                    use_acceptance_loss=use_acceptance_loss,
                )
                losses["total"].backward()
                optimizer.step()
                history = {key: float(value.detach().cpu()) for key, value in losses.items()}
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "draft_art.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, checkpoint_path)
    return TrainResult(model=model, history=history, training_time_s=timing["training"], checkpoint_path=checkpoint_path)
