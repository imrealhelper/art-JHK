"""Continuous-control speculative draft-and-verify inference for ART."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from transformer.draft_losses import rollout_dynamics

TeacherFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass
class SpeculativeResult:
    """Output container for speculative inference."""

    warm_start_states: torch.Tensor
    warm_start_controls: torch.Tensor
    accepted_prefix_lengths: List[int]
    rejection_count: int
    full_art_call_count: int
    timing_dictionary: Dict[str, float]


def _longest_prefix(mask: torch.Tensor) -> int:
    prefix = 0
    for item in mask.tolist():
        if bool(item):
            prefix += 1
        else:
            break
    return prefix


def speculative_inference(
    *,
    initial_state: torch.Tensor,
    initial_action: torch.Tensor,
    initial_fuel_to_go: torch.Tensor,
    initial_constraint_to_go: torch.Tensor,
    full_art_verifier: Optional[TeacherFn],
    draft_model: torch.nn.Module,
    num_intervals: int,
    draft_chunk_length: int,
    time_step_s: float,
    control_scale_m_s: float,
    accept_threshold: float,
    u_max_m_s: float,
    device: torch.device,
) -> SpeculativeResult:
    """Run speculative-decoding-inspired continuous-control draft verification.

    Args:
        initial_state: ``float32[1, 6]`` initial RTN/ROE state; RTN units are
            position ``[m]`` and velocity ``[m/s]`` when unnormalized.
        initial_action: ``float32[1, 3]`` previous control in ``[m/s]``.
        initial_fuel_to_go: ``float32[1, 1]`` positive fuel-to-go ``phi_0`` in ``[m/s]``.
        initial_constraint_to_go: ``float32[1, 1]`` CTG ``psi_0`` dimensionless.
        full_art_verifier: Callable returning ``float32[1, H, 3]`` verifier controls,
            or None to use zero verifier controls.
        draft_model: Draft model returning ``float32[1, H, 3]`` controls.
        num_intervals: Number of control intervals ``N``.
        draft_chunk_length: Speculative chunk length ``K``.
        time_step_s: Step duration ``Delta t`` in ``[s]``.
        control_scale_m_s: Agreement distance normalizer in ``[m/s]``.
        accept_threshold: Squared normalized verifier threshold.
        u_max_m_s: Maximum impulse norm in ``[m/s]``.
        device: Torch device.

    Returns:
        :class:`SpeculativeResult` with states ``float32[1, M + 1, 6]`` and controls
        ``float32[1, M, 3]`` where ``M == num_intervals``.
    """
    start = time.perf_counter()
    current_state = initial_state.to(device).float()
    state_context = current_state.unsqueeze(1)
    action_context = initial_action.to(device).float().unsqueeze(1)
    fuel_context = initial_fuel_to_go.to(device).float().unsqueeze(1)
    constraint_context = initial_constraint_to_go.to(device).float().unsqueeze(1)
    accepted_states = [current_state]
    accepted_controls: List[torch.Tensor] = []
    prefix_lengths: List[int] = []
    rejection_count = 0
    full_art_call_count = 0
    k = 0
    draft_model.eval()
    while k < num_intervals:
        horizon = min(draft_chunk_length, num_intervals - k)
        with torch.no_grad():
            draft_control = draft_model(
                state_context,
                action_context,
                fuel_context,
                constraint_context,
                horizon=horizon,
            )
            if full_art_verifier is not None:
                teacher_control = full_art_verifier(
                    state_context, action_context, fuel_context, constraint_context, horizon
                )
                full_art_call_count += 1
            else:
                teacher_control = torch.zeros_like(draft_control)
        distance = ((draft_control - teacher_control) / max(control_scale_m_s, 1e-8)).pow(2).sum(dim=-1)[0]
        model_agree = distance <= accept_threshold
        control_bound = torch.linalg.norm(draft_control[0], dim=-1) <= u_max_m_s
        accept_mask = model_agree & control_bound
        prefix_len = _longest_prefix(accept_mask)
        prefix_lengths.append(prefix_len)
        controls_to_append: List[torch.Tensor] = []
        if prefix_len > 0:
            controls_to_append.extend([draft_control[:, j, :] for j in range(prefix_len)])
        if prefix_len < horizon:
            controls_to_append.append(teacher_control[:, prefix_len, :])
            rejection_count += 1
        chunk_control = torch.stack(controls_to_append, dim=1).squeeze(2)
        chunk_state = rollout_dynamics(current_state, chunk_control, time_step_s)
        for step in range(chunk_control.shape[1]):
            accepted_controls.append(chunk_control[:, step, :])
            accepted_states.append(chunk_state[:, step + 1, :])
            state_context = torch.cat((state_context, chunk_state[:, step + 1 : step + 2, :]), dim=1)
            action_context = torch.cat((action_context, chunk_control[:, step : step + 1, :]), dim=1)
            spent = torch.linalg.norm(chunk_control[:, step : step + 1, :], dim=-1, keepdim=True)
            fuel_context = torch.cat((fuel_context, fuel_context[:, -1:, :] - spent), dim=1)
            constraint_context = torch.cat((constraint_context, constraint_context[:, -1:, :]), dim=1)
        current_state = accepted_states[-1]
        k += chunk_control.shape[1]
    elapsed = time.perf_counter() - start
    return SpeculativeResult(
        warm_start_states=torch.stack(accepted_states, dim=1).squeeze(2),
        warm_start_controls=torch.stack(accepted_controls, dim=1).squeeze(2),
        accepted_prefix_lengths=prefix_lengths,
        rejection_count=rejection_count,
        full_art_call_count=full_art_call_count,
        timing_dictionary={"speculative_inference_time_s": elapsed},
    )
