"""Loss functions and differentiable rollout for PhysicsInformedDraftART."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F


@dataclass
class DraftLossConfig:
    """Configurable loss weights and normalizers."""

    lambda_imit: float = 1.0
    lambda_distill: float = 1.0
    lambda_control: float = 0.5
    lambda_fuel: float = 0.2
    lambda_koz: float = 1.0
    lambda_cone: float = 0.5
    lambda_reach: float = 0.0
    lambda_terminal: float = 1.0
    lambda_accept: float = 0.1
    control_scale_m_s: float = 0.1
    fuel_scale_m_s: float = 1.0
    position_scale_m: float = 100.0
    velocity_scale_m_s: float = 0.1
    u_max_m_s: float = 10.0
    koz_margin: float = 0.0
    accept_threshold: float = 1.0
    accept_temperature: float = 0.1


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(dtype=values.dtype, device=values.device)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def imitation_loss(
    draft_control_dv: torch.Tensor,
    expert_control_dv: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    control_scale_m_s: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    """Imitation loss against expert/SCP controls.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` predicted RTN impulses in ``[m/s]``.
        expert_control_dv: ``float32[B, K, 3]`` expert RTN impulses in ``[m/s]``.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        control_scale_m_s: Normalizer for dimensional residuals in ``[m/s]``.
        loss_type: ``huber`` or ``l1``.

    Returns:
        Scalar ``float32`` normalized imitation loss.
    """
    residual = (draft_control_dv - expert_control_dv) / max(control_scale_m_s, 1e-8)
    if loss_type == "l1":
        values = residual.abs().sum(dim=-1)
    else:
        values = F.huber_loss(residual, torch.zeros_like(residual), reduction="none").sum(dim=-1)
    return _masked_mean(values, loss_mask)


def distillation_loss(
    draft_control_dv: torch.Tensor,
    teacher_control_dv: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    control_scale_m_s: float,
    teacher_covariance: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Distill draft controls from a frozen ART teacher.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` draft controls in ``[m/s]``.
        teacher_control_dv: ``float32[B, K, 3]`` ART mean controls in ``[m/s]``.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        control_scale_m_s: Euclidean residual normalizer in ``[m/s]``.
        teacher_covariance: optional ``float32[B, K, 3, 3]`` covariance in
            normalized control units for Mahalanobis distillation.
        epsilon: Positive jitter for covariance inversion.

    Returns:
        Scalar normalized distillation loss.
    """
    residual = (draft_control_dv - teacher_control_dv) / max(control_scale_m_s, 1e-8)
    if teacher_covariance is None:
        values = residual.pow(2).sum(dim=-1)
    else:
        eye = torch.eye(3, device=draft_control_dv.device, dtype=draft_control_dv.dtype)
        cov = teacher_covariance + epsilon * eye
        maha = torch.einsum("bki,bkij,bkj->bk", residual, torch.linalg.inv(cov), residual)
        logdet = torch.linalg.slogdet(cov).logabsdet
        values = 0.5 * (maha + logdet)
    return _masked_mean(values, loss_mask)


def control_bound_loss(
    draft_control_dv: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    u_max_m_s: float,
) -> torch.Tensor:
    """Penalize impulses exceeding a maximum magnitude.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` RTN impulses in ``[m/s]``.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        u_max_m_s: Maximum allowed impulse magnitude in ``[m/s]``.

    Returns:
        Scalar squared normalized bound violation.
    """
    norms = torch.linalg.norm(draft_control_dv, dim=-1)
    values = F.relu((norms - u_max_m_s) / max(u_max_m_s, 1e-8)).pow(2)
    return _masked_mean(values, loss_mask)


def fuel_consistency_loss(
    draft_control_dv: torch.Tensor,
    fuel_to_go: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    fuel_scale_m_s: float,
) -> torch.Tensor:
    """Ensure predicted chunk fuel does not exceed available fuel-to-go.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` RTN impulses in ``[m/s]``.
        fuel_to_go: ``float32[B, 1]`` positive ``phi_k`` in ``[m/s]``.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        fuel_scale_m_s: Fuel normalizer in ``[m/s]``.

    Returns:
        Scalar squared normalized fuel violation.
    """
    if loss_mask is None:
        weights = torch.ones(draft_control_dv.shape[:2], device=draft_control_dv.device)
    else:
        weights = loss_mask.to(draft_control_dv.dtype)
    chunk_fuel = (torch.linalg.norm(draft_control_dv, dim=-1) * weights).sum(dim=-1, keepdim=True)
    values = F.relu((chunk_fuel - fuel_to_go) / max(fuel_scale_m_s, 1e-8)).pow(2)
    return values.mean()


def rollout_dynamics(
    initial_state: torch.Tensor,
    control_dv: torch.Tensor,
    time_step_s: float,
    state_transition_matrices: Optional[torch.Tensor] = None,
    control_input_matrices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Roll out controls through differentiable approximate dynamics.

    Purpose:
        Use provided repository linearized STM/CIM tensors when available. If not
        available, use a differentiable RTN constant-velocity approximation for
        training; repository NumPy dynamics remain the preferred evaluation path.

    Args:
        initial_state: ``float32[B, 6]`` initial RTN/ROE state. RTN units are
            position ``[m]`` and velocity ``[m/s]``.
        control_dv: ``float32[B, K, 3]`` impulsive controls in RTN ``[m/s]``.
        time_step_s: Discrete step duration ``Delta t`` in ``[s]``.
        state_transition_matrices: optional ``float32[B, K, 6, 6]`` STM.
        control_input_matrices: optional ``float32[B, K, 6, 3]`` CIM.

    Returns:
        ``float32[B, K + 1, 6]`` rolled states in the same state representation
        as ``initial_state``.
    """
    states = [initial_state]
    state = initial_state
    for step in range(control_dv.shape[1]):
        if state_transition_matrices is not None and control_input_matrices is not None:
            stm = state_transition_matrices[:, step]
            cim = control_input_matrices[:, step]
            state = torch.bmm(stm, (state + torch.bmm(cim, control_dv[:, step].unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        else:
            next_state = state.clone()
            next_state[:, :3] = state[:, :3] + state[:, 3:] * time_step_s
            next_state[:, 3:] = state[:, 3:] + control_dv[:, step]
            state = next_state
        states.append(state)
    return torch.stack(states, dim=1)


def koz_loss(
    rolled_state_rtn: torch.Tensor,
    koz_matrix: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    koz_active_mask: Optional[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    """Keep-out-zone ellipsoid loss.

    Args:
        rolled_state_rtn: ``float32[B, K + 1, 6]`` RTN states; positions in ``[m]``.
        koz_matrix: ``float32[3, 3]`` ellipsoid matrix ``D^{-2}`` in ``[1/m^2]``.
        loss_mask: optional ``bool[B, K]`` valid-step mask.
        koz_active_mask: required ``bool[B, K]`` mask for active KOZ steps.
        margin: Dimensionless conservative safety margin.

    Returns:
        Scalar dimensionless KOZ loss. Returns zero if mask is not provided.
    """
    if koz_active_mask is None:
        return rolled_state_rtn.sum() * 0.0
    position = rolled_state_rtn[:, 1:, :3]
    safety = torch.einsum("bki,ij,bkj->bk", position, koz_matrix.to(position), position) - 1.0
    values = F.relu(margin - safety).pow(2)
    mask = koz_active_mask if loss_mask is None else (koz_active_mask.bool() & loss_mask.bool())
    return _masked_mean(values, mask)


def cone_loss(
    rolled_state_rtn: torch.Tensor,
    dock_port_rtn: torch.Tensor,
    dock_axis: torch.Tensor,
    cone_half_angle_rad: float,
    loss_mask: Optional[torch.Tensor],
    cone_active_mask: Optional[torch.Tensor],
    position_scale_m: float,
) -> torch.Tensor:
    """Approach cone violation loss.

    Args:
        rolled_state_rtn: ``float32[B, K + 1, 6]`` RTN states; positions in ``[m]``.
        dock_port_rtn: ``float32[3]`` docking port position in ``[m]``.
        dock_axis: ``float32[3]`` docking axis unit vector.
        cone_half_angle_rad: Cone half-angle in ``[rad]``.
        loss_mask: optional ``bool[B, K]`` valid-step mask.
        cone_active_mask: optional ``bool[B, K]`` active cone mask.
        position_scale_m: Position normalizer in ``[m]``.

    Returns:
        Scalar normalized cone loss. Returns zero when inactive.
    """
    if cone_active_mask is None:
        return rolled_state_rtn.sum() * 0.0
    position = rolled_state_rtn[:, 1:, :3]
    axis = dock_axis.to(position)
    rho = position - dock_port_rtn.to(position).view(1, 1, 3)
    z = torch.einsum("bki,i->bk", rho, axis)
    perp = rho - z.unsqueeze(-1) * axis.view(1, 1, 3)
    values = F.relu((torch.linalg.norm(perp, dim=-1) - z * torch.tan(torch.tensor(cone_half_angle_rad, device=position.device))) / max(position_scale_m, 1e-8)).pow(2)
    mask = cone_active_mask if loss_mask is None else (cone_active_mask.bool() & loss_mask.bool())
    return _masked_mean(values, mask)


def terminal_loss(
    rolled_state: torch.Tensor,
    target_state: torch.Tensor,
    terminal_mask: Optional[torch.Tensor],
    position_scale_m: float,
    velocity_scale_m_s: float,
) -> torch.Tensor:
    """Terminal state loss for chunks that include terminal time.

    Args:
        rolled_state: ``float32[B, K + 1, 6]`` states. RTN units are ``[m, m/s]``.
        target_state: ``float32[B, 6]`` target states in same units.
        terminal_mask: optional ``bool[B]`` True when the chunk reaches ``N``.
        position_scale_m: Position normalizer in ``[m]``.
        velocity_scale_m_s: Velocity normalizer in ``[m/s]``.

    Returns:
        Scalar normalized terminal error.
    """
    residual = rolled_state[:, -1] - target_state
    scale = torch.tensor(
        [position_scale_m] * 3 + [velocity_scale_m_s] * 3,
        device=rolled_state.device,
        dtype=rolled_state.dtype,
    )
    values = (residual / scale.clamp_min(1e-8)).pow(2).sum(dim=-1)
    return _masked_mean(values, terminal_mask)


def acceptance_proxy_loss(
    draft_control_dv: torch.Tensor,
    teacher_control_dv: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    control_scale_m_s: float,
    accept_threshold: float,
    accept_temperature: float,
) -> torch.Tensor:
    """Differentiable verifier-acceptance proxy.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` draft controls in ``[m/s]``.
        teacher_control_dv: ``float32[B, K, 3]`` full ART verifier controls in ``[m/s]``.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        control_scale_m_s: Control normalizer in ``[m/s]``.
        accept_threshold: Squared normalized agreement threshold.
        accept_temperature: Positive softplus temperature.

    Returns:
        Scalar normalized acceptance proxy loss.
    """
    distance = ((draft_control_dv - teacher_control_dv) / max(control_scale_m_s, 1e-8)).pow(2).sum(dim=-1)
    values = F.softplus((distance - accept_threshold) / max(accept_temperature, 1e-8))
    return _masked_mean(values, loss_mask)


def total_draft_loss(
    draft_control_dv: torch.Tensor,
    expert_control_dv: torch.Tensor,
    config: DraftLossConfig,
    loss_mask: Optional[torch.Tensor] = None,
    teacher_control_dv: Optional[torch.Tensor] = None,
    initial_state_rtn: Optional[torch.Tensor] = None,
    fuel_to_go: Optional[torch.Tensor] = None,
    use_physics_loss: bool = False,
    use_distillation: bool = False,
    use_acceptance_loss: bool = False,
    time_step_s: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Combine configured draft losses into a weighted objective.

    Args:
        draft_control_dv: ``float32[B, K, 3]`` draft controls in ``[m/s]``.
        expert_control_dv: ``float32[B, K, 3]`` expert controls in ``[m/s]``.
        config: Loss weights and normalizers.
        loss_mask: optional ``bool[B, K]`` valid-control mask.
        teacher_control_dv: optional ``float32[B, K, 3]`` ART controls in ``[m/s]``.
        initial_state_rtn: optional ``float32[B, 6]`` RTN initial state in ``[m, m/s]``.
        fuel_to_go: optional ``float32[B, 1]`` positive fuel-to-go in ``[m/s]``.
        use_physics_loss: Enable control/fuel physics-informed losses.
        use_distillation: Enable teacher distillation loss.
        use_acceptance_loss: Enable acceptance proxy loss.
        time_step_s: Dynamics step in ``[s]`` for optional rollout users.

    Returns:
        Dictionary containing scalar component losses and ``total``.
    """
    del initial_state_rtn, time_step_s
    losses: Dict[str, torch.Tensor] = {}
    losses["imitation"] = imitation_loss(
        draft_control_dv, expert_control_dv, loss_mask, config.control_scale_m_s
    )
    total = config.lambda_imit * losses["imitation"]
    if use_distillation and teacher_control_dv is not None:
        losses["distillation"] = distillation_loss(
            draft_control_dv, teacher_control_dv, loss_mask, config.control_scale_m_s
        )
        total = total + config.lambda_distill * losses["distillation"]
    if use_physics_loss:
        losses["control_bound"] = control_bound_loss(draft_control_dv, loss_mask, config.u_max_m_s)
        total = total + config.lambda_control * losses["control_bound"]
        if fuel_to_go is not None:
            losses["fuel"] = fuel_consistency_loss(
                draft_control_dv, fuel_to_go, loss_mask, config.fuel_scale_m_s
            )
            total = total + config.lambda_fuel * losses["fuel"]
    if use_acceptance_loss and teacher_control_dv is not None:
        losses["acceptance"] = acceptance_proxy_loss(
            draft_control_dv,
            teacher_control_dv,
            loss_mask,
            config.control_scale_m_s,
            config.accept_threshold,
            config.accept_temperature,
        )
        total = total + config.lambda_accept * losses["acceptance"]
    losses["total"] = total
    return losses
