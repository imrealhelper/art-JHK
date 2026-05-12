"""Metric computation and serialization for draft ART evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from optimization.rpod_scenario import EE_koz, dock_wyp_sample, state_rtn_target


def fuel_cost(control_dv: np.ndarray) -> float:
    """Compute trajectory fuel cost.

    Args:
        control_dv: Array with shape ``[N, 3]`` or ``[3, N]`` in ``[m/s]``.

    Returns:
        Sum of per-step Euclidean impulse magnitudes in ``[m/s]``.
    """
    controls = np.asarray(control_dv, dtype=float)
    if controls.ndim != 2:
        return float("nan")
    if controls.shape[-1] != 3 and controls.shape[0] == 3:
        controls = controls.T
    return float(np.linalg.norm(controls, axis=-1).sum())


def terminal_errors(
    state_rtn: np.ndarray, target_rtn: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute terminal RTN position and velocity errors.

    Args:
        state_rtn: RTN state array with shape ``[N + 1, 6]`` or ``[6, N + 1]``;
            positions are in ``[m]`` and velocities in ``[m/s]``.
        target_rtn: Optional target state with shape ``[6]`` in RTN units.

    Returns:
        Dictionary with ``terminal_position_error_m`` and
        ``terminal_velocity_error_m_s``.
    """
    states = np.asarray(state_rtn, dtype=float)
    if states.ndim != 2:
        return {"terminal_position_error_m": float("nan"), "terminal_velocity_error_m_s": float("nan")}
    if states.shape[-1] != 6 and states.shape[0] == 6:
        states = states.T
    target = np.asarray(state_rtn_target if target_rtn is None else target_rtn, dtype=float).reshape(6)
    err = states[-1] - target
    return {
        "terminal_position_error_m": float(np.linalg.norm(err[:3])),
        "terminal_velocity_error_m_s": float(np.linalg.norm(err[3:])),
    }


def koz_violation(state_rtn: np.ndarray) -> Dict[str, float]:
    """Compute keep-out-zone violation metrics for the repository ellipsoid.

    Args:
        state_rtn: RTN state array with shape ``[N + 1, 6]`` or ``[6, N + 1]``;
            positions are in ``[m]``. The KOZ is inactive after the repository
            docking waypoint sample.

    Returns:
        Dictionary with maximum and sum KOZ violation. Values are dimensionless.
    """
    states = np.asarray(state_rtn, dtype=float)
    if states.ndim != 2:
        return {"koz_max_violation": float("nan"), "koz_sum_violation": float("nan")}
    if states.shape[-1] != 6 and states.shape[0] == 6:
        states = states.T
    active = states[: min(dock_wyp_sample, states.shape[0]), :3]
    if active.size == 0:
        return {"koz_max_violation": 0.0, "koz_sum_violation": 0.0}
    safety = np.einsum("bi,ij,bj->b", active, EE_koz, active) - 1.0
    violation = np.maximum(0.0, -safety)
    return {"koz_max_violation": float(violation.max()), "koz_sum_violation": float(violation.sum())}


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Average numeric fields across evaluation records.

    Args:
        records: Iterable of dictionaries containing scalar metrics.

    Returns:
        Dictionary with ``*_mean`` and ``*_std`` keys for numeric inputs.
    """
    rows = list(records)
    keys = sorted({key for row in rows for key in row.keys()})
    summary: Dict[str, float] = {}
    for key in keys:
        vals: List[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                vals.append(float(value))
        finite = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
        if finite.size:
            summary[f"{key}_mean"] = float(finite.mean())
            summary[f"{key}_std"] = float(finite.std())
    return summary


def save_metrics(metrics: Mapping[str, Any], output_dir: str) -> None:
    """Save metrics as JSON and a flat CSV table.

    Args:
        metrics: JSON-serializable metrics dictionary.
        output_dir: Destination directory.

    Returns:
        None. Writes ``metrics.json`` and ``metrics.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
