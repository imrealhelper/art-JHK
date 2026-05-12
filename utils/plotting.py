"""Matplotlib plotting utilities for draft ART comparisons."""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_runtime_comparison(runtime_means: Mapping[str, float], output_dir: str) -> str:
    """Create a runtime bar plot.

    Args:
        runtime_means: Mapping of method name to mean runtime in seconds.
        output_dir: Plot directory.

    Returns:
        Path to ``runtime_comparison.png``.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "runtime_comparison.png")
    labels = list(runtime_means.keys())
    values = [float(runtime_means[label]) for label in labels]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, values)
    plt.ylabel("Mean runtime [s]")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_metric_bar(values: Mapping[str, float], ylabel: str, filename: str, output_dir: str) -> str:
    """Create a generic metric bar plot.

    Args:
        values: Mapping of method name to scalar value.
        ylabel: Y-axis label with units.
        filename: Output file name.
        output_dir: Plot directory.

    Returns:
        Saved plot path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    labels = list(values.keys())
    metric_values = [float(values[label]) for label in labels]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, metric_values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_prefix_histogram(prefix_lengths: Sequence[int], output_dir: str) -> Optional[str]:
    """Create accepted prefix length histogram for speculative inference.

    Args:
        prefix_lengths: Accepted prefix lengths in control steps.
        output_dir: Plot directory.

    Returns:
        Saved path, or None if no prefix lengths are available.
    """
    if not prefix_lengths:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accepted_prefix_histogram.png")
    plt.figure(figsize=(6, 4))
    plt.hist(prefix_lengths, bins=range(0, max(prefix_lengths) + 2), align="left")
    plt.xlabel("Accepted prefix length [steps]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_example_trajectory(
    trajectories: Mapping[str, np.ndarray], output_dir: str
) -> Optional[str]:
    """Plot representative RTN position trajectories.

    Args:
        trajectories: Mapping of method name to RTN states with shape ``[N + 1, 6]``
            or ``[6, N + 1]``. Position units are ``[m]``.
        output_dir: Plot directory.

    Returns:
        Saved path, or None if no trajectories are provided.
    """
    if not trajectories:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "example_trajectory_comparison.png")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for label, states in trajectories.items():
        arr = np.asarray(states, dtype=float)
        if arr.ndim != 2:
            continue
        if arr.shape[-1] != 6 and arr.shape[0] == 6:
            arr = arr.T
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], label=label)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("T [m]")
    ax.set_zlabel("N [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
