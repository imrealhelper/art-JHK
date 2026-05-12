"""Reproducibility helpers for physics-informed draft ART experiments."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and Torch seeds.

    Args:
        seed: Integer random seed.
        deterministic: If True, request deterministic cuDNN kernels where practical.

    Returns:
        None. All operations are side effects on global RNG state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_repository_commit_hash(repo_root: str) -> str:
    """Return the current Git commit hash for metadata logging.

    Args:
        repo_root: Repository root path.

    Returns:
        Git commit hash string, or ``unknown`` if Git is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def collect_run_metadata(
    *,
    seed: int,
    device: torch.device,
    repo_root: str,
    dataset_path: str,
    train_samples: int,
    eval_samples: int,
    model_parameter_count: int,
    full_art_checkpoint: Optional[str],
    draft_checkpoint: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect reproducibility metadata for a run.

    Args:
        seed: Random seed.
        device: Torch device used for computation.
        repo_root: Repository root path.
        dataset_path: Dataset path or synthetic fallback identifier.
        train_samples: Number of training samples used.
        eval_samples: Number of evaluation samples used.
        model_parameter_count: Number of trainable plus frozen draft parameters.
        full_art_checkpoint: Optional full ART checkpoint path.
        draft_checkpoint: Optional draft checkpoint path.
        extra: Optional additional metadata.

    Returns:
        JSON-serializable metadata dictionary.
    """
    device_name = "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
    metadata: Dict[str, Any] = {
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "repository_commit_hash": get_repository_commit_hash(repo_root),
        "dataset_path": dataset_path,
        "train_test_split_rule": "first 90% train, final 10% eval for repository data; seeded synthetic fallback otherwise",
        "number_of_training_samples": train_samples,
        "number_of_evaluation_samples": eval_samples,
        "model_parameter_count": model_parameter_count,
        "full_art_checkpoint_path": full_art_checkpoint,
        "draft_checkpoint_path": draft_checkpoint,
        "solver_name": "repository SCP/CVX if available, otherwise not run",
        "solver_version": None,
        "cwd": os.getcwd(),
    }
    if extra:
        metadata.update(extra)
    return metadata
