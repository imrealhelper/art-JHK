"""Timing utilities for CPU and CUDA ART experiments."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before or after a timed region.

    Args:
        device: Torch device. CUDA synchronization is called only for CUDA devices.

    Returns:
        None.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def timed_block(name: str, device: torch.device, output: Dict[str, float]) -> Iterator[None]:
    """Measure a named code block with ``time.perf_counter``.

    Args:
        name: Metric key to write in seconds.
        device: Torch device, synchronized if CUDA.
        output: Mutable dictionary receiving ``name`` in seconds.

    Yields:
        None. The elapsed time is recorded on context exit.
    """
    synchronize_if_cuda(device)
    start = time.perf_counter()
    yield
    synchronize_if_cuda(device)
    output[name] = time.perf_counter() - start
