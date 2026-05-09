"""Runtime utilities shared by AdaptiveStreamFL entry points."""

from __future__ import annotations

import random
import sys
from typing import Optional

import numpy as np


DATASET_FEATURES = {
    "covtype": 54,
    "kddcup99": 41,
    "mnist": 784,
    "adult": 14,
    "electricity": 8,
    "occupancy": 5,
    "shuttle": 9,
}


def configure_standard_streams() -> None:
    """Prefer UTF-8 console output when the host stream supports it."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def set_random_seed(seed: Optional[int]) -> None:
    """Set all random seeds used by this package when reproducibility is requested."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def on_off(value: object) -> str:
    """Format a boolean-like flag for command-line output."""
    return "on" if value else "off"
