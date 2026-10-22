"""Synthetic multivariate time series used ONLY by the demos and tests.

These data do not reproduce any result of the manuscript. They allow the
complete workflow to be executed without the project datasets.
"""
from __future__ import annotations

import numpy as np


def make_runs(n_classes: int, runs_per_class: int, length: int, d: int, seed: int = 0):
    """Return (runs, run_labels): class-dependent sinusoids with noise and drifts."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    runs, labels = [], []
    for c in range(n_classes):
        freq = 1.0 / (20 + 7 * c)
        for _ in range(runs_per_class):
            phase = rng.uniform(0, 2 * np.pi, d)
            base = np.sin(2 * np.pi * freq * t[:, None] + phase)
            drift = (c % 3) * 0.01 * t[:, None] * rng.normal(1, 0.1, d)
            runs.append(base + drift + 0.2 * rng.normal(size=(length, d)))
            labels.append(c)
    return runs, np.asarray(labels)
