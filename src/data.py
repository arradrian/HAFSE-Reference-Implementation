"""Data partitioning, normalization and window extraction.

Protocol (Sec. 4.1.1 of the manuscript):

1. Partition the data at the level of complete simulation runs (trajectories)
   BEFORE any window extraction: 70% of the runs of each class for training,
   10% for validation and 20% for testing. Windows originating from the same
   trajectory therefore never appear in more than one subset.
2. Estimate normalization parameters (mean, standard deviation) on the
   training runs only and apply them to all subsets.
3. Extract windows of T consecutive observations with stride s from each run.

Each run is given as an array of shape (n_observations, d) together with an
integer class label, or with a per-observation label array.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunSplit:
    train: list[int]
    val: list[int]
    test: list[int]


def split_runs(run_labels, fractions=(0.7, 0.1, 0.2), seed: int = 0) -> RunSplit:
    """Stratified run-level split. ``run_labels`` holds one class label per run."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1.")
    rng = np.random.default_rng(seed)
    run_labels = np.asarray(run_labels)
    train, val, test = [], [], []
    for c in np.unique(run_labels):
        idx = rng.permutation(np.flatnonzero(run_labels == c))
        n_tr = int(round(fractions[0] * len(idx)))
        n_va = int(round(fractions[1] * len(idx)))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        test += idx[n_tr + n_va:].tolist()
    return RunSplit(sorted(train), sorted(val), sorted(test))


class TrainOnlyScaler:
    """Z-score normalization fitted on training runs only."""

    def fit(self, runs):
        x = np.concatenate([np.asarray(r, float) for r in runs], axis=0)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, run):
        return (np.asarray(run, float) - self.mean_) / self.std_


def extract_windows(run, labels, T: int, stride: int, label_rule: str = "last"):
    """Windows of T consecutive observations with the given stride.

    ``labels`` is a scalar (one label per run) or an array with one label per
    observation. With ``label_rule='last'`` each window takes the label of its
    last observation.
    """
    run = np.asarray(run)
    per_obs = np.ndim(labels) > 0
    xs, ys = [], []
    for start in range(0, len(run) - T + 1, stride):
        xs.append(run[start:start + T])
        if per_obs:
            lab = labels[start + T - 1] if label_rule == "last" else np.bincount(labels[start:start + T]).argmax()
        else:
            lab = labels
        ys.append(lab)
    return np.asarray(xs), np.asarray(ys)


def build_datasets(runs, run_labels, T: int, stride: int, fractions=(0.7, 0.1, 0.2),
                   seed: int = 0, obs_labels=None):
    """Run-level split -> train-only normalization -> windowing.

    Returns dict with (X, y) for 'train', 'val' and 'test', the split and scaler.
    """
    split = split_runs(run_labels, fractions, seed)
    scaler = TrainOnlyScaler().fit([runs[i] for i in split.train])
    out = {"split": split, "scaler": scaler}
    for name, ids in (("train", split.train), ("val", split.val), ("test", split.test)):
        xs, ys = [], []
        for i in ids:
            lab = obs_labels[i] if obs_labels is not None else run_labels[i]
            x, y = extract_windows(scaler.transform(runs[i]), lab, T, stride)
            if len(x):
                xs.append(x)
                ys.append(y)
        out[name] = (np.concatenate(xs), np.concatenate(ys)) if xs else (np.empty(0), np.empty(0))
    return out
