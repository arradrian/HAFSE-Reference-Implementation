"""Open-set recognition metrics used in the systematic evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc_known_unknown(y_unknown_true, novelty_score) -> float:
    """AUROC with unknown class encoded as 1."""
    return float(roc_auc_score(np.asarray(y_unknown_true), np.asarray(novelty_score)))


def unknown_recall(y_unknown_true, y_unknown_pred) -> float:
    y_t = np.asarray(y_unknown_true, dtype=int)
    y_p = np.asarray(y_unknown_pred, dtype=int)
    denom = np.sum(y_t == 1)
    return float(np.sum((y_t == 1) & (y_p == 1)) / denom) if denom else np.nan


def false_unknown_rate(y_unknown_true, y_unknown_pred) -> float:
    """Fraction of known observations incorrectly rejected as unknown."""
    y_t = np.asarray(y_unknown_true, dtype=int)
    y_p = np.asarray(y_unknown_pred, dtype=int)
    denom = np.sum(y_t == 0)
    return float(np.sum((y_t == 0) & (y_p == 1)) / denom) if denom else np.nan


def event_detection_rate(event_detected: np.ndarray) -> float:
    """Fraction of true unknown events that triggered a novel-event decision."""
    arr = np.asarray(event_detected, dtype=bool)
    return float(np.mean(arr)) if arr.size else np.nan


def mean_detection_delay(delays) -> float:
    vals = np.asarray([d for d in delays if d is not None and np.isfinite(d)], dtype=float)
    return float(np.mean(vals)) if vals.size else np.nan


def oscr_curve(
    known_confidence: np.ndarray,
    known_correct: np.ndarray,
    unknown_confidence: np.ndarray,
):
    """Compute an OSCR curve and area.

    Convention:
    - ``known_confidence``: confidence that a known sample belongs to its predicted
      known class (larger = more confidently known).
    - ``known_correct``: 1 if the known sample was correctly classified.
    - ``unknown_confidence``: maximum known-class confidence assigned to unknown samples.

    For threshold t:
      CCR(t) = correctly classified known samples with confidence >= t / #known
      FPR(t) = unknown samples with confidence >= t / #unknown

    OSCR is the area under CCR versus FPR.
    """
    kc = np.asarray(known_confidence, dtype=float)
    corr = np.asarray(known_correct, dtype=bool)
    uc = np.asarray(unknown_confidence, dtype=float)

    thresholds = np.unique(np.concatenate(([np.inf], kc, uc, [-np.inf])))[::-1]
    ccr, fpr = [], []
    for t in thresholds:
        ccr.append(np.mean(corr & (kc >= t)) if kc.size else np.nan)
        fpr.append(np.mean(uc >= t) if uc.size else np.nan)

    fpr = np.asarray(fpr, dtype=float)
    ccr = np.asarray(ccr, dtype=float)
    order = np.argsort(fpr)
    fpr, ccr = fpr[order], ccr[order]
    area = float(np.trapz(ccr, fpr))
    return fpr, ccr, area


def summarize_runs(run_metrics: list[dict]) -> dict:
    """Return mean and sample standard deviation for scalar metrics across runs."""
    keys = sorted(set().union(*(m.keys() for m in run_metrics)))
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in run_metrics if k in m and np.isfinite(m[k])], float)
        if vals.size:
            out[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            }
    return out
