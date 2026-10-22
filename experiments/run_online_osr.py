"""Systematic open-set evaluation (Sec. 4.3.2 and 5.2.2).

Protocol
--------
* Each online stream has 500 observations and is built from CONTIGUOUS
  segments of real observations of the known classes and of the withheld
  (unknown) class. The temporal order inside each segment is preserved; only
  the selection of segments and their placement in the stream are randomized.
  Individual observations are never shuffled.
* Five stream realizations are generated (seeds 0-4). When a class has few
  observations, realizations necessarily share observations, so they are
  randomized realizations rather than independent test sets.
* Sample-level metrics: AUROC, U-Recall, FUR (and OSCR via src.metrics).
* Event-level metrics: EDR and detection delay, computed with Algorithm 2
  (tumbling windows of w observations, decision at window closure), and the
  sample-level latency to the first observation flagged as anomalous.

Usage:  python experiments/run_online_osr.py --synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (auroc_known_unknown, event_detection_rate, false_unknown_rate,
                         mean_detection_delay, summarize_runs, unknown_recall)
from src.online_detection import (delay_bounds, detection_delay, first_flag_latency,
                                  process_online_stream)
from src.reproducibility import REFERENCE_SEEDS

N_SAMPLES_PER_STREAM = 500


def build_segment_stream(known_runs, unknown_runs, length=N_SAMPLES_PER_STREAM,
                         n_unknown_segments=1, unknown_segment_length=150,
                         known_segment_length=(50, 150), seed=0):
    """Compose a stream from contiguous segments with randomized placement.

    Returns x (length x d), y_unknown (1 for withheld-class observations) and
    the list of unknown events as (onset_index, end_index).
    """
    rng = np.random.default_rng(seed)

    def segment(runs, n):
        r = runs[rng.integers(len(runs))]
        n = min(n, len(r))
        start = rng.integers(0, len(r) - n + 1)
        return r[start:start + n]               # contiguous, order preserved

    blocks = [(segment(unknown_runs, unknown_segment_length), 1) for _ in range(n_unknown_segments)]
    n_unknown = sum(len(b) for b, _ in blocks)
    remaining = length - n_unknown
    while remaining > 0:
        n = min(remaining, int(rng.integers(*known_segment_length)))
        blocks.append((segment(known_runs, n), 0))
        remaining -= len(blocks[-1][0])
    order = rng.permutation(len(blocks))        # randomized placement of segments
    xs, ys, events, pos = [], [], [], 0
    for k in order:
        b, lab = blocks[k]
        if lab == 1:
            events.append((pos, pos + len(b) - 1))
        xs.append(b)
        ys.append(np.full(len(b), lab))
        pos += len(b)
    return np.concatenate(xs)[:length], np.concatenate(ys)[:length], events


def evaluate_stream(x, y_unknown, events, lof, classifier, w, thcr_percent, h_stage2=5):
    novelty_score = -lof.score_samples(x)       # larger = more anomalous
    y_pred = (lof.predict(x) == -1).astype(int)
    decisions = process_online_stream(x, lof, classifier, w, thcr_percent, h_stage2=h_stage2)
    delays = [detection_delay(decisions, on, end) for on, end in events]
    latencies = [first_flag_latency(y_pred, on, end) for on, end in events]
    return {
        "AUROC": auroc_known_unknown(y_unknown, novelty_score),
        "U_Recall": unknown_recall(y_unknown, y_pred),
        "FUR": false_unknown_rate(y_unknown, y_pred),
        "EDR": event_detection_rate([d is not None for d in delays]),
        "Delay": mean_detection_delay(delays),                 # onset -> NE decision
        "FirstFlagLatency": mean_detection_delay(latencies),   # onset -> first flagged sample
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--w", type=int, default=50)
    ap.add_argument("--thcr", type=float, default=60.0)
    args = ap.parse_args()
    if not args.synthetic:
        raise SystemExit("Provide project-specific known/unknown runs, LOF detector and classifier.")

    rng = np.random.default_rng(0)
    known_runs = [rng.normal(0, 1, (300, 6)) + c for c in (0, 3) for _ in range(5)]
    known_labels = [c for c in (0, 1) for _ in range(5)]
    unknown_runs = [rng.normal(0, 0.6, (300, 6)) + 7 for _ in range(3)]
    x_known = np.concatenate(known_runs)
    y_known = np.repeat(known_labels, 300)
    lof = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(x_known)
    clf = LogisticRegression(max_iter=500).fit(x_known, y_known)

    runs = []
    for seed in REFERENCE_SEEDS:
        x, y, ev = build_segment_stream(known_runs, unknown_runs, seed=seed)
        runs.append(evaluate_stream(x, y, ev, lof, clf, args.w, args.thcr))
    print("Synthetic demonstration (not a manuscript result):")
    for k, v in summarize_runs(runs).items():
        print(f"  {k:<9} {v['mean']:.3f} ± {v['std']:.3f}")
    lo, hi, mean = delay_bounds(args.w, args.thcr)
    print(f"Analytical delay bounds for w={args.w}, ThCr={args.thcr}%: "
          f"min={lo}, max={hi}, mean={mean:.1f} samples")


if __name__ == "__main__":
    main()
