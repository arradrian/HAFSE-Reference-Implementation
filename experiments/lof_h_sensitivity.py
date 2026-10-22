"""Sensitivity of the LOF neighborhood size h (Sec. 4.3.2).

h is selected on KNOWN-CLASS validation data only: for each h, LOF is fitted on
known-class training observations and its false-alarm rate is measured on
known-class validation observations, together with a validation AUROC against
synthetic out-of-range observations (no withheld-class data are used).

Usage:  python experiments/lof_h_sensitivity.py --synthetic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml

cfg = yaml.safe_load(open(ROOT / "config" / "default_config.yaml", encoding="utf-8"))
H_VALUES = cfg["online"]["lof_h_sensitivity"]


def evaluate_h(x_train, x_val, x_surrogate, h):
    lof = LocalOutlierFactor(n_neighbors=h, novelty=True).fit(x_train)
    far = float(np.mean(lof.predict(x_val) == -1))
    scores = -lof.score_samples(np.vstack([x_val, x_surrogate]))
    y = np.r_[np.zeros(len(x_val)), np.ones(len(x_surrogate))]
    return far, float(roc_auc_score(y, scores))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if not args.synthetic:
        raise SystemExit("Provide known-class training/validation observations.")
    rng = np.random.default_rng(0)
    x_train, x_val = rng.normal(size=(1000, 6)), rng.normal(size=(300, 6))
    x_sur = rng.uniform(4, 6, size=(100, 6))   # synthetic out-of-range observations
    print("h   false-alarm rate   validation AUROC   (synthetic data)")
    for h in H_VALUES:
        far, auc = evaluate_h(x_train, x_val, x_sur, h)
        print(f"{h:<3} {far:>16.3f} {auc:>18.3f}")
