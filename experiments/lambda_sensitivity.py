"""Sensitivity of the regularization coefficient lambda (Sec. 3.5.3).

For each lambda in the configured set, the final HAFSE model with a fixed
structural configuration (N, M) is trained and its validation accuracy is
reported. With --synthetic the workflow runs on synthetic data.

Usage:  python experiments/lambda_sensitivity.py --synthetic --N 2 --M 3
"""
import argparse
import sys
from pathlib import Path

import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_datasets
from src.hafse import compile_model
from src.models import build_source_model, build_target_model
from src.reproducibility import set_global_seed
from src.synthetic import make_runs

cfg = yaml.safe_load(open(ROOT / "config" / "default_config.yaml", encoding="utf-8"))


def to_tf(x, y, c):
    return tf.data.Dataset.from_tensor_slices((x.astype("float32"), tf.one_hot(y, c))).batch(64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    if not args.synthetic:
        raise SystemExit("Provide project-specific source and target data.")
    set_global_seed(0)
    T, stride = 32, 8
    sr, sl = make_runs(6, 20, 120, 12, seed=0)
    tr, tl = make_runs(4, 10, 120, 6, seed=1)
    S, D = build_datasets(sr, sl, T, stride), build_datasets(tr, tl, T, stride)
    source = build_source_model(T, 12, 6)
    source.compile(optimizer="adam", loss="categorical_crossentropy")
    source.fit(to_tf(*S["train"], 6), epochs=2, verbose=0)
    print("lambda    validation accuracy   (synthetic data)")
    for lam in cfg["hafse"]["lambda_sensitivity"]:
        set_global_seed(0)
        m = build_target_model(source, T, 6, 4, args.N, args.M,
                               cfg["architecture"]["expansion_units"])
        compile_model(m, source, lam, cfg["hafse"]["learning_rate"])
        m.fit(to_tf(*D["train"], 4), epochs=args.epochs, verbose=0)
        _, acc = m.evaluate(to_tf(*D["val"], 4), verbose=0)
        print(f"{lam:<9g} {acc:>18.3f}")
