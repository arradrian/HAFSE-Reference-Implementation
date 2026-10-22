"""Run HAFSE (Algorithm 1).

Usage
-----
    python experiments/run_hafse.py --synthetic            # quick end-to-end demo
    python experiments/run_hafse.py --synthetic --scheme joint --method exhaustive

To use the project datasets, replace ``load_domains`` with loaders returning
lists of runs (arrays n_obs x d) and one class label per run for the source
and target domains. TE and Two-Tank data are public; GPN data are available
from the authors of the corresponding reference upon request.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_datasets
from src.hafse import compile_model, run_hafse
from src.models import build_source_model, transfer_summary
from src.reproducibility import set_global_seed
from src.synthetic import make_runs


def load_domains(synthetic: bool):
    if not synthetic:
        raise SystemExit("Provide project-specific loaders in load_domains().")
    src = make_runs(n_classes=6, runs_per_class=20, length=120, d=12, seed=0)
    tgt = make_runs(n_classes=4, runs_per_class=10, length=120, d=6, seed=1)
    return src, tgt


def to_tf(x, y, n_classes, batch=64, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), tf.one_hot(y, n_classes)))
    return (ds.shuffle(len(x)) if shuffle else ds).batch(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "default_config.yaml", encoding="utf-8"))
    h = cfg["hafse"]
    set_global_seed(args.seed)

    (src_runs, src_lab), (tgt_runs, tgt_lab) = load_domains(args.synthetic)
    S = build_datasets(src_runs, src_lab, args.T, args.stride, seed=args.seed)
    D = build_datasets(tgt_runs, tgt_lab, args.T, args.stride, seed=args.seed)
    cs, ct = int(src_lab.max()) + 1, int(tgt_lab.max()) + 1
    d_src, d_tgt = src_runs[0].shape[1], tgt_runs[0].shape[1]

    source = build_source_model(args.T, d_src, cs)
    source.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    source.fit(to_tf(*S["train"], cs, shuffle=True), validation_data=to_tf(*S["val"], cs),
               epochs=3 if args.synthetic else 50, verbose=0)

    de_kwargs = dict(population_size=cfg["differential_evolution"]["population_size"],
                     crossover_probability=cfg["differential_evolution"]["crossover_probability"],
                     scaling_factor=cfg["differential_evolution"]["scaling_factor"],
                     max_iterations=2 if args.synthetic else cfg["differential_evolution"]["max_iterations"],
                     tolerance=cfg["differential_evolution"]["tolerance"])
    res = run_hafse(source, args.T, d_tgt, ct,
                    to_tf(*D["train"], ct, shuffle=True), to_tf(*D["val"], ct),
                    N_space=h["N_space"], M_space=h["M_space"], lambda_reg=h["lambda"],
                    learning_rate=h["learning_rate"],
                    epochs_search=1 if args.synthetic else h["epochs_search"],
                    epochs_final=2 if args.synthetic else h["epochs_final"],
                    expansion_units=cfg["architecture"]["expansion_units"],
                    scheme=args.scheme or h["scheme"], method=args.method or h["search_method"],
                    seed=args.seed, de_kwargs=de_kwargs,
                    criterion=h["selection_criterion"])

    print(f"N* = {res.N_star}, M* = {res.M_star}")
    for name, r in res.searches.items():
        print(f"  search {name}: best={r.value}, evaluations={r.n_evaluations}, "
              f"distinct candidates={r.n_unique_candidates}")
    print("\nLayer-wise transfer summary of the final model (Table X):")
    for row in transfer_summary(res.model):
        print(f"  {row['layer']:<22} params={row['params']:>8}  trainable={row['trainable']!s:<5}  {row['status']}")
    total = res.model.count_params()
    trainable = int(sum(np.prod(w.shape) for w in res.model.trainable_weights))
    print(f"Total parameters: {total}  (trainable: {trainable})")
    loss, acc = res.model.evaluate(to_tf(*D["test"], ct), verbose=0)
    print(f"Test accuracy (synthetic data, not a manuscript result): {acc:.3f}")


if __name__ == "__main__":
    main()
