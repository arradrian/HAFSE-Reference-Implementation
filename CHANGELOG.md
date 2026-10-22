# Changelog

## 2.0.0

Alignment of the implementation with the revised manuscript.

* **Architecture** (`src/models.py`): the backbone now follows Fig. 4 — three
  Conv1D blocks (64/128/256 filters, kernels 7/5/3) with batch normalization,
  pooling and dropout; three LSTM blocks (128/64/32) with batch normalization
  and dropout; Dense-128 head. Version 1.0.0 used a reduced backbone (two
  Conv1D layers and one LSTM layer).
* **Definition of N**: N is defined over the five transferred parametric layers
  (N=1: LSTM-3, …, N=5: + Conv1D-2). In 1.0.0 only three layers were exposed and
  N was capped at 3, so N = 3, 4, 5 produced the same configuration.
* **Input adapter**: the first Conv1D block is target-specific, randomly
  initialized and always trainable. In 1.0.0 it could remain frozen with random
  weights when N < 3.
* **Seeds**: a single reference seed set (0–4) is used everywhere, consistent
  with the manuscript. 1.0.0 listed [1, 7, 21, 42, 123] and used 42 by default.
* **Online decision rule**: PAnSa ≥ ThCr by default (Algorithm 2). 1.0.0 used ">".
* **Open-set streams**: streams are built from contiguous segments with the
  temporal order preserved and randomized placement. 1.0.0 shuffled individual
  observations, which was inconsistent with the manuscript.
* **Detection delay**: explicit computation according to Algorithm 2 and
  analytical delay bounds, plus the sample-level latency to the first flagged
  observation (`src/online_detection.py`).
* **Selection criterion**: J = (1 − Acc_val)² (Eq. 2) by default, consistent with the manuscript; 1.0.0 used the validation loss.
* **Search**: exhaustive enumeration and the reversed/joint/alternating schemes
  of Table 8; evaluation counts are reported.
* **Data pipeline** (`src/data.py`): run-level split before windowing and
  train-only normalization.
* **Algorithm 3**: documented median aggregation of decorrelation times;
  optional held-out calibration set for α.
* Sensitivity scripts for λ and h are now executable.
* Compressed `.rar` archives removed.

## 1.0.0

Initial reference implementation.
