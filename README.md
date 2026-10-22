# HAFSE — Implementation of the adaptive transfer and open-set monitoring workflow

Implementation of the methods described in the manuscript *"An Adaptive Deep
Transfer Learning Framework for Condition Monitoring with Open-Set Recognition
under Severe Domain Shift"*:

* **Algorithm 1 — HAFSE**: transfer of the CNN-LSTM backbone between processes
  with different input dimensionality, sequential selection of the number of
  unfrozen transferred layers (N) and of new target-specific layers (M), and
  final training with source-weight L2 pull-back.
* **Algorithm 2 — online classification and novel-event detection** (two-stage LOF).
* **Algorithm 3 — data-driven selection rule for w and ThCr.**
* Open-set metrics (AUROC, OSCR, U-Recall, FUR, EDR, detection delay).


## Correspondence with the manuscript

| Manuscript | File |
|---|---|
| Fig. 4 (architecture, layer-wise transfer status, definition of N) | `src/models.py` |
| Sec. 3.5, Eqs. (5)–(8), Algorithm 1 (HAFSE) | `src/hafse.py` |
| DE (Sec. 3.5, 4.3.1) and exhaustive enumeration (K = 5) | `src/search.py` |
| Table 8 (sequential, reversed, joint, alternating schemes) | `src/hafse.py::select_structure` |
| Sec. 4.1.1 (run-level split, train-only normalization, windows) | `src/data.py` |
| Algorithm 2, detection delay | `src/online_detection.py` |
| Sec. 4.1.3, Eqs. (9)–(13), Algorithm 3 | `src/adaptive_parameters.py` |
| Sec. 4.3.2, 5.2.2 (systematic open-set protocol) | `experiments/run_online_osr.py` |
| Open-set metrics | `src/metrics.py` |
| λ and h sensitivity | `experiments/lambda_sensitivity.py`, `experiments/lof_h_sensitivity.py` |
| Seeds (Sec. 4.3.1) | `src/reproducibility.py`, `config/default_config.yaml` |

## Architecture and transfer

| Layer | Status in the target model |
|---|---|
| Conv1D-1 (64, k=7) + BN | Target-specific input adapter: kernel (7, d, 64) depends on d → random init, always trainable |
| Conv1D-2 (128, k=5) + BN | Transferred; fine-tuned if N ≥ 5 |
| Conv1D-3 (256, k=3) + BN | Transferred; fine-tuned if N ≥ 4 |
| LSTM-1 (128) + BN | Transferred; fine-tuned if N ≥ 3 |
| LSTM-2 (64) + BN | Transferred; fine-tuned if N ≥ 2 |
| LSTM-3 (32) + BN | Transferred; fine-tuned if N ≥ 1 |
| M × Dense(64, ReLU) | New target-specific expansion layers |
| Dense-128 + BN + Dropout(0.5) + softmax | Target-specific classification head (source head discarded) |

Each value of N = 1, …, 5 yields a distinct set of trainable weights
(`python examples/synthetic_demo.py` prints them).

## Main configuration (`config/default_config.yaml`)

* K = 5, N ∈ {1,…,5}, M ∈ {1,…,5}, M0 = 0; selection criterion J = (1 − Acc_val)² (Eq. 2), i.e. maximization of target validation accuracy (validation loss available as an option).
* λ = 1e-3 (sensitivity {1e-4, 1e-3, 1e-2, 1e-1}).
* DE: population 10, CR = 0.5, F = 0.1, 100 iterations, tolerance 1e-4.
  For K = 5, `search_method: exhaustive` is an equivalent alternative.
* Online stage: w = 50, ThCr = 60 % (fixed configuration used in the
  experiments), rule PAnSa ≥ ThCr, LOF h = 5, stage-2 coherent fraction ≥ 0.60.
* Seeds: 0, 1, 2, 3, 4.

## Usage

```bash
pip install -r requirements.txt
python examples/synthetic_demo.py                      # Table X and N configurations
python experiments/run_hafse.py --synthetic            # Algorithm 1 end to end
python experiments/run_hafse.py --synthetic --scheme joint --method exhaustive
python experiments/run_online_osr.py --synthetic       # Algorithm 2 + open-set protocol
python experiments/adaptive_parameter_demo.py          # Algorithm 3
python experiments/lambda_sensitivity.py --synthetic
python experiments/lof_h_sensitivity.py --synthetic
```

To use the study datasets, replace the synthetic loaders with project loaders
(see `data/README.md`).


## Version history

See `CHANGELOG.md`.
