# HAFSE Reproducibility

Reference implementation of the core methodological workflow described in the HAFSE manuscript.

## Scope
This repository provides a reference implementation of:
- target-specific input and classifier construction;
- transfer of a compatible pretrained CNN-LSTM representation backbone;
- sequential HAFSE optimization (N -> M);
- final target training with source-weight L2 pull-back;
- Differential Evolution configuration;
- reproducibility settings.

The source-domain classifier is discarded. Transfer concerns reusable latent representations rather than source class semantics.

## HAFSE workflow
1. Adaptive Fine-Tuning: search N in {1,2,3,4,5}.
2. Conditional Structural Expansion: fix N=N* and search M in {1,2,3,4,5}.
3. Final Model Training: optimize target loss + lambda ||theta_t-theta_S^(t)||_2^2.

## Main configuration
- K = 5
- lambda = 1e-3; sensitivity: {1e-4,1e-3,1e-2,1e-1}
- DE: population=10, CR=0.5, F=0.1, max_iterations=100, tolerance=1e-4
- LOF h=5; sensitivity: {3,5,7,10,15}
- 5 reference runs

The fixed seeds distributed here are reference-implementation seeds for deterministic reruns; they are not claimed to be the historical randomly generated seeds used in the original experiments.

## Install
```bash
pip install -r requirements.txt
```

## Structure
- `src/models.py`: CNN-LSTM source/target construction
- `src/differential_evolution.py`: discrete DE search
- `src/hafse.py`: Algorithm 1
- `src/reproducibility.py`: deterministic reference seeds
- `config/default_config.yaml`: main parameters
- `experiments/`: execution/sensitivity templates
- `examples/synthetic_demo.py`: minimal model-construction demo

## Online novelty detection

The repository also implements Algorithms 2 and 3 from the revised manuscript:

- `src/online_detection.py`: Stage-1 LOF sample screening, anomaly buffer `B_A`,
  `PAnSa`, threshold triggering, and Stage-2 LOF analysis of the buffered anomalies.
- `src/adaptive_parameters.py`: data-driven estimation of `w` and `ThCr`
  from temporal correlation, false-alarm rate, and a binomial tail criterion.
- `src/metrics.py`: AUROC, OSCR, U-Recall, FUR, EDR, detection delay, and
  run-level aggregation.

### Systematic open-set protocol

The revised evaluation protocol is documented in `experiments/run_online_osr.py`:
- 500 samples per scenario;
- 5 independent runs;
- independently randomized stream ordering across runs;
- real withheld conditions as unknown classes;
- mean ± standard deviation across runs.

No synthetic outliers are used as surrogates for the withheld unknown classes in the
systematic evaluation.


## Controlled novel-event update

Candidate novel events detected online are not automatically incorporated into the
deployed model. They require controlled external characterization/validation before
being appended to the historical training database and used in a later retraining cycle.
This mirrors the workflow described in the manuscript.
