# Code Availability Statement

An implementation of HAFSE and of the online novelty-detection workflow
(Algorithms 1–3) is available in this repository (release 2.0.0). It contains
the model definition corresponding to Fig. 4 and Table X, the HAFSE search and
training procedure, the data-partitioning pipeline (run-level split,
train-only normalization and window extraction), the online detection and
adaptive-parameter modules, the open-set evaluation metrics and the
reproducibility configuration (seeds 0–4).

The repository is a clean implementation that follows the manuscript; it is
not the set of original notebooks used during the study, and re-executions
reproduce the experimental protocol rather than the exact historical runs.
The TE and Two-Tank datasets are public; the GPN data are available from the
authors of the corresponding reference upon request.
