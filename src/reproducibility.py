"""Reproducibility settings.

The seeds 0, 1, 2, 3 and 4 (Sec. 4.3.1) allow deterministic re-execution of the
experimental protocol. They are not the seeds of the historical runs reported
in the manuscript, which were randomly generated and not recorded.
"""
import os
import random

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

REFERENCE_SEEDS = (0, 1, 2, 3, 4)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
