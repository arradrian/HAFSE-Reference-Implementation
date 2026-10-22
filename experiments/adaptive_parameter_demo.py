"""Synthetic demonstration of Algorithm 3."""
from pathlib import Path
import sys
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_parameters import estimate_adaptive_parameters
from src.reproducibility import set_global_seed

set_global_seed(0)
rng = np.random.default_rng(0)

# Synthetic known historical multivariate signal.
n = 1000
t = np.arange(n)
x = np.column_stack([
    np.sin(2*np.pi*t/30) + 0.10*rng.normal(size=n),
    np.cos(2*np.pi*t/45) + 0.10*rng.normal(size=n),
])

lof = LocalOutlierFactor(n_neighbors=5, novelty=True, contamination=0.02)
lof.fit(x)

params = estimate_adaptive_parameters(
    historical_known_x=x,
    lof_detector=lof,
    sampling_period=1.0,
    epsilon=0.2,
    beta=0.05,
)

print(params)
