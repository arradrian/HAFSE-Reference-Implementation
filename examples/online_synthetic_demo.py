"""Minimal demonstration of Algorithm 2 using LOF in both stages."""
from pathlib import Path
import sys
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.online_detection import process_online_window
from src.reproducibility import set_global_seed

set_global_seed(0)
rng = np.random.default_rng(0)

# Known training data.
x0 = rng.normal(loc=-1.0, scale=0.25, size=(150, 2))
x1 = rng.normal(loc=+1.0, scale=0.25, size=(150, 2))
X_known = np.vstack([x0, x1])
y_known = np.array([0]*len(x0) + [1]*len(x1))

classifier = LogisticRegression().fit(X_known, y_known)
stage1_lof = LocalOutlierFactor(
    n_neighbors=5, novelty=True, contamination=0.03
).fit(X_known)

# Illustrative 50-sample stream: 10 known + 40 coherent unknown observations.
known = rng.normal(loc=1.0, scale=0.25, size=(10, 2))
unknown = rng.normal(loc=4.0, scale=0.12, size=(40, 2))
stream = np.vstack([known, unknown])
# Known and unknown observations kept as contiguous segments (no shuffling).

decision = process_online_window(
    stream_x=stream,
    lof_detector=stage1_lof,
    dl_classifier=classifier,
    w=50,
    thcr_percent=60.0,
    h_stage2=5,
    minimum_new_event_fraction=0.60,
    inclusive_threshold=True,   # PAnSa >= ThCr (Algorithm 2)
)

print("State:", decision.state)
print("PAnSa (%):", round(decision.pansa_percent, 2))
print("Sa_counter:", decision.sample_count)
print("AnSa_counter:", decision.anomaly_count)
if decision.stage2_result:
    print("Stage-2 coherent fraction:",
          round(decision.stage2_result.new_event_fraction, 3))
    print("Stage-2 outlier fraction:",
          round(decision.stage2_result.outlier_fraction, 3))
