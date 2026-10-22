"""Algorithm 3: Adaptive estimation of window size w and threshold ThCr.

The implementation follows Sec. 4.1.3 and Algorithm 3 of the manuscript:
1) estimate the false-alarm rate alpha of the LOF detector on known-class data
   (optionally on a held-out calibration subset not used to fit LOF);
2) compute the ACF of each normalized process variable (Eq. 9) and its
   decorrelation time tau_c,i = min{tau : rho_i(tau) < epsilon};
3) aggregate tau_c = median_i tau_c,i;
4) derive w = max(2, ceil(tau_c / Ts)) (Eq. 10);
5) obtain the minimum anomaly count k_th from a Binomial(w, alpha) tail (Eq. 13);
6) compute ThCr = 100 * k_th / w (Eq. 12).

Note: the binomial model treats anomaly decisions within a window as
independent. Positively correlated decisions may make the actual false-alarm
probability exceed beta, so beta should be interpreted as a nominal level.

This module is intentionally generic and operates on historical known-class data.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import binom


@dataclass
class AdaptiveParameters:
    w: int
    thcr_percent: float
    alpha: float
    beta: float
    tau_c: float
    k_threshold: int


def autocorrelation_1d(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("At least two finite samples are required.")
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.ones(1, dtype=float)
    if max_lag is None:
        max_lag = min(x.size - 1, 1000)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf


def estimate_tau_c(
    historical_x: np.ndarray,
    epsilon: float = 0.2,
    sampling_period: float = 1.0,
) -> float:
    """Estimate decorrelation time using the first lag where ACF < epsilon.

    For multivariate data, the median decorrelation time across variables is used.
    """
    x = np.asarray(historical_x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    taus = []
    for j in range(x.shape[1]):
        acf = autocorrelation_1d(x[:, j])
        idx = np.where(acf < epsilon)[0]
        lag = int(idx[0]) if idx.size else max(1, len(acf) - 1)
        taus.append(lag * sampling_period)

    return float(np.median(taus))


def estimate_false_alarm_rate(
    lof_detector,
    historical_known_x: np.ndarray,
) -> float:
    """Estimate alpha as the fraction of known historical samples flagged anomalous.

    The detector must expose ``predict(X)`` with scikit-learn LOF novelty semantics:
    +1 = inlier/known, -1 = outlier/anomalous.
    """
    pred = np.asarray(lof_detector.predict(historical_known_x))
    if pred.size == 0:
        raise ValueError("Historical dataset cannot be empty.")
    return float(np.mean(pred == -1))


def binomial_threshold(w: int, alpha: float, beta: float = 0.05) -> int:
    """Return the smallest k such that P(K >= k) <= beta, K~Binomial(w,alpha)."""
    if w < 1:
        raise ValueError("w must be >= 1.")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for k in range(1, w + 1):
        tail = float(binom.sf(k - 1, w, alpha))
        if tail <= beta:
            return k
    return w


def estimate_adaptive_parameters(
    historical_known_x: np.ndarray,
    lof_detector,
    sampling_period: float = 1.0,
    epsilon: float = 0.2,
    beta: float = 0.05,
    minimum_window: int = 2,
    calibration_known_x: np.ndarray | None = None,
) -> AdaptiveParameters:
    """Algorithm 3. If ``calibration_known_x`` is given, alpha is estimated on it."""
    alpha_data = historical_known_x if calibration_known_x is None else calibration_known_x
    alpha = estimate_false_alarm_rate(lof_detector, alpha_data)
    tau_c = estimate_tau_c(
        historical_known_x,
        epsilon=epsilon,
        sampling_period=sampling_period,
    )
    w = max(minimum_window, int(math.ceil(tau_c / sampling_period)))
    k_th = binomial_threshold(w, alpha, beta)
    thcr = 100.0 * k_th / w
    return AdaptiveParameters(
        w=w,
        thcr_percent=thcr,
        alpha=alpha,
        beta=beta,
        tau_c=tau_c,
        k_threshold=k_th,
    )
