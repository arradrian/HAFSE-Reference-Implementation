"""Algorithm 2: Online Classification and Novel-Event Detection.

Implementation of Algorithm 2 of the manuscript.

Two LOF uses are represented:

Stage 1 -- sample screening
    Incoming normalized samples are evaluated by a trained LOF novelty detector.
    Known samples are passed to the DL classifier.
    Anomalous samples are stored in the anomaly buffer B_A and counted once.

Window decision
    Windows are non-overlapping (tumbling) blocks of w consecutive observations.
    Once Sa_counter reaches w, PAnSa = AnSa_counter / w * 100 is computed.
    The decision is taken at the closure of the window, and the buffer and
    counters are reset before the next window.

Stage 2 -- anomaly-set analysis
    If PAnSa >= ThCr, a second LOF analysis is applied to B_A to separate
    sparse/outlier behavior from a coherent candidate novel-event structure:
    B_A is standardized, LOF (h neighbors) is fitted within B_A, and a new
    event is declared when the fraction of locally supported (inlier) members
    reaches ``minimum_new_event_fraction`` (0.60 by default).

Detection delay (event level)
    Number of samples between the onset of an unknown event and the closure of
    the first window whose decision is a new event (see ``detection_delay``).

A detected candidate new event is NOT automatically incorporated into the model.
It must first be externally characterized/validated before being added to the
historical training database, consistent with the manuscript.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class Stage2LOFResult:
    is_new_event: bool
    labels: np.ndarray
    new_event_fraction: float
    outlier_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowDecision:
    state: str
    pansa_percent: float
    sample_count: int
    anomaly_count: int
    known_predictions: list[Any] = field(default_factory=list)
    anomaly_indices: list[int] = field(default_factory=list)
    anomaly_buffer: np.ndarray | None = None
    stage2_result: Stage2LOFResult | None = None


def _predict_known_class(dl_classifier, x: np.ndarray):
    """Support sklearn-style and Keras-style classifiers."""
    x = np.asarray(x)
    if not hasattr(dl_classifier, "predict"):
        raise TypeError("DL classifier must expose a predict method.")
    try:
        pred = dl_classifier.predict(x[None, ...], verbose=0)
    except TypeError:
        pred = dl_classifier.predict(x[None, ...])

    pred = np.asarray(pred)
    if pred.ndim >= 2 and pred.shape[-1] > 1:
        return int(np.argmax(pred, axis=-1).ravel()[0])
    return pred.ravel()[0]


def second_stage_lof_analysis(
    anomaly_buffer: np.ndarray,
    h: int = 5,
    minimum_new_event_fraction: float = 0.60,
    contamination: str | float = "auto",
) -> Stage2LOFResult:
    """LOF-based analysis of the buffered anomalous set.

    LOF itself produces local outlier scores rather than semantic class labels.
    To operationalize the manuscript's second-stage distinction between sparse
    outliers and a coherent candidate novel-event structure, we:

    1. standardize B_A;
    2. compute LOF labels/scores within B_A;
    3. regard locally supported/inlier members (+1) as the coherent portion;
    4. regard LOF outliers (-1) as sparse outliers;
    5. declare a candidate New Event only if the coherent fraction reaches the
       configured minimum fraction.

    This remains a LOF-based density decision and does not introduce a separate
    clustering algorithm into the methodological workflow.
    """
    x = np.asarray(anomaly_buffer, dtype=float)
    if x.ndim > 2:
        x = x.reshape(len(x), -1)

    if len(x) < max(3, h + 1):
        return Stage2LOFResult(
            is_new_event=False,
            labels=np.full(len(x), -1, dtype=int),
            new_event_fraction=0.0,
            outlier_fraction=1.0 if len(x) else 0.0,
            metadata={"reason": "insufficient_samples_for_second_stage_lof"},
        )

    xz = StandardScaler().fit_transform(x)
    n_neighbors = min(h, len(x) - 1)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    labels = lof.fit_predict(xz)  # +1 local inlier/coherent, -1 sparse outlier

    coherent_fraction = float(np.mean(labels == 1))
    outlier_fraction = float(np.mean(labels == -1))
    is_new_event = coherent_fraction >= minimum_new_event_fraction

    return Stage2LOFResult(
        is_new_event=is_new_event,
        labels=labels,
        new_event_fraction=coherent_fraction,
        outlier_fraction=outlier_fraction,
        metadata={
            "reason": "coherent_candidate_event" if is_new_event else "sparse_outliers",
            "h": int(n_neighbors),
            "minimum_new_event_fraction": float(minimum_new_event_fraction),
        },
    )


def process_online_window(
    stream_x: np.ndarray,
    lof_detector,
    dl_classifier,
    w: int,
    thcr_percent: float,
    h_stage2: int = 5,
    minimum_new_event_fraction: float = 0.60,
    inclusive_threshold: bool = True,
) -> WindowDecision:
    """Process one temporal window according to Algorithm 2.

    ``inclusive_threshold=True`` (default) implements the manuscript rule
    PAnSa >= ThCr (Algorithm 2, line 16).
    """
    x = np.asarray(stream_x)
    if len(x) < w:
        raise ValueError(f"stream_x contains {len(x)} samples but w={w}.")
    x = x[:w]

    # Sa_counter counts all processed observations in the window.
    sa_counter = 0
    # AnSa_counter counts ONLY observations flagged anomalous by Stage-1 LOF.
    ansa_counter = 0
    anomaly_buffer = []
    anomaly_indices = []
    known_predictions = []

    stage1 = np.asarray(lof_detector.predict(x))
    for j, (sample, status) in enumerate(zip(x, stage1)):
        sa_counter += 1
        if status != -1:
            known_predictions.append((j, _predict_known_class(dl_classifier, sample)))
        else:
            anomaly_buffer.append(sample)
            anomaly_indices.append(j)
            ansa_counter += 1

    pansa = 100.0 * ansa_counter / sa_counter if sa_counter else 0.0
    trigger = pansa >= thcr_percent if inclusive_threshold else pansa > thcr_percent

    if anomaly_buffer:
        b_a = np.asarray(anomaly_buffer)
    else:
        b_a = np.empty((0,) + x.shape[1:], dtype=x.dtype)

    stage2 = None
    if trigger and ansa_counter > 0:
        stage2 = second_stage_lof_analysis(
            b_a,
            h=h_stage2,
            minimum_new_event_fraction=minimum_new_event_fraction,
        )
        state = "CANDIDATE_NEW_EVENT" if stage2.is_new_event else "OUTLIERS"
    elif ansa_counter > 0:
        state = "ISOLATED_ANOMALIES"
    else:
        state = "KNOWN_ONLY"

    return WindowDecision(
        state=state,
        pansa_percent=pansa,
        sample_count=sa_counter,
        anomaly_count=ansa_counter,
        known_predictions=known_predictions,
        anomaly_indices=anomaly_indices,
        anomaly_buffer=b_a,
        stage2_result=stage2,
    )


def validate_and_archive_new_event(
    decision: WindowDecision,
    expert_validation_callback: Callable[[np.ndarray], dict[str, Any]],
    historical_database_append: Callable[[np.ndarray, dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Controlled external validation before historical-database incorporation."""
    if decision.state != "CANDIDATE_NEW_EVENT" or decision.anomaly_buffer is None:
        return None

    characterization = expert_validation_callback(decision.anomaly_buffer)
    if characterization.get("validated", False):
        historical_database_append(decision.anomaly_buffer, characterization)
    return characterization


def process_online_stream(
    stream_x: np.ndarray,
    lof_detector,
    dl_classifier,
    w: int,
    thcr_percent: float,
    h_stage2: int = 5,
    minimum_new_event_fraction: float = 0.60,
) -> list[tuple[int, WindowDecision]]:
    """Apply Algorithm 2 to consecutive non-overlapping windows of a stream.

    Returns a list of (window_end_index, decision), where window_end_index is
    the index of the last observation of the window (decision time).
    Trailing observations that do not complete a window are not processed.
    """
    x = np.asarray(stream_x)
    out = []
    for start in range(0, len(x) - w + 1, w):
        d = process_online_window(
            x[start:start + w], lof_detector, dl_classifier, w, thcr_percent,
            h_stage2=h_stage2, minimum_new_event_fraction=minimum_new_event_fraction,
        )
        out.append((start + w - 1, d))
    return out


def detection_delay(decisions: list[tuple[int, WindowDecision]], onset: int,
                    end: int | None = None) -> float | None:
    """Event-level detection delay according to Algorithm 2.

    ``onset`` is the index of the first observation of the unknown event and
    ``end`` (optional) the index of its last observation. The delay is the
    number of samples between the onset and the closure of the first window
    ending at or after the onset whose decision is a new event. Windows that
    start after ``end`` (if given) are not considered. Returns None if the
    event is not detected.
    """
    for window_end, d in decisions:
        if window_end < onset:
            continue
        if end is not None and window_end - d.sample_count + 1 > end:
            break
        if d.state == "CANDIDATE_NEW_EVENT":
            return float(window_end - onset)
    return None


def first_flag_latency(anomaly_flags, onset: int, end: int | None = None) -> float | None:
    """Sample-level detection latency.

    Number of samples between the onset of an unknown event and the first
    observation of that event flagged as anomalous by the stage-1 detector.
    The new-event decision itself is taken later, at the closure of the window
    containing the accumulated anomalies (see ``detection_delay``).
    """
    flags = np.asarray(anomaly_flags, dtype=bool)
    stop = len(flags) if end is None else min(len(flags), end + 1)
    idx = np.flatnonzero(flags[onset:stop])
    return float(idx[0]) if idx.size else None


def delay_bounds(w: int, thcr_percent: float) -> tuple[int, int, float]:
    """Analytical delay bounds for tumbling windows and perfect sample detection.

    Returns (minimum, maximum, mean) delay in samples, assuming the onset is
    uniformly distributed within a window and every post-onset observation is
    flagged as anomalous. Useful to check the consistency of reported delays.
    """
    import math
    k_th = math.ceil(thcr_percent / 100.0 * w)
    delays = []
    for p in range(1, w + 1):            # onset at position p of a window
        if w - p + 1 >= k_th:
            delays.append(w - p)         # declared at the end of the same window
        else:
            delays.append(2 * w - p)     # declared at the end of the next window
    return min(delays), max(delays), float(np.mean(delays))
