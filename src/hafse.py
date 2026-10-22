"""HAFSE: Hybrid Adaptive Fine-Tuning and Structural Expansion (Algorithm 1).

Structural selection (Eqs. 2, 6-7) minimizes a validation criterion J:

    N* = argmin_N  J( f_T^(N, M0) ),   M0 = 0 (no expansion)
    M* = argmin_M  J( f_T^(N*, M) )

By default J = (1 - Acc_val)^2 (Eq. 2), a monotonic transformation of the
target-domain validation accuracy, i.e. selection maximizes validation
accuracy. ``criterion="val_loss"`` uses the validation loss instead.

The final model f_T^(N*, M*) is then trained with the source-preservation
penalty of Eqs. (5) and (8):

    L_tr + lambda * || theta_t - theta_S^(t) ||_2^2

where theta_t are the transferred parameters that are unfrozen for N*.

Besides the sequential N -> M scheme of the manuscript, the reversed (M -> N),
joint (exhaustive over N x M, 25 configurations for K = 5) and alternating
schemes evaluated in Sec. 5.1.4 (Table 8) are also provided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import tensorflow as tf

from .models import TRANSFERRED_LAYERS, build_target_model
from .search import SearchResult, discrete_de, exhaustive_search


@dataclass
class HAFSEResult:
    model: object
    N_star: int
    M_star: int
    searches: dict = field(default_factory=dict)  # SearchResult per stage


def pullback(model, source_model):
    """Squared L2 distance between unfrozen transferred weights and source weights."""
    terms = []
    for name in TRANSFERRED_LAYERS:
        t, s = model.get_layer(name), source_model.get_layer(name)
        if not t.trainable:
            continue
        for tw, sw in zip(t.trainable_weights, s.trainable_weights):
            terms.append(tf.reduce_sum(tf.square(tw - tf.stop_gradient(sw))))
    return tf.add_n(terms) if terms else tf.constant(0.0, tf.float32)


def compile_model(model, source_model, lambda_reg=1e-3, learning_rate=1e-4):
    ce = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_true, y_pred):
        return ce(y_true, y_pred) + lambda_reg * pullback(model, source_model)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss,
                  metrics=["accuracy"])


def make_evaluator(source_model, timesteps, target_features, target_classes,
                   train_data, val_data, lambda_reg=1e-3, learning_rate=1e-4,
                   epochs_search=5, expansion_units=64, criterion="val_accuracy"):
    """Return J(N, M) after a short training run.

    criterion="val_accuracy": J = (1 - best validation accuracy)^2  (Eq. 2)
    criterion="val_loss":     J = minimum validation loss
    """
    if criterion not in ("val_accuracy", "val_loss"):
        raise ValueError("criterion must be 'val_accuracy' or 'val_loss'.")

    def evaluate(N, M):
        model = build_target_model(source_model, timesteps, target_features,
                                   target_classes, N, M, expansion_units)
        compile_model(model, source_model, lambda_reg, learning_rate)
        hist = model.fit(train_data, validation_data=val_data,
                         epochs=epochs_search, verbose=0)
        if criterion == "val_accuracy":
            return float((1.0 - max(hist.history["val_accuracy"])) ** 2)
        return float(min(hist.history["val_loss"]))
    return evaluate


def _search(candidates, objective, method, seed, de_kwargs):
    if method == "exhaustive":
        return exhaustive_search(list(candidates), objective)
    if method == "de":
        return discrete_de(candidates, objective, seed=seed, **(de_kwargs or {}))
    raise ValueError("method must be 'de' or 'exhaustive'.")


def select_structure(evaluate, scheme="sequential", N_space=(1, 2, 3, 4, 5),
                     M_space=(1, 2, 3, 4, 5), M0=0, method="de", seed=0,
                     de_kwargs=None, max_alternations=3):
    """Select (N*, M*) with one of the schemes compared in Table 8."""
    searches = {}
    if scheme == "sequential":          # N -> M (HAFSE)
        rn = _search(N_space, lambda n: evaluate(n, M0), method, seed, de_kwargs)
        rm = _search(M_space, lambda m: evaluate(rn.value, m), method, seed + 1, de_kwargs)
        searches.update(N=rn, M=rm)
        return rn.value, rm.value, searches
    if scheme == "reversed":            # M -> N
        n0 = min(N_space)
        rm = _search(M_space, lambda m: evaluate(n0, m), method, seed, de_kwargs)
        rn = _search(N_space, lambda n: evaluate(n, rm.value), method, seed + 1, de_kwargs)
        searches.update(M=rm, N=rn)
        return rn.value, rm.value, searches
    if scheme == "joint":               # exhaustive over N x M
        rj = exhaustive_search(list(product(N_space, M_space)), lambda nm: evaluate(*nm))
        searches.update(joint=rj)
        return rj.value[0], rj.value[1], searches
    if scheme == "alternating":         # N <-> M until no change
        n, m = min(N_space), M0
        for k in range(max_alternations):
            rn = _search(N_space, lambda x: evaluate(x, m), method, seed + 2 * k, de_kwargs)
            rm = _search(M_space, lambda x: evaluate(rn.value, x), method, seed + 2 * k + 1, de_kwargs)
            searches[f"N_{k}"], searches[f"M_{k}"] = rn, rm
            if (rn.value, rm.value) == (n, m):
                break
            n, m = rn.value, rm.value
        return n, m, searches
    raise ValueError("scheme must be sequential, reversed, joint or alternating.")


def run_hafse(source_model, timesteps, target_features, target_classes, train_data,
              val_data, N_space=(1, 2, 3, 4, 5), M_space=(1, 2, 3, 4, 5), lambda_reg=1e-3,
              learning_rate=1e-4, epochs_search=5, epochs_final=20, expansion_units=64,
              scheme="sequential", method="de", seed=0, de_kwargs=None,
              criterion="val_accuracy"):
    """Algorithm 1: structural selection followed by final target training."""
    evaluate = make_evaluator(source_model, timesteps, target_features, target_classes,
                              train_data, val_data, lambda_reg, learning_rate,
                              epochs_search, expansion_units, criterion)
    n_star, m_star, searches = select_structure(evaluate, scheme, N_space, M_space,
                                                method=method, seed=seed, de_kwargs=de_kwargs)
    final = build_target_model(source_model, timesteps, target_features, target_classes,
                               n_star, m_star, expansion_units)
    compile_model(final, source_model, lambda_reg, learning_rate)
    final.fit(train_data, validation_data=val_data, epochs=epochs_final, verbose=1)
    return HAFSEResult(final, n_star, m_star, searches)
