"""Discrete search over the HAFSE structural variables.

Two search procedures are provided for a bounded candidate set:

* ``discrete_de``: Differential Evolution (Storn and Price, 1997) over a set of
  integer candidates. Continuous individuals are projected onto the nearest
  admissible candidate before evaluation. Parameters follow Sec. 4.3.1
  (population 10, CR = 0.5, F = 0.1, 100 iterations, tolerance 1e-4).
* ``exhaustive_search``: evaluates every candidate once. For the bounded
  setting K = 5 used in the manuscript, this is a valid and transparent
  alternative to DE (Sec. 3.5).

Both functions report the number of objective evaluations and of distinct
candidates evaluated, so that the architecture-selection cost can be reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np


@dataclass
class SearchResult:
    value: Hashable
    objective: float
    iterations: int
    n_evaluations: int
    n_unique_candidates: int
    history: list = field(default_factory=list)  # (candidate, objective) pairs


class _CountingObjective:
    """Wraps an objective, counts evaluations and optionally caches results."""

    def __init__(self, objective: Callable, cache: bool):
        self.objective = objective
        self.cache = cache
        self.memo: dict = {}
        self.history: list = []

    def __call__(self, candidate):
        if self.cache and candidate in self.memo:
            return self.memo[candidate]
        value = float(self.objective(candidate))
        self.history.append((candidate, value))
        self.memo.setdefault(candidate, value)
        return value

    @property
    def n_evaluations(self) -> int:
        return len(self.history)

    @property
    def n_unique(self) -> int:
        return len({c for c, _ in self.history})


def discrete_de(
    candidates: Iterable[int],
    objective: Callable[[int], float],
    population_size: int = 10,
    crossover_probability: float = 0.5,
    scaling_factor: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int | None = 0,
    cache: bool = False,
) -> SearchResult:
    """DE/rand/1/bin over an ordered set of integer candidates (minimization)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(sorted(set(candidates)), dtype=int)
    lo, hi = float(vals.min()), float(vals.max())
    project = lambda x: int(vals[np.argmin(np.abs(vals - x))])
    f = _CountingObjective(objective, cache)

    pop = rng.uniform(lo, hi, population_size)
    scores = np.array([f(project(x)) for x in pop], dtype=float)
    previous = float(scores.min())
    iterations = max_iterations
    for it in range(1, max_iterations + 1):
        for i in range(population_size):
            a, b, c = rng.choice([j for j in range(population_size) if j != i], 3, replace=False)
            mutant = np.clip(pop[a] + scaling_factor * (pop[b] - pop[c]), lo, hi)
            trial = mutant if rng.random() < crossover_probability else pop[i]
            score = f(project(trial))
            if score <= scores[i]:
                pop[i], scores[i] = trial, score
        best = float(scores.min())
        if abs(previous - best) <= tolerance:
            iterations = it
            break
        previous = best
    idx = int(np.argmin(scores))
    return SearchResult(project(pop[idx]), float(scores[idx]), iterations,
                        f.n_evaluations, f.n_unique, f.history)


def exhaustive_search(
    candidates: Sequence[Hashable],
    objective: Callable[[Hashable], float],
) -> SearchResult:
    """Evaluate every candidate once and return the minimizer."""
    f = _CountingObjective(objective, cache=True)
    scores = [(c, f(c)) for c in candidates]
    best_c, best_v = min(scores, key=lambda t: t[1])
    return SearchResult(best_c, best_v, 1, f.n_evaluations, f.n_unique, f.history)
