"""Job-to-worker assignment under capacity limits, plus the baselines it is compared with.

Every function returns an integer array `a` of length n_jobs where a[j] is the index of the
worker that gets job j. No worker receives more jobs than its capacity.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _capacities(capacity, n_jobs: int, n_workers: int) -> np.ndarray:
    caps = np.minimum(np.broadcast_to(np.asarray(capacity, dtype=int), (n_workers,)), n_jobs)
    if caps.sum() < n_jobs:
        raise ValueError(f"total capacity {caps.sum()} is less than the {n_jobs} jobs to assign")
    return caps


def solve_assignment(score, capacity, cost=None, lam: float = 0.0) -> np.ndarray:
    """Optimal assignment: maximise sum of (score - lam * cost) over the chosen (job, worker) pairs.

    score    : (n_jobs, n_workers) predicted success probabilities
    capacity : int or (n_workers,) max jobs per worker
    cost     : optional (n_workers,) cost of using a worker for one job
    lam      : weight of cost against success probability

    Each worker is expanded into `capacity` identical slots, which turns the problem into a
    standard one-to-one assignment problem that is solved exactly (Hungarian-type algorithm).
    """
    score = np.asarray(score, dtype=float)
    n_jobs, n_workers = score.shape
    slots = np.repeat(np.arange(n_workers), _capacities(capacity, n_jobs, n_workers))
    value = score if cost is None else score - lam * np.asarray(cost, dtype=float)[None, :]
    rows, cols = linear_sum_assignment(-value[:, slots])
    out = np.empty(n_jobs, dtype=int)
    out[rows] = slots[cols]
    return out


def assign_random(n_jobs: int, n_workers: int, capacity, rng) -> np.ndarray:
    """Baseline: jobs are spread over the free slots at random."""
    slots = np.repeat(np.arange(n_workers), _capacities(capacity, n_jobs, n_workers))
    return rng.permutation(slots)[:n_jobs]


def assign_by_track_record(rating, n_jobs: int, capacity, rng) -> np.ndarray:
    """Baseline: fill the best-rated worker first, then the next best. Jobs are taken in random
    order because this rule knows nothing about the individual job."""
    rating = np.asarray(rating, dtype=float)
    caps = _capacities(capacity, n_jobs, len(rating))
    best_first = np.argsort(-rating, kind="stable")
    slots = np.repeat(best_first, caps[best_first])[:n_jobs]
    out = np.empty(n_jobs, dtype=int)
    out[rng.permutation(n_jobs)] = slots
    return out


def assign_greedy(score, capacity, rng) -> np.ndarray:
    """Jobs arrive one at a time (random order); each takes the free worker with the highest score."""
    score = np.asarray(score, dtype=float)
    n_jobs, n_workers = score.shape
    free = _capacities(capacity, n_jobs, n_workers).copy()
    out = np.empty(n_jobs, dtype=int)
    for j in rng.permutation(n_jobs):
        w = int(np.argmax(np.where(free > 0, score[j], -np.inf)))
        out[j] = w
        free[w] -= 1
    return out
