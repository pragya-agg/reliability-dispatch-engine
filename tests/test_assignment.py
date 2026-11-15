import itertools

import numpy as np
import pytest

from src.assignment import assign_by_track_record, assign_greedy, assign_random, solve_assignment


def test_capacity_is_respected_by_every_method():
    rng = np.random.default_rng(0)
    score = rng.random((30, 4))
    for pick in (solve_assignment(score, 8), assign_random(30, 4, 8, rng),
                 assign_by_track_record(score.mean(0), 30, 8, rng), assign_greedy(score, 8, rng)):
        assert len(pick) == 30 and np.bincount(pick, minlength=4).max() <= 8


def test_optimal_matches_brute_force():
    rng = np.random.default_rng(1)
    score = rng.random((6, 3))
    best = max(score[np.arange(6), list(c)].sum() for c in itertools.product(range(3), repeat=6)
               if np.bincount(c, minlength=3).max() <= 2)
    assert score[np.arange(6), solve_assignment(score, 2)].sum() == pytest.approx(best)


def test_cost_weight_moves_jobs_to_cheaper_workers():
    score = np.array([[0.9, 0.8]] * 4)
    assert (solve_assignment(score, 4) == 0).all()
    assert (solve_assignment(score, 4, cost=[1.0, 0.0], lam=0.5) == 1).all()


def test_too_little_capacity_raises():
    with pytest.raises(ValueError):
        solve_assignment(np.ones((5, 2)), 2)
