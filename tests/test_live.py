"""Invariants of the live dispatch engine and the courier replay."""
import numpy as np
import pytest

from src.live_dispatch import POLICIES, LiveDispatch, ReliabilityModel


@pytest.fixture(scope="module")
def rm():
    return ReliabilityModel()


def test_model_never_trains_on_replayed_tasks(rm):
    assert not set(rm.train_ids) & set(rm.replay_ids)
    assert set(rm.train.instance_id) == set(rm.train_ids)


def test_every_job_done_once_capacity_respected_outcomes_recorded(rm):
    sim = LiveDispatch(rm, pool_size=6, slots=2, load=1.2, batch_minutes=20, seed=3)
    while not sim.finished:
        sim.step(15)
        for w in sim.worlds.values():
            busy = np.bincount([k for _, _, k in w.running], minlength=sim.K)
            assert (busy <= sim.slots).all() and (w.free == sim.slots - busy).all()
    for p in POLICIES:
        w = sim.worlds[p]
        assert sorted(w.done) == list(range(sim.n))
        for j, (k, start, end, y) in w.done.items():
            assert y == sim.Y[j, k]                      # success is the agent's recorded result
            assert start >= sim.arrival[j] and end - start == sim.work[j]


def test_all_policies_see_the_same_jobs(rm):
    sim = LiveDispatch(rm, seed=5)
    sim.step(300)
    queues = {p: len(w.queue) + len(w.running) + len(w.done) for p, w in sim.worlds.items()}
    assert len(set(queues.values())) == 1 and queues["random"] == sim.next_job
