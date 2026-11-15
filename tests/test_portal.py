"""Service portal: eligibility, preferences, capacity and where results come from."""
import time

import pytest

from src.live_dispatch import ReliabilityModel
from src.portal import CAPACITY, SERVICES, Portal


@pytest.fixture(scope="module")
def portal():
    p = Portal(ReliabilityModel())
    p.auto, p.speed = False, 600
    return p


def run(p, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        p.tick()
        time.sleep(0.02)


def test_unavailable_worker_type_is_refused(portal):
    with pytest.raises(ValueError):
        portal.submit("x", "plumbing", "robot")


def test_preference_capacity_and_recorded_results(portal):
    ids = [portal.submit("t", "bug_fix", "agent", "quality", repo="django/django", difficulty=d % 3) for d in range(12)]
    ids += [portal.submit("t", "warehouse_picking", "robot", "cost") for _ in range(8)]
    ids += [portal.submit("t", "parcel_pickup", "any") for _ in range(5)]
    by_id = {w["id"]: w for w in portal.workers}
    for _ in range(40):
        run(portal, 0.1)
        for w in portal.workers:
            assert len(w["busy"]) <= CAPACITY[w["type"]]
    run(portal, 3)
    reqs = {r["id"]: r for r in portal.requests}
    for i in ids:
        r = reqs[i]
        assert r["status"] in ("done", "failed")
        for a in r["attempts"]:
            w = by_id[a["worker"]]
            assert w["type"] in r["allowed"] and r["service"] in w["caliber"]
            if r["service"] == "bug_fix" and w["type"] == "agent":
                assert a["success"] == bool(portal.rm.Y[r["task"]["idx"], w["agent_idx"]])
                assert a["outcome_source"] == "recorded SWE-bench result"
        assert len({a["worker"] for a in r["attempts"]}) == len(r["attempts"])   # never the same worker twice
        if r["service"] == "warehouse_picking":
            assert all(a["type"] == "robot" for a in r["attempts"])


def test_bug_fix_tasks_are_unseen_by_the_model(portal):
    used = {r["task"]["id"] for r in portal.requests if "task" in r}
    assert used and not used & set(portal.rm.train_ids)


def test_every_service_has_workers(portal):
    for s, cfg in SERVICES.items():
        for t in cfg["types"]:
            assert any(w["type"] == t and s in w["caliber"] for w in portal.workers), (s, t)
