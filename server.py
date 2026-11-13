"""Live dispatch control room.   Run:  .venv/bin/python server.py   then open http://localhost:8000"""
import itertools
import os
from collections import OrderedDict

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.live_couriers import CODE, CourierMonitor, risk_threshold
from src.live_dispatch import DIFF_LABEL, POLICIES, POLICY_LABEL, LiveDispatch, ReliabilityModel, short_name

app = FastAPI(title="Workforce Dispatch & Reliability Engine")
app.mount("/static", StaticFiles(directory="web"), name="static")
RM = ReliabilityModel()
THRESHOLD = risk_threshold()
SESSIONS: "OrderedDict[str, object]" = OrderedDict()
_ids = itertools.count(1)


def _keep(sim) -> str:
    sid = str(next(_ids))
    SESSIONS[sid] = sim
    while len(SESSIONS) > 40:
        SESSIONS.popitem(last=False)
    return sid


def _get(sid: str, kind):
    sim = SESSIONS.get(sid)
    if not isinstance(sim, kind):
        raise HTTPException(404, "session expired; press Reset")
    return sim


class AgentConfig(BaseModel):
    pool_size: int = Field(8, ge=3, le=16)
    slots: int = Field(2, ge=1, le=4)
    load: float = Field(0.9, ge=0.3, le=1.5)
    batch_minutes: int = Field(30, ge=5, le=120)
    seed: int = Field(1, ge=0, le=99999)


class NewJob(BaseModel):
    repo: str
    difficulty: int = Field(1, ge=0, le=3)
    statement_len: int = Field(1500, ge=50, le=50000)
    year: int = Field(2023, ge=2012, le=2030)


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/api/meta")
def meta():
    ev = pd.read_csv("results/live_dispatch_eval.csv") if os.path.exists("results/live_dispatch_eval.csv") else None
    return {"train_tasks": RM.n_train_tasks, "train_rows": RM.n_train_rows, "replay_tasks": len(RM.replay_ids),
            "agents_total": len(RM.agents), "repos": RM.repos, "difficulty": DIFF_LABEL,
            "policies": [{"id": p, "label": POLICY_LABEL[p]} for p in POLICIES], "cities": list(CODE),
            "evaluation": None if ev is None else ev.to_dict("records"),
            "offline": {name: pd.read_csv(f"results/{f}").round(4).to_dict("records")
                        for name, f in [("couriers", "lade_results.csv"), ("agents", "swebench_prediction.csv"),
                                        ("assignment", "swebench_assignment.csv")] if os.path.exists(f"results/{f}")}}


@app.post("/api/agents/new")
def agents_new(cfg: AgentConfig):
    sim = LiveDispatch(RM, **cfg.model_dump())
    return {"sid": _keep(sim), "config": cfg.model_dump(), "jobs": [sim.job_info(j) for j in range(sim.n)],
            "pool": [{"idx": k, "name": short_name(RM.agents[a]), "full_name": RM.agents[a],
                      "rating": round(float(RM.rating[a]), 3)} for k, a in enumerate(sim.pool)],
            "state": sim.state()}


@app.post("/api/agents/{sid}/step")
def agents_step(sid: str, minutes: int = 5):
    sim = _get(sid, LiveDispatch)
    out = sim.step(max(1, min(minutes, 600)))
    for e in out["events"]:
        if e["type"] == "complete":   # recorded outcome of every pool agent on this task, for the detail view
            e["all_outcomes"] = [int(x) for x in sim.Y[e["job"]]]
    return out


@app.post("/api/agents/{sid}/whatif")
def agents_whatif(sid: str, job: NewJob):
    sim = _get(sid, LiveDispatch)
    if job.repo not in RM.repos:
        raise HTTPException(400, "unknown repository")
    p = RM.score_new_job(sim.pool, job.repo, job.difficulty, job.statement_len, job.year)
    free = (sim.slots - sim.worlds["learning"].free) < sim.slots
    order = sorted(range(sim.K), key=lambda k: -p[k])
    return {"ranking": [{"idx": k, "name": short_name(RM.agents[sim.pool[k]]), "p": round(float(p[k]), 3),
                         "has_free_slot": bool(free[k])} for k in order]}


@app.post("/api/couriers/new")
def couriers_new(city: str = "Shanghai"):
    if city not in CODE:
        raise HTTPException(400, "unknown city")
    mon = CourierMonitor(city, THRESHOLD)
    return {"sid": _keep(mon), "meta": mon.meta()}


@app.post("/api/couriers/{sid}/step")
def couriers_step(sid: str, minutes: int = 5):
    return _get(sid, CourierMonitor).step(max(1, min(minutes, 600)))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
