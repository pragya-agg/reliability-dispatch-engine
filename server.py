"""Live dispatch control room.   Run:  .venv/bin/python server.py   then open http://localhost:8000"""
import itertools
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.live_couriers import CODE, CourierMonitor, risk_threshold
from src.live_dispatch import DIFF_LABEL, POLICIES, POLICY_LABEL, LiveDispatch, ReliabilityModel, short_name
from src.portal import Portal, run_clock

RM = ReliabilityModel()
PORTAL = Portal(RM)
_STOP = threading.Event()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=run_clock, args=(PORTAL, _STOP), daemon=True).start()
    yield
    _STOP.set()


app = FastAPI(title="Workforce Dispatch & Reliability Engine", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.middleware("http")
async def no_stale_pages(request, call_next):
    """Make browsers re-check the page files, so an edit never shows up half-applied."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response
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


# ---------------------------------------------------------------- service portal
class ServiceRequest(BaseModel):
    customer: str = Field("Customer", min_length=1, max_length=40)
    service: str
    preference: Literal["any", "human", "agent", "robot"] = "any"
    priority: Literal["quality", "balanced", "cost"] = "balanced"
    repo: Optional[str] = None
    difficulty: int = Field(1, ge=0, le=3)
    client: Optional[str] = Field(None, max_length=64)


class PortalSettings(BaseModel):
    speed: Optional[float] = Field(None, ge=0.5, le=60)
    auto: Optional[bool] = None
    rate: Optional[float] = Field(None, ge=0, le=120)


@app.get("/portal")
def portal_page():
    return FileResponse("web/portal.html")


@app.get("/api/portal/catalog")
def portal_catalog():
    return PORTAL.catalog()


@app.post("/api/portal/request")
def portal_request(req: ServiceRequest):
    try:
        return {"id": PORTAL.submit(**req.model_dump())}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/portal/state")
def portal_state(client: Optional[str] = None):
    return PORTAL.state(client=client)


@app.post("/api/portal/settings")
def portal_settings(s: PortalSettings):
    with PORTAL.lock:
        for k, v in s.model_dump().items():
            if v is not None:
                setattr(PORTAL, k, v)
    return {"speed": PORTAL.speed, "auto": PORTAL.auto, "rate": PORTAL.rate}


@app.post("/api/portal/reset")
def portal_reset():
    global PORTAL
    with PORTAL.lock:
        fresh = Portal(RM)
        PORTAL.__dict__.update({k: v for k, v in fresh.__dict__.items() if k != "lock"})
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
