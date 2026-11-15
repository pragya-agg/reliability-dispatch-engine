"""Service portal engine: customers request a service, the control room assigns a human, AI agent or robot.

Grounding (shown to the user on every service and every result):
  * "Fix a software bug": AI agents are real SWE-bench Verified submissions. A request is matched to a
    recorded task from the requested repository that the reliability model never trained on, so an agent's
    caliber is the model's prediction and its result is the RECORDED result. Human developers are simulated.
  * "Parcel pickup": couriers are real LaDe-P couriers; their caliber is measured (1 - average predicted late
    risk on a real day). Whether this particular pickup is on time is simulated from that caliber.
    Delivery robots are simulated.
  * All other services: worker profiles, caliber, durations and outcomes are simulated from the assumption
    table ASSUMED_CALIBER below.
"""
import threading
import time

import numpy as np

from src.live_couriers import load_day
from src.live_dispatch import ReliabilityModel, short_name

TYPES = {"human": "Human", "agent": "AI agent", "robot": "Robot"}
COST = {"agent": 1.0, "robot": 3.0, "human": 8.0}                 # assumed cost units per job
CAPACITY = {"agent": 2, "robot": 1, "human": 1}                  # parallel jobs per worker
PRIORITY = {"quality": 0.0, "balanced": 0.15, "cost": 0.5}      # weight of normalised cost against caliber
DIFF_LABEL = {0: "<15 min", 1: "15 min-1 h", 2: "1-4 h", 3: ">4 h"}

SERVICES = {
    "bug_fix": {"name": "Fix a software bug", "types": ["agent", "human"], "grounding": "measured",
                "note": "AI agents: real SWE-bench Verified agents; caliber from the trained model, result is the recorded one. Human developers: simulated."},
    "parcel_pickup": {"name": "Parcel pickup", "types": ["human", "robot"], "grounding": "partly measured",
                      "note": "Couriers: real LaDe-P couriers with measured reliability; this pickup's outcome is simulated from it. Delivery robots: simulated."},
    "it_support": {"name": "IT support ticket", "types": ["agent", "human"], "grounding": "simulated", "note": "All profiles simulated."},
    "scheduling": {"name": "Appointment scheduling", "types": ["agent", "human"], "grounding": "simulated", "note": "All profiles simulated."},
    "plumbing": {"name": "Plumbing repair", "types": ["human"], "grounding": "simulated", "note": "All profiles simulated."},
    "warehouse_picking": {"name": "Warehouse order picking", "types": ["robot", "human"], "grounding": "simulated", "note": "All profiles simulated."},
    "inspection": {"name": "Equipment inspection", "types": ["robot", "human"], "grounding": "simulated", "note": "All profiles simulated."},
}
# Assumed caliber range per (service, worker type) for simulated workers. Not measured.
ASSUMED_CALIBER = {
    ("bug_fix", "human"): (0.60, 0.85), ("parcel_pickup", "robot"): (0.80, 0.90),
    ("it_support", "agent"): (0.55, 0.75), ("it_support", "human"): (0.80, 0.92),
    ("scheduling", "agent"): (0.88, 0.97), ("scheduling", "human"): (0.85, 0.93),
    ("plumbing", "human"): (0.82, 0.95),
    ("warehouse_picking", "robot"): (0.93, 0.98), ("warehouse_picking", "human"): (0.88, 0.95),
    ("inspection", "robot"): (0.80, 0.92), ("inspection", "human"): (0.85, 0.95),
}
# Assumed work time in simulated minutes per (service, worker type); bug fixes also depend on difficulty.
DURATION = {("parcel_pickup", "human"): 30, ("parcel_pickup", "robot"): 40, ("it_support", "agent"): 5,
            ("it_support", "human"): 20, ("scheduling", "agent"): 2, ("scheduling", "human"): 10,
            ("plumbing", "human"): 60, ("warehouse_picking", "robot"): 6, ("warehouse_picking", "human"): 10,
            ("inspection", "robot"): 25, ("inspection", "human"): 40}
BUG_DURATION = {"agent": [8, 15, 30, 45], "human": [20, 45, 120, 240]}
SHORT = {"bug_fix": "DEV", "parcel_pickup": "DEL", "it_support": "IT", "scheduling": "SCH", "plumbing": "PLB",
         "warehouse_picking": "WH", "inspection": "INS"}
SIM_HUMANS = {"bug_fix": 3, "it_support": 3, "scheduling": 2, "plumbing": 4, "warehouse_picking": 3, "inspection": 3}
SIM_ROBOTS = {"parcel_pickup": 3, "warehouse_picking": 3, "inspection": 2}
SIM_AGENTS = {"it_support": 3, "scheduling": 2}


def _worker(wid, name, wtype, source, caliber):
    return {"id": wid, "name": name, "type": wtype, "source": source, "caliber": caliber,
            "capacity": CAPACITY[wtype], "cost": COST[wtype], "busy": [], "done": 0, "ok": 0}


class Portal:
    def __init__(self, rm: ReliabilityModel, seed: int = 7):
        self.rm, self.rng, self.lock = rm, np.random.default_rng(seed), threading.Lock()
        self.clock, self.speed, self.auto, self.rate = 0.0, 2.0, True, 8.0  # sim minutes; sim min per real second
        self.requests, self.next_id, self.task_cursor = [], 1, {}
        self.workers = self._roster()
        self._last = time.monotonic()

    # ---------------------------------------------------------------- roster
    def _roster(self):
        rm, ws = self.rm, []
        # Real AI agents: 8 submissions spread across the range of track records.
        order = np.argsort(rm.rating)
        picks = order[np.linspace(len(order) * 0.3, len(order) - 1, 8).astype(int)]
        for a in picks:
            ws.append(_worker(f"A-{a}", short_name(rm.agents[a]), "agent", "real (SWE-bench Verified)", {"bug_fix": None}))
            ws[-1]["agent_idx"] = int(a)
        # Real couriers: 8 Shanghai couriers with at least 40 pickups on the replayed day.
        d = load_day("Shanghai")
        g = d.groupby("courier_id").agg(n=("p_late", "size"), risk=("p_late", "mean"))
        g = g[g.n >= 40].sort_values("risk")
        for cid, row in g.iloc[np.linspace(0, len(g) - 1, 8).astype(int)].iterrows():
            ws.append(_worker(f"C-{cid}", f"Courier #{cid} (Shanghai)", "human", "real (LaDe-P)",
                              {"parcel_pickup": round(1 - float(row.risk), 3)}))
        # Simulated workers.
        for kind, table, prefix in (("human", SIM_HUMANS, "H"), ("robot", SIM_ROBOTS, "R"), ("agent", SIM_AGENTS, "S")):
            for svc, n in table.items():
                for i in range(n):
                    lo, hi = ASSUMED_CALIBER[(svc, kind)]
                    cal = {svc: round(float(self.rng.uniform(lo, hi)), 3)}
                    # Some simulated humans and agents also cover a second, related service.
                    if svc == "it_support" and kind in ("human", "agent") and i == 0:
                        lo2, hi2 = ASSUMED_CALIBER[("scheduling", kind)]
                        cal["scheduling"] = round(float(self.rng.uniform(lo2, hi2)), 3)
                    label = {"human": "Technician", "robot": "Robot", "agent": "Assistant agent"}[kind]
                    if svc == "bug_fix":
                        label = "Developer"
                    wid = f"{prefix}-{SHORT[svc]}-{i + 1}"
                    ws.append(_worker(wid, f"{label} {wid}", kind, "simulated", cal))
        return ws

    # ---------------------------------------------------------------- requests
    def _match_task(self, repo, difficulty):
        t = self.rm.tasks
        cand = np.nonzero((t.repo == repo).to_numpy() & (t.difficulty.fillna(1).astype(int) == difficulty).to_numpy())[0]
        if len(cand) == 0:
            cand = np.nonzero((t.repo == repo).to_numpy())[0]
        if len(cand) == 0:
            cand = np.arange(len(t))
        k = self.task_cursor.get((repo, difficulty), 0)
        self.task_cursor[(repo, difficulty)] = k + 1
        return int(cand[k % len(cand)])

    def submit(self, customer, service, preference="any", priority="balanced", repo=None, difficulty=1, client=None, auto=False):
        if service not in SERVICES:
            raise ValueError("unknown service")
        allowed = SERVICES[service]["types"] if preference == "any" else [preference]
        if not set(allowed) <= set(SERVICES[service]["types"]):
            raise ValueError(f"{TYPES.get(preference, preference)} workers do not offer this service")
        with self.lock:
            r = {"id": self.next_id, "customer": customer, "client": client, "auto": auto, "service": service,
                 "preference": preference, "allowed": allowed, "priority": priority, "created": self.clock,
                 "status": "waiting", "attempts": [], "timeline": [(self.clock, "Request received")]}
            if service == "bug_fix":
                repo = repo if repo in self.rm.repos else self.rng.choice(self.rm.repos)
                j = self._match_task(repo, int(difficulty))
                r["task"] = {"idx": j, "id": self.rm.tasks.instance_id.iloc[j], "repo": repo,
                             "difficulty": int(self.rm.tasks.difficulty.fillna(1).iloc[j])}
            self.next_id += 1
            self.requests.append(r)
            self._dispatch()
            return r["id"]

    def caliber(self, w, r):
        if r["service"] == "bug_fix":
            if w["type"] == "agent":
                return float(self.rm.P[r["task"]["idx"], w["agent_idx"]])
            return max(0.05, w["caliber"]["bug_fix"] - 0.07 * r["task"]["difficulty"])
        return w["caliber"][r["service"]]

    def candidates(self, r, exclude=()):
        out = []
        for w in self.workers:
            if r["service"] not in w["caliber"] or w["type"] not in r["allowed"] or w["id"] in exclude:
                continue
            cal = self.caliber(w, r)
            out.append({"worker": w["id"], "name": w["name"], "type": w["type"], "source": w["source"],
                        "caliber": round(cal, 3), "cost": w["cost"], "free": len(w["busy"]) < w["capacity"],
                        "score": round(cal - PRIORITY[r["priority"]] * w["cost"] / max(COST.values()), 3)})
        return sorted(out, key=lambda c: -c["score"])

    def _duration(self, w, r):
        if r["service"] == "bug_fix":
            return BUG_DURATION[w["type"]][r["task"]["difficulty"]]
        return DURATION[(r["service"], w["type"])]

    def _dispatch(self):
        """Oldest waiting request first; each takes the best-scoring free eligible worker."""
        by_id = {w["id"]: w for w in self.workers}
        for r in self.requests:
            if r["status"] != "waiting":
                continue
            tried = {a["worker"] for a in r["attempts"]}
            cands = self.candidates(r, exclude=tried)
            free = [c for c in cands if c["free"]]
            if not cands:
                r["status"], r["result"] = "failed", False
                r["timeline"].append((self.clock, "No other eligible worker left"))
                continue
            if not free:
                continue
            c = free[0]
            w = by_id[c["worker"]]
            dur = self._duration(w, r)
            a = {"worker": w["id"], "name": w["name"], "type": w["type"], "caliber": c["caliber"], "start": self.clock,
                 "end": self.clock + dur, "candidates": cands, "reason": self._reason(r, c, cands)}
            w["busy"].append(r["id"])
            r["attempts"].append(a)
            r["status"] = "in_progress"
            r["timeline"].append((self.clock, f"Assigned to {w['name']} ({TYPES[w['type']]}, caliber {c['caliber']:.0%})"))

    @staticmethod
    def _reason(r, c, cands):
        busy_better = [x for x in cands if not x["free"] and x["score"] > c["score"]]
        txt = f"Highest score among free {', '.join(TYPES[t] for t in r['allowed'])} workers for this job"
        if r["priority"] != "quality":
            txt += f" (score = caliber − cost weight, priority '{r['priority']}')"
        if busy_better:
            txt += f"; {len(busy_better)} better-scoring worker(s) were busy"
        return txt + "."

    def _finish(self, r, a, w):
        if r["service"] == "bug_fix" and w["type"] == "agent":
            ok, src = bool(self.rm.Y[r["task"]["idx"], w["agent_idx"]]), "recorded SWE-bench result"
        else:
            ok, src = bool(self.rng.random() < a["caliber"]), "simulated from caliber"
        a.update(success=ok, outcome_source=src)
        w["busy"].remove(r["id"])
        w["done"] += 1
        w["ok"] += ok
        if ok:
            r["status"], r["result"] = "done", True
            r["timeline"].append((self.clock, f"Completed by {w['name']} ✓ ({src})"))
        elif len(r["attempts"]) < 2:
            r["status"] = "waiting"
            r["timeline"].append((self.clock, f"{w['name']} could not complete it ({src}); re-dispatching to the next best worker"))
        else:
            r["status"], r["result"] = "failed", False
            r["timeline"].append((self.clock, f"{w['name']} could not complete it ({src}); request closed as failed"))

    # ---------------------------------------------------------------- clock
    def tick(self):
        now = time.monotonic()
        dt, self._last = now - self._last, now
        with self.lock:
            new_clock = self.clock + dt * self.speed
            by_id = {w["id"]: w for w in self.workers}
            # finish jobs in end-time order up to the new clock
            running = sorted(((a["end"], r["id"], r, a) for r in self.requests if r["status"] == "in_progress"
                              for a in r["attempts"][-1:] if "success" not in a), key=lambda x: (x[0], x[1]))
            for end, _, r, a in running:
                if end > new_clock:
                    break
                self.clock = end
                self._finish(r, a, by_id[a["worker"]])
                self._dispatch()
            self.clock = new_clock
            if self.auto and self.rng.random() < 1 - np.exp(-self.rate / 60 * dt * self.speed):
                auto_job = True
            else:
                auto_job = False
            self._dispatch()
        if auto_job:
            svc = self.rng.choice(list(SERVICES))
            pref = self.rng.choice(["any", "any", "any"] + SERVICES[svc]["types"])
            self.submit(f"Customer {self.rng.integers(1000, 9999)}", svc, pref, self.rng.choice(list(PRIORITY)),
                        repo=self.rng.choice(self.rm.repos), difficulty=int(self.rng.integers(0, 3)), auto=True)

    # ---------------------------------------------------------------- views
    def catalog(self):
        return {"services": [{"id": k, **v, "type_labels": [TYPES[t] for t in v["types"]],
                              "workers": {t: sum(1 for w in self.workers if k in w["caliber"] and w["type"] == t) for t in v["types"]}}
                             for k, v in SERVICES.items()],
                "types": TYPES, "priorities": list(PRIORITY), "repos": self.rm.repos, "difficulty": DIFF_LABEL,
                "assumptions": {"caliber": [{"service": SERVICES[s]["name"], "type": TYPES[t], "range": f"{lo:.0%}–{hi:.0%}"}
                                            for (s, t), (lo, hi) in ASSUMED_CALIBER.items()],
                                "cost": {TYPES[t]: c for t, c in COST.items()}, "capacity": {TYPES[t]: c for t, c in CAPACITY.items()}}}

    def state(self, client=None, limit=80):
        with self.lock:
            reqs = self.requests[-limit:] if client is None else [r for r in self.requests if r["client"] == client]
            stats = {}
            for t in TYPES:
                done = [a for r in self.requests for a in r["attempts"] if a["type"] == t and "success" in a]
                stats[t] = {"jobs": len(done), "ok": sum(a["success"] for a in done),
                            "busy": sum(len(w["busy"]) for w in self.workers if w["type"] == t),
                            "capacity": sum(w["capacity"] for w in self.workers if w["type"] == t)}
            closed = [r for r in self.requests if r["status"] in ("done", "failed")]
            return {"clock": self.clock, "speed": self.speed, "auto": self.auto, "rate": self.rate,
                    "totals": {"requests": len(self.requests), "waiting": sum(r["status"] == "waiting" for r in self.requests),
                               "in_progress": sum(r["status"] == "in_progress" for r in self.requests),
                               "done": sum(r["status"] == "done" for r in self.requests),
                               "failed": sum(r["status"] == "failed" for r in self.requests),
                               "first_try": sum(1 for r in closed if r["attempts"] and r["attempts"][0].get("success")),
                               "closed": len(closed)},
                    "by_type": stats,
                    "workers": [{k: w[k] for k in ("id", "name", "type", "source", "capacity", "done", "ok")}
                                | {"busy": list(w["busy"]), "services": {s: (None if v is None else v) for s, v in w["caliber"].items()}}
                                for w in self.workers],
                    "requests": [self._view(r) for r in reqs]}

    def _view(self, r):
        v = {k: r[k] for k in ("id", "customer", "auto", "service", "preference", "priority", "created", "status")}
        v["service_name"] = SERVICES[r["service"]]["name"]
        v["timeline"] = [{"t": t, "text": s} for t, s in r["timeline"]]
        v["attempts"] = [{k: a[k] for k in a if k != "candidates"} | {"candidates": a["candidates"][:12]} for a in r["attempts"]]
        if "task" in r:
            v["task"] = r["task"] | {"difficulty_label": DIFF_LABEL[r["task"]["difficulty"]]}
        return v


def run_clock(portal: Portal, stop: threading.Event, every: float = 0.25):
    while not stop.is_set():
        portal.tick()
        stop.wait(every)
