"""Live dispatch engine for AI agents, replaying real SWE-bench Verified tasks.

What is real:  the tasks, their order (issue creation date), and the outcome of every
               (agent, task) pair, because every agent attempted every task.
What is assumed: arrival timing (Poisson stream), work time per job (from the SWE-bench
               difficulty label, which is an estimated *human* fix time), and agent capacity.

The reliability model is trained only on the older tasks; the replay uses the newer tasks
it never saw. While the replay runs, the "learning" dispatcher also updates each agent's
caliber from the outcomes it observes (a Beta-style update with the model as the prior).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from xgboost import XGBClassifier

from src.swebench_data import load_outcomes, load_tasks
from src.train_swebench import FEATURES, add_history

TRAIN_FRAC = 0.6                                # oldest 60% of tasks train the model, newest 40% are replayed
WORK_MIN = {0: 10, 1: 35, 2: 150, 3: 300}       # assumed minutes of work per difficulty level
DIFF_LABEL = {0: "<15 min", 1: "15 min-1 h", 2: "1-4 h", 3: ">4 h"}
PRIOR_WEIGHT = 4.0                              # how many observed outcomes the model's estimate is worth
POLICIES = ["random", "track record", "model", "learning"]
POLICY_LABEL = {"random": "Random", "track record": "Best track record first",
                "model": "Reliability model + optimal", "learning": "Model + online learning + optimal"}


def short_name(agent: str) -> str:
    return agent[9:] if agent[:8].isdigit() else agent


class ReliabilityModel:
    """XGBoost reliability model trained on the older tasks; scores any (agent, task) pair."""

    def __init__(self, train_frac: float = TRAIN_FRAC):
        tasks = load_tasks()
        data = load_outcomes(tasks)
        order = tasks.sort_values("created").instance_id.to_numpy()
        cut = int(len(order) * train_frac)
        self.train_ids, self.replay_ids = order[:cut], order[cut:]
        self.train = data[data.instance_id.isin(self.train_ids)]
        replay = data[data.instance_id.isin(self.replay_ids)]
        self.repos = sorted(set(data.repo))
        tr, te = add_history(self.train, replay)
        for d in (tr, te):
            d["repo"] = pd.Categorical(d.repo, categories=self.repos)
        self.model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                                   n_jobs=-1, random_state=0).fit(tr[FEATURES], tr.resolved)
        te["p"] = self.model.predict_proba(te[FEATURES])[:, 1]
        self.agents = np.array(sorted(data.agent.unique()))
        self.tasks = tasks.set_index("instance_id").loc[self.replay_ids].reset_index()
        piv = lambda col: te.pivot(index="instance_id", columns="agent", values=col).loc[self.replay_ids, self.agents].to_numpy()
        self.Y, self.P = piv("resolved").astype(int), piv("p")
        self.rating = self.train.groupby("agent").resolved.mean().loc[self.agents].to_numpy()
        self.n_train_tasks, self.n_train_rows = len(self.train_ids), len(self.train)
        self.agent_days = data.groupby("agent").agent_days.first().loc[self.agents].to_numpy()

    def score_new_job(self, agent_idx, repo: str, difficulty: int, statement_len: int, year: int) -> np.ndarray:
        """Predicted success of each listed agent on a job that is not in the data (no outcome known)."""
        rows = pd.DataFrame({"agent": self.agents[agent_idx], "instance_id": "new", "resolved": 0,
                             "repo": repo, "difficulty": difficulty, "statement_len": statement_len,
                             "year": year, "agent_days": self.agent_days[agent_idx]})
        _, te = add_history(self.train, rows)
        te["repo"] = pd.Categorical(te.repo, categories=self.repos)
        return self.model.predict_proba(te[FEATURES])[:, 1]


@dataclass
class World:
    """State of one dispatch policy. Every policy sees the same arriving jobs."""
    name: str
    free: np.ndarray
    queue: list = field(default_factory=list)
    running: list = field(default_factory=list)      # (end_minute, job, agent)
    done: dict = field(default_factory=dict)         # job -> (agent, start, end, outcome)
    live_s: dict = field(default_factory=dict)       # (agent, repo) -> successes seen live
    live_n: dict = field(default_factory=dict)
    agent_s: np.ndarray = None
    agent_n: np.ndarray = None


class LiveDispatch:
    def __init__(self, rm: ReliabilityModel, pool_size=8, slots=2, load=0.9, batch_minutes=30, seed=1):
        self.rm, self.rng = rm, np.random.default_rng(seed)
        self.pool = np.sort(self.rng.choice(len(rm.agents), pool_size, replace=False))
        self.K, self.slots, self.batch = pool_size, slots, batch_minutes
        self.Y, self.P, self.rating = rm.Y[:, self.pool], rm.P[:, self.pool], rm.rating[self.pool]
        t = rm.tasks
        self.repo = t.repo.to_numpy()
        self.diff = t.difficulty.fillna(1).astype(int).to_numpy()
        self.work = np.array([WORK_MIN[d] for d in self.diff])
        rate_per_min = load * pool_size * slots / self.work.mean()   # load = expected share of busy slots
        self.arrival = np.floor(np.cumsum(self.rng.exponential(1 / rate_per_min, len(t)))).astype(int)
        self.n = len(t)
        self.now, self.next_job = 0, 0
        self.worlds = {p: World(p, np.full(pool_size, slots), agent_s=np.zeros(pool_size), agent_n=np.zeros(pool_size))
                       for p in POLICIES}
        self.policy_rng = {p: np.random.default_rng(seed * 100 + i) for i, p in enumerate(POLICIES)}

    # ---- caliber: how each policy scores agent k for job j -------------------------------
    def caliber(self, w: World, jobs: list) -> np.ndarray:
        J = np.array(jobs)
        if w.name == "random":
            return self.policy_rng["random"].random((len(J), self.K))
        if w.name == "track record":
            n0 = self.rm.n_train_tasks
            r = (self.rating * n0 + w.agent_s) / (n0 + w.agent_n)
            return np.tile(r, (len(J), 1))
        prior = self.P[J]
        if w.name == "model":
            return prior
        s = np.array([[w.live_s.get((k, self.repo[j]), 0) for k in range(self.K)] for j in J])
        n = np.array([[w.live_n.get((k, self.repo[j]), 0) for k in range(self.K)] for j in J])
        return (PRIOR_WEIGHT * prior + s) / (PRIOR_WEIGHT + n)

    # ---- one dispatch round ----------------------------------------------------------------
    def dispatch(self, w: World) -> list:
        slots = np.repeat(np.arange(self.K), w.free)
        if len(slots) == 0 or not w.queue:
            return []
        jobs = w.queue[: len(slots)]                     # oldest jobs first, so no job waits forever
        score = self.caliber(w, jobs)
        if w.name in ("random", "track record"):         # one job at a time, best free agent by rating
            pick, free = [], w.free.copy()
            for i in range(len(jobs)):
                k = int(np.argmax(np.where(free > 0, score[i], -np.inf)))
                pick.append(k)
                free[k] -= 1
        else:                                            # whole batch at once, optimal matching
            r, c = linear_sum_assignment(-score[:, slots])
            pick = list(slots[c[np.argsort(r)]])
        events = []
        for i, (j, k) in enumerate(zip(jobs, pick)):
            w.free[k] -= 1
            w.running.append((self.now + self.work[j], j, k))
            events.append({"type": "assign", "job": int(j), "agent": int(k), "t": self.now,
                           "wait": int(self.now - self.arrival[j]), "score": round(float(score[i, k]), 3),
                           "scores": [round(float(x), 3) for x in score[i]]})
        w.queue = w.queue[len(jobs):]
        return events

    def complete(self, w: World) -> list:
        events, still = [], []
        for end, j, k in w.running:
            if end > self.now:
                still.append((end, j, k))
                continue
            y = int(self.Y[j, k])
            w.free[k] += 1
            w.done[j] = (k, end - self.work[j], end, y)
            w.agent_s[k] += y
            w.agent_n[k] += 1
            key = (k, self.repo[j])
            w.live_s[key] = w.live_s.get(key, 0) + y
            w.live_n[key] = w.live_n.get(key, 0) + 1
            events.append({"type": "complete", "job": int(j), "agent": int(k), "t": self.now, "resolved": y})
        w.running = still
        return events

    @property
    def finished(self) -> bool:
        return all(len(w.done) == self.n for w in self.worlds.values())

    def step(self, minutes: int = 1) -> dict:
        """Advance the clock; return what happened to the featured policy plus every policy's totals."""
        events = []
        for _ in range(minutes):
            if self.finished:
                break
            while self.next_job < self.n and self.arrival[self.next_job] <= self.now:
                for w in self.worlds.values():
                    w.queue.append(self.next_job)
                events.append({"type": "arrive", "job": self.next_job, "t": self.now})
                self.next_job += 1
            for name, w in self.worlds.items():
                ev = self.complete(w)
                if self.now % self.batch == 0:
                    ev += self.dispatch(w)
                if name == "learning":
                    events += ev
            self.now += 1
        return {"now": self.now, "finished": self.finished, "events": events, "state": self.state()}

    # ---- views for the dashboard ------------------------------------------------------------
    def job_info(self, j: int) -> dict:
        return {"id": self.rm.tasks.instance_id.iloc[j], "repo": self.repo[j], "difficulty": DIFF_LABEL[self.diff[j]],
                "work_min": int(self.work[j]), "arrival": int(self.arrival[j]),
                "solvable_by": int(self.Y[j].sum())}

    def summary(self, w: World) -> dict:
        res = [v[3] for v in w.done.values()]
        waits = [v[1] - self.arrival[j] for j, v in w.done.items()]
        return {"policy": w.name, "label": POLICY_LABEL[w.name], "done": len(res), "resolved": int(sum(res)),
                "rate": float(np.mean(res)) if res else None, "avg_wait": float(np.mean(waits)) if waits else None,
                "queue": len(w.queue), "running": len(w.running)}

    def state(self) -> dict:
        w = self.worlds["learning"]
        busy = self.slots - w.free
        cal = self.caliber(w, list(range(self.n)))       # current caliber matrix of the learning policy
        agents = []
        for k in range(self.K):
            jobs_k = [v for v in w.done.values() if v[0] == k]
            agents.append({"idx": k, "name": short_name(self.rm.agents[self.pool[k]]), "busy": int(busy[k]),
                           "slots": self.slots, "rating": round(float(self.rating[k]), 3),
                           "done": len(jobs_k), "resolved": int(sum(v[3] for v in jobs_k)),
                           "caliber_by_repo": {r: round(float(cal[np.argmax(self.repo == r), k]), 3)
                                               for r in sorted(set(self.repo))},
                           "running": [int(j) for e, j, kk in w.running if kk == k]})
        oracle = float(np.mean(self.Y[: max(self.next_job, 1)].max(1)))
        return {"now": self.now, "arrived": self.next_job, "total": self.n, "queue": [int(j) for j in w.queue],
                "agents": agents, "policies": [self.summary(x) for x in self.worlds.values()],
                "oracle_rate_so_far": oracle}


def evaluate(rm: ReliabilityModel, runs: int = 200, **kw) -> pd.DataFrame:
    """Repeat whole replays with different agent pools and arrival streams."""
    rows = []
    for seed in range(runs):
        sim = LiveDispatch(rm, seed=seed, **kw)
        while not sim.finished:
            sim.step(60)
        w = sim.worlds["learning"]
        rows.append({p: np.mean([v[3] for v in x.done.values()]) for p, x in sim.worlds.items()}
                    | {"oracle": sim.Y.max(1).mean(),
                       "avg_wait_min": np.mean([v[1] - sim.arrival[j] for j, v in w.done.items()])})
    return pd.DataFrame(rows)


def main(runs: int = 200):
    """Compare dispatch policies over many replays at three load levels; writes results/live_dispatch_eval.csv."""
    rm = ReliabilityModel()
    out = []
    for load in (0.6, 0.9, 1.2):
        d = evaluate(rm, runs=runs, load=load)
        row = {"load": load, "runs": runs, **d.mean().round(4).to_dict()}
        for a, b in [("model", "track record"), ("learning", "track record"), ("learning", "model")]:
            diff = d[a] - d[b]
            half = 1.96 * diff.std() / np.sqrt(len(diff))
            row[f"{a} - {b}"] = round(diff.mean(), 4)
            row[f"{a} - {b} (95% CI)"] = f"{diff.mean() - half:+.3f} to {diff.mean() + half:+.3f}"
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv("results/live_dispatch_eval.csv", index=False)
    print(f"{len(rm.replay_ids)} replayed tasks (newest 40%), model trained on {rm.n_train_tasks} older tasks; "
          f"8 agents x 2 slots, {runs} runs per load level")
    print(res.T.to_string(header=False))


if __name__ == "__main__":
    main()
