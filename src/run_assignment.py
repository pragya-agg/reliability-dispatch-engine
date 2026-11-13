"""Capacity-limited dispatch experiment on SWE-bench Verified.

Every agent attempted every task, so the real outcome of ANY assignment can be looked up.
A batch of unseen tasks is given to a random pool of agents, each with a job limit, and the
share of tasks actually resolved is compared across assignment methods.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.assignment import assign_by_track_record, assign_greedy, assign_random, solve_assignment
from src.swebench_data import load_outcomes, load_tasks
from src.train_swebench import fit_predict

PRED_CACHE = "data/swebench_predictions.parquet"
OUT = "results"
POOLS = 300          # random agent pools per scenario
POOL_SIZES = (5, 10)
SLACKS = (1.0, 1.5, 2.0)  # total capacity / number of tasks; 1.0 = every slot must be used
METHODS = ["random", "track record", "model + greedy", "model + optimal", "oracle"]


def out_of_fold_predictions(use_cache: bool = True) -> pd.DataFrame:
    """Predicted success for every (agent, task); each task is scored by a model that never saw it."""
    if use_cache and os.path.exists(PRED_CACHE):
        return pd.read_parquet(PRED_CACHE)
    data = load_outcomes(load_tasks())
    ids = np.array(sorted(data.instance_id.unique()))
    parts = []
    for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(ids):
        te, _ = fit_predict(data[data.instance_id.isin(ids[tr_i])], data[data.instance_id.isin(ids[te_i])])
        parts.append(te)
    keep = ["agent", "instance_id", "repo", "difficulty", "resolved", "p_model", "agent_rate"]
    pred = pd.concat(parts)[keep].reset_index(drop=True)
    pred.to_parquet(PRED_CACHE)
    return pred


def as_matrices(pred: pd.DataFrame):
    """Return (tasks, agents, Y, P, R): outcome, prediction and agent-rating matrices, tasks x agents."""
    Y = pred.pivot(index="instance_id", columns="agent", values="resolved")
    P = pred.pivot(index="instance_id", columns="agent", values="p_model")[Y.columns]
    R = pred.pivot(index="instance_id", columns="agent", values="agent_rate")[Y.columns]
    return Y.index.to_numpy(), Y.columns.to_numpy(), Y.to_numpy(), P.to_numpy(), R.to_numpy()


def assign_all(y, p, r, cap, rng) -> dict:
    """One assignment per method for one pool. y, p, r are tasks x pool-agents."""
    n, k = y.shape
    return {
        "random": assign_random(n, k, cap, rng),
        "track record": assign_by_track_record(r.mean(0), n, cap, rng),
        "model + greedy": assign_greedy(p, cap, rng),
        "model + optimal": solve_assignment(p, cap),
        "oracle": solve_assignment(y.astype(float), cap),
    }


def run_scenario(Y, P, R, k, slack, rng):
    """Per-task success (averaged over pools) for each method."""
    n = Y.shape[0]
    cap = int(np.ceil(slack * n / k))
    hits = {m: np.zeros(n) for m in METHODS}
    for _ in range(POOLS):
        a = rng.choice(Y.shape[1], k, replace=False)
        y = Y[:, a]
        for m, pick in assign_all(y, P[:, a], R[:, a], cap, rng).items():
            hits[m] += y[np.arange(n), pick]
    return {m: v / POOLS for m, v in hits.items()}, cap


def bootstrap_ci(diff, rng, n=2000):
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    _, _, Y, P, R = as_matrices(out_of_fold_predictions(use_cache=False))
    rows = []
    for k in POOL_SIZES:
        for slack in SLACKS:
            per_task, cap = run_scenario(Y, P, R, k, slack, rng)
            row = {"agents_in_pool": k, "jobs_per_agent_limit": cap, "capacity_vs_jobs": slack}
            row.update({m: per_task[m].mean() for m in METHODS})
            for base in ("track record", "model + greedy"):
                d = per_task["model + optimal"] - per_task[base]
                lo, hi = bootstrap_ci(d, rng)
                tag = base.replace(" + ", "_").replace(" ", "_")
                row.update({f"optimal_minus_{tag}": d.mean(), f"{tag}_ci_low": lo, f"{tag}_ci_high": hi})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/swebench_assignment.csv", index=False)
    print(f"{Y.shape[0]} tasks, {Y.shape[1]} agents, {POOLS} random pools per row; success = share of tasks resolved\n")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
