"""SWE-bench Verified agent x task outcomes as a long table, with task metadata."""
import glob
import json
import os

import numpy as np
import pandas as pd

RESULTS_GLOB = "data/raw/swebench_experiments/evaluation/verified/*/results/results.json"
TASKS = "data/raw/swebench_tasks/verified_test.parquet"
DIFFICULTY = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}


def load_tasks() -> pd.DataFrame:
    t = pd.read_parquet(TASKS, columns=["instance_id", "repo", "created_at", "difficulty", "problem_statement"])
    t["difficulty"] = t.difficulty.map(DIFFICULTY)
    t["year"] = pd.to_datetime(t.created_at).dt.year
    t["created"] = pd.to_datetime(t.created_at)
    t["statement_len"] = t.problem_statement.str.len()
    return t.drop(columns=["created_at", "problem_statement"])


def load_outcomes(tasks: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, task). Assumes every agent attempted all tasks; unresolved = not listed."""
    ids = tasks.instance_id.tolist()
    rows = []
    for f in sorted(glob.glob(RESULTS_GLOB)):
        agent = f.split("/")[-3]
        resolved = set(json.load(open(f)).get("resolved", []))
        date = pd.to_datetime(agent[:8], format="%Y%m%d", errors="coerce")
        rows.append(pd.DataFrame({
            "agent": agent, "instance_id": ids,
            "resolved": [int(i in resolved) for i in ids], "agent_date": date,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["agent_days"] = (df.agent_date - df.agent_date.min()).dt.days
    return df.merge(tasks, on="instance_id")


if __name__ == "__main__":
    out = load_outcomes(load_tasks())
    print(out.shape, "agents", out.agent.nunique(), "tasks", out.instance_id.nunique())
    print("overall resolve rate", out.resolved.mean().round(3))
    print(out.groupby("agent").resolved.mean().describe().round(3).to_dict())
    print(out.groupby("difficulty").resolved.mean().round(3).to_dict())
    print(out.groupby("repo").resolved.mean().round(3).sort_values().to_dict())
    print("tasks solved by no agent:", int((out.groupby("instance_id").resolved.sum() == 0).sum()))
