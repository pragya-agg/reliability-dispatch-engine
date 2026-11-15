"""Predict whether an AI agent resolves a SWE-bench Verified task, then test agent selection."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from src.swebench_data import load_outcomes, load_tasks

GROUPS = {"agent_rate": ["agent"], "agent_repo_rate": ["agent", "repo"], "agent_diff_rate": ["agent", "difficulty"]}
FEATURES = ["agent_rate", "agent_repo_rate", "agent_diff_rate", "agent_days", "difficulty", "repo", "year", "statement_len"]
SMOOTH = 5  # pseudo-observations pulling a sparse group rate towards its parent rate
POOL_SIZE, DRAWS = 5, 200
OUT = "results"


def add_history(train: pd.DataFrame, test: pd.DataFrame):
    """Agent track-record features from training tasks only; train rows use leave-one-out."""
    tr, te = train.copy(), test.copy()
    prior_tr = prior_te = train.resolved.mean()  # agent_rate shrinks to the global rate, group rates to agent_rate
    for name, keys in GROUPS.items():
        s = train.groupby(keys).resolved.agg(s="sum", n="count").reset_index()
        tr, te = tr.merge(s, on=keys, how="left"), te.merge(s, on=keys, how="left")
        tr[name] = (tr.s - tr.resolved + SMOOTH * prior_tr) / (tr.n - 1 + SMOOTH)
        te[name] = (te.s.fillna(0) + SMOOTH * prior_te) / (te.n.fillna(0) + SMOOTH)
        tr, te = tr.drop(columns=["s", "n"]), te.drop(columns=["s", "n"])
        if name == "agent_rate":
            prior_tr, prior_te = tr[name], te[name]
    return tr, te


def fit_predict(train, test):
    tr, te = add_history(train, test)
    for d in (tr, te):
        d["repo"] = pd.Categorical(d.repo, categories=sorted(set(train.repo) | set(test.repo)))
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          n_jobs=-1, random_state=0)
    model.fit(tr[FEATURES], tr.resolved)
    te["p_model"] = model.predict_proba(te[FEATURES])[:, 1]
    te["p_agent_only"] = te.agent_rate
    te["p_const"] = train.resolved.mean()
    return te, model


def dispatch_eval(te: pd.DataFrame, rng, pool_size):
    """For each test task, choose among random agent pools; outcomes are real (all agents ran every task)."""
    res = {"random": [], "best-overall agent": [], "model pick": [], "oracle (any can solve)": []}
    for _, g in te.groupby("instance_id"):
        y, p, base = g.resolved.to_numpy(), g.p_model.to_numpy(), g.agent_rate.to_numpy()
        per_task = {k: [] for k in res}
        for _ in range(DRAWS if pool_size < len(g) else 1):
            idx = rng.choice(len(g), pool_size, replace=False) if pool_size < len(g) else np.arange(len(g))
            per_task["random"].append(y[idx].mean())
            per_task["best-overall agent"].append(y[idx[np.argmax(base[idx])]])
            per_task["model pick"].append(y[idx[np.argmax(p[idx])]])
            per_task["oracle (any can solve)"].append(y[idx].max())
        for k, v in per_task.items():
            res[k].append(np.mean(v))
    return {k: np.array(v) for k, v in res.items()}  # one value per task


def report(name, te, rng):
    y = te.resolved
    rows = [{"scheme": name, "model": m, "roc_auc": roc_auc_score(y, te[c]), "pr_auc": average_precision_score(y, te[c]),
             "log_loss": log_loss(y, np.clip(te[c], 1e-4, 1 - 1e-4))}
            for m, c in [("Constant (base rate)", "p_const"), ("Agent overall rate only", "p_agent_only"),
                         ("XGBoost (agent history + task)", "p_model")]]
    disp = []
    for k in (POOL_SIZE, te.agent.nunique()):
        per_task = dispatch_eval(te, rng, k)
        diff = per_task["model pick"] - per_task["best-overall agent"]
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        disp += [{"scheme": name, "pool": f"{k} agents", "method": m, "success_rate": v.mean(),
                  "model_minus_best_overall": diff.mean(), "ci_low": lo, "ci_high": hi}
                 for m, v in per_task.items()]
    return pd.DataFrame(rows), pd.DataFrame(disp)


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = load_tasks()
    data = load_outcomes(tasks)
    rng = np.random.default_rng(0)

    # Chronological: newest 25% of tasks are held out.
    by_date = tasks.sort_values("created")
    test_ids = set(by_date.instance_id.iloc[int(len(by_date) * 0.75):])
    te, model = fit_predict(data[~data.instance_id.isin(test_ids)], data[data.instance_id.isin(test_ids)])
    m1, d1 = report("chronological (newest 25% tasks)", te, rng)

    # Random 5-fold over tasks, predictions pooled.
    parts = []
    ids = np.array(sorted(data.instance_id.unique()))
    for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(ids):
        parts.append(fit_predict(data[data.instance_id.isin(ids[tr_i])], data[data.instance_id.isin(ids[te_i])])[0])
    m2, d2 = report("random 5-fold over tasks", pd.concat(parts), rng)

    metrics, dispatch = pd.concat([m1, m2]), pd.concat([d1, d2])
    metrics.to_csv(f"{OUT}/swebench_prediction.csv", index=False)
    dispatch.to_csv(f"{OUT}/swebench_dispatch.csv", index=False)
    print(metrics.round(3).to_string(index=False))
    print()
    print(dispatch.pivot_table(index=["scheme", "pool"], columns="method", values="success_rate").round(3).to_string())
    print("\nmodel pick minus best-overall agent (mean, 95% bootstrap CI over tasks):")
    print(dispatch.drop_duplicates(["scheme", "pool"])[["scheme", "pool", "model_minus_best_overall", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print("\ntest tasks (chronological):", te.instance_id.nunique())


if __name__ == "__main__":
    main()
