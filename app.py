"""Dashboard for the Workforce Dispatch & Reliability Engine.   Run:  streamlit run app.py"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.assignment import assign_by_track_record, assign_random, solve_assignment
from src.run_assignment import as_matrices, out_of_fold_predictions

st.set_page_config(page_title="Dispatch & Reliability Engine", layout="wide")
LADE_PRED = "data/lade_test_predictions.parquet"


@st.cache_data
def csv(name):
    return pd.read_csv(f"results/{name}")


@st.cache_data
def agent_predictions():
    return out_of_fold_predictions()


@st.cache_data
def courier_risk():
    """Test jobs sorted from riskiest to safest, with running totals for the flagging slider."""
    d = pd.read_parquet(LADE_PRED).sort_values("p_late", ascending=False).reset_index(drop=True)
    d["late_found"] = d.late.cumsum()
    return d


def need(path, command):
    if os.path.exists(path):
        return True
    st.warning(f"`{path}` not found. Run `{command}` first.")
    return False


st.title("AI-Driven Workforce Dispatch & Reliability Engine")
st.caption("Reliability scoring and job assignment for human workers (LaDe-P couriers) and AI agents "
           "(SWE-bench Verified). All numbers are from held-out data the models never trained on.")
overview, humans, agents, dispatch = st.tabs(["Overview", "Human workers", "AI agents", "Dispatch simulator"])

with overview:
    if need("results/lade_results.csv", "python run_all.py"):
        lade, swe, asg = csv("lade_results.csv"), csv("swebench_prediction.csv"), csv("swebench_assignment.csv")
        l_model, l_hist = lade.iloc[-1], lade[lade.model == "Courier past late rate only"].iloc[0]
        s = swe[swe.scheme.str.startswith("chrono")].set_index("model")
        tight = asg[(asg.agents_in_pool == 10) & (asg.capacity_vs_jobs == 1.0)].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Couriers: late-pickup model ROC-AUC", f"{l_model.roc_auc:.3f}",
                  f"{l_model.roc_auc - l_hist.roc_auc:+.3f} vs courier history only")
        c2.metric("AI agents: task-success model ROC-AUC", f"{s.roc_auc.iloc[-1]:.3f}",
                  f"{s.roc_auc.iloc[-1] - s.roc_auc.iloc[1]:+.3f} vs agent rating only")
        c3.metric("Dispatch: tasks resolved (10 agents, tight capacity)", f"{tight['model + optimal']:.1%}",
                  f"{tight.optimal_minus_track_record * 100:+.1f} pts vs best-track-record rule")
        st.subheader("Pipeline")
        st.markdown("**1. Outcome logs** → **2. Past-only worker history features** → **3. Reliability model "
                    "(XGBoost)** → **4. Capacity-limited optimal assignment** → **5. Comparison with baselines**")
        st.subheader("What is and is not shown")
        st.markdown(
            "- Couriers: the model is only slightly better than a courier's own past late rate.\n"
            "- AI agents: with no job limits, picking by model is not significantly better than always using "
            "the best-rated agent.\n"
            "- With job limits per agent, optimal assignment is better than the best-track-record rule by "
            "about 1 to 2.5 points, and the 95% confidence interval excludes zero.\n"
            "- Assignment is tested only on AI agents, because only there is the outcome of every "
            "worker-job pair known. Robots are not part of this study.")

with humans:
    if need(LADE_PRED, "python -m src.train_lade"):
        st.subheader("Predicting late pickups (test period, 1.53 million jobs)")
        st.dataframe(csv("lade_results.csv").round(4), hide_index=True, width="stretch")
        d = courier_risk()
        pct = st.slider("Flag the riskiest X% of jobs for the dispatcher", 1, 50, 10)
        k = int(len(d) * pct / 100)
        found, total = int(d.late_found.iloc[k - 1]), int(d.late.sum())
        a, b, c = st.columns(3)
        a.metric("Jobs flagged", f"{k:,}")
        b.metric("Late pickups caught", f"{found / total:.1%}", f"{found:,} of {total:,}")
        c.metric("Flagged jobs that were really late", f"{found / k:.1%}", f"base rate {total / len(d):.1%}")
        left, right = st.columns(2)
        left.image("results/lade_pr_curves.png")
        by = d.groupby("courier_id").agg(jobs=("late", "size"), predicted_late_rate=("p_late", "mean"),
                                         actual_late_rate=("late", "mean"), city=("city", "first"))
        right.markdown("**Couriers with the highest predicted risk** (at least 200 test jobs)")
        right.dataframe(by[by.jobs >= 200].sort_values("predicted_late_rate", ascending=False).head(15).round(3))

with agents:
    if need("results/swebench_prediction.csv", "python -m src.train_swebench"):
        pred = agent_predictions()
        st.subheader("Predicting whether an AI agent resolves a task")
        st.dataframe(csv("swebench_prediction.csv").round(3), hide_index=True, width="stretch")
        task = st.selectbox("Pick a task (the model never saw this task in training)", sorted(pred.instance_id.unique()))
        g = pred[pred.instance_id == task].sort_values("p_model", ascending=False)
        st.caption(f"Repository: {g.repo.iloc[0]} · {int(g.resolved.sum())} of {len(g)} agents actually resolved it")
        show = g.head(15)[["agent", "p_model", "agent_rate", "resolved"]].rename(columns={
            "p_model": "predicted success", "agent_rate": "agent overall rating", "resolved": "actually resolved"})
        st.dataframe(show.round(3), hide_index=True, width="stretch")

with dispatch:
    if need("results/swebench_assignment.csv", "python -m src.run_assignment"):
        st.subheader("Dispatch a batch of tasks to a pool of AI agents")
        tasks, names, Y, P, R = as_matrices(agent_predictions())
        c1, c2, c3, c4 = st.columns(4)
        k = c1.slider("Agents in the pool", 3, 20, 8)
        n = c2.slider("Tasks in the batch", 50, 500, 200, step=50)
        slack = c3.slider("Total capacity ÷ tasks", 1.0, 3.0, 1.0, step=0.25)
        seed = c4.number_input("Random seed", 0, 9999, 1)
        lam = st.slider("Cost weight λ (what-if only: cost is ASSUMED equal to the agent's overall rating, "
                        "because no real price data exists)", 0.0, 1.0, 0.0, step=0.05)
        rng = np.random.default_rng(int(seed))
        a, t = rng.choice(len(names), k, replace=False), rng.choice(len(tasks), n, replace=False)
        y, p, r = Y[np.ix_(t, a)], P[np.ix_(t, a)], R[np.ix_(t, a)]
        cap, cost = int(np.ceil(slack * n / k)), r.mean(0)
        picks = {
            "Random": assign_random(n, k, cap, rng),
            "Best track record first": assign_by_track_record(cost, n, cap, rng),
            "Model + optimal assignment": solve_assignment(p, cap, cost=cost, lam=lam),
            "Oracle (upper bound)": solve_assignment(y.astype(float), cap),
        }
        rows = np.arange(n)
        summary = pd.DataFrame({"tasks resolved": {m: y[rows, pk].mean() for m, pk in picks.items()},
                                "assumed cost per task": {m: cost[pk].mean() for m, pk in picks.items()}})
        st.caption(f"Each agent may take at most {cap} tasks. Outcomes are the agents' real recorded results.")
        left, right = st.columns([1, 1])
        left.bar_chart(summary["tasks resolved"], horizontal=True)
        right.dataframe(summary.style.format({"tasks resolved": "{:.1%}", "assumed cost per task": "{:.3f}"}))
        pk = picks["Model + optimal assignment"]
        plan = pd.DataFrame({"task": tasks[t], "assigned agent": names[a][pk],
                             "predicted success": p[rows, pk].round(3), "actually resolved": y[rows, pk]})
        load = plan.groupby("assigned agent").agg(tasks=("task", "size"), resolved=("actually resolved", "mean"))
        l2, r2 = st.columns([2, 1])
        l2.markdown("**Assignment plan (model + optimal)**")
        l2.dataframe(plan, hide_index=True, width="stretch", height=300)
        r2.markdown("**Load per agent**")
        r2.dataframe(load.round(3), height=300)
        st.subheader("Full experiment (300 random pools per row, all 500 tasks)")
        st.dataframe(csv("swebench_assignment.csv").round(3), hide_index=True, width="stretch")
