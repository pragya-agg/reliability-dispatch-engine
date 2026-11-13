# AI-Driven Workforce Dispatch & Reliability Engine

Predicts how reliably a worker will complete a job, then uses those predictions to assign jobs.
Tested on two public, real-world datasets: human couriers (LaDe-P) and AI agents (SWE-bench Verified).

## Status

| Module | State |
|---|---|
| Data pipelines with past-only (leakage-safe) worker history | Done |
| Reliability model for human couriers (late pickup) | Done |
| Reliability model for AI agents (task resolved or not) | Done |
| Capacity- and cost-aware optimal assignment + baselines | Done |
| Assignment experiment with confidence intervals | Done |
| Dashboard (4 tabs) | Done |
| Unit tests (assignment correctness, leakage check) | Done |
| Better task features, ablation study, final report | To do |
| Robot workers | Out of scope (no public outcome data found) |

## Results (held-out data)

**Human couriers, LaDe-P** — 6.14M pickups, 1.85% late, time-ordered split (test = last 25%).

| Model | ROC-AUC | PR-AUC | Late pickups caught in riskiest 10% |
|---|---|---|---|
| Courier's past late rate only | 0.842 | 0.157 | 51.5% |
| XGBoost, job + courier features | 0.863 | 0.155 | 57.0% |

**AI agents, SWE-bench Verified** — 135 agent submissions × 500 tasks, newest 25% of tasks held out.

| Model | ROC-AUC |
|---|---|
| Agent's overall rating only | 0.698 |
| XGBoost, agent history + task features | 0.754 |

**Assignment, AI agents** — 500 unseen tasks, random pools of agents, each agent limited to a number of
tasks. Share of tasks actually resolved (300 pools per row):

| Agents | Capacity ÷ tasks | Random | Best track record first | Model + optimal | Oracle | Gain over track record (95% CI) |
|---|---|---|---|---|---|---|
| 5 | 1.0 | 51.4% | 51.3% | 53.5% | 78.1% | +2.2 pts (1.2 to 3.1) |
| 5 | 2.0 | 50.4% | 63.9% | 64.6% | 78.8% | +0.8 pts (0.2 to 1.3) |
| 10 | 1.0 | 52.3% | 52.3% | 54.9% | 84.4% | +2.5 pts (1.2 to 3.7) |
| 10 | 2.0 | 51.9% | 66.6% | 67.6% | 84.7% | +1.0 pts (0.4 to 1.7) |

Full tables are in `results/`.

## Limitations

- For couriers, the model barely beats the courier's own past late rate.
- With no capacity limit, picking an agent by model is not significantly better than always using the
  best-rated agent (76.8% vs 76.0%, 95% CI −1.6 to +4.0 points).
- With capacity limits the gain is statistically significant but small (about 1 to 2.5 points), far below the oracle.
- Assignment is evaluated only on AI agents: only there is the outcome of every worker-job pair known.
- Humans and AI agents come from separate datasets with different jobs; routing one job between a human and
  an agent is not tested.
- The SWE-bench "chronological" split orders tasks by issue date, not by when agents ran.
- Agent submissions that skipped a task count as failures. Many submissions share the same underlying model.
- The cost slider in the dashboard uses an assumed cost (no real price data).
- Adding text-similarity features (TF-IDF nearest tasks) was tried and did not improve agent selection.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py          # rebuild every result (about 3 minutes)
.venv/bin/python -m pytest -q tests  # tests
.venv/bin/streamlit run app.py       # dashboard
```

## Data (not committed; place under `data/raw/`)

- `lade/pickup/*.parquet` — LaDe dataset, Hugging Face `Cainiao-AI/LaDe` (pickup files).
- `swebench_experiments/` — clone of https://github.com/SWE-bench/experiments (snapshot used: 3 Sep 2026, commit `40f164d`).
- `swebench_tasks/verified_test.parquet` — test split of the SWE-bench Verified dataset.

## Layout

```
src/lade_data.py        courier data, label, past-only history features
src/train_lade.py       late-pickup model and baselines
src/swebench_data.py    agent x task outcome table
src/train_swebench.py   task-success model, agent selection without capacity limits
src/assignment.py       optimal assignment under capacity and cost, plus baselines
src/run_assignment.py   capacity-limited assignment experiment
app.py                  dashboard
run_all.py              rebuilds every result
tests/                  unit tests
docs/                   synopsis
```
