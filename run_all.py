"""Rebuild every result from the raw data:  python run_all.py"""
from src import live_dispatch, run_assignment, train_lade, train_swebench

for name, step in [("1/4 Human workers: late-pickup model (LaDe-P)", train_lade.main),
                   ("2/4 AI agents: task-success model (SWE-bench Verified)", train_swebench.main),
                   ("3/4 Capacity-limited assignment experiment", run_assignment.main),
                   ("4/4 Live dispatch replays: 200 runs per load level", live_dispatch.main)]:
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    step()
