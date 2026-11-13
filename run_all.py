"""Rebuild every result from the raw data:  python run_all.py"""
from src import run_assignment, train_lade, train_swebench

for name, step in [("1/3 Human workers: late-pickup model (LaDe-P)", train_lade.main),
                   ("2/3 AI agents: task-success model (SWE-bench Verified)", train_swebench.main),
                   ("3/3 Capacity-limited assignment experiment", run_assignment.main)]:
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    step()
