# reliability-dispatch-engine
Reliability-aware job dispatch across human, robot, and AI-agent workers — ML-based worker reliability scoring + optimization-based assignment.

AI-Driven Workforce Dispatch & Reliability Engine is a research project exploring how job dispatch should work when the available workforce includes humans, robots, and AI agents together. It combines a gradient-boosted reliability prediction model (XGBoost) with an optimization-based dispatch layer, and includes a live LLM-based triage agent. Validated with a documented proof-of-concept (0.801 accuracy, 0.838 ROC-AUC vs. a 0.614 majority-class baseline; 32.2% improvement in expected job success via model-guided assignment over random assignment).
