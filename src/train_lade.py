"""Train and evaluate late-pickup prediction on LaDe-P (chronological split)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from src.lade_data import COURIER_FEATURES, JOB_FEATURES, load_features

TRAIN_FRAC, VAL_FRAC = 0.60, 0.15  # remaining 25% is the test set
OUT = "results"


def split_by_time(df: pd.DataFrame):
    df = df.sort_values("accept_time").reset_index(drop=True)
    a, b = int(len(df) * TRAIN_FRAC), int(len(df) * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def fit_xgb(train, val, features):
    model = XGBClassifier(
        n_estimators=500, learning_rate=0.08, max_depth=6, subsample=0.8,
        colsample_bytree=0.9, tree_method="hist", enable_categorical=True,
        eval_metric="aucpr", early_stopping_rounds=30, n_jobs=-1, random_state=0,
    )
    model.fit(train[features], train.late, eval_set=[(val[features], val.late)], verbose=False)
    return model


def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return t[np.argmax(f1)]


def evaluate(name, y_test, score_test, threshold):
    pred = score_test >= threshold
    tp = int((pred & (y_test == 1)).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / int((y_test == 1).sum())
    top = score_test >= np.quantile(score_test, 0.9)
    return {
        "model": name,
        "pr_auc": average_precision_score(y_test, score_test),
        "roc_auc": roc_auc_score(y_test, score_test),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "recall_in_top10pct": float((top & (y_test == 1)).sum() / (y_test == 1).sum()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    train, val, test = split_by_time(load_features())
    y_val, y_test = val.late.to_numpy(), test.late.to_numpy()
    print(f"rows train/val/test: {len(train):,}/{len(val):,}/{len(test):,}")
    print(f"late rate train {train.late.mean():.4f} val {val.late.mean():.4f} test {test.late.mean():.4f}")
    print(f"test jobs by couriers with <20 past jobs: {(test.past_n < 20).mean():.3f}; "
          f"couriers never seen in train: {(~test.courier_id.isin(train.courier_id)).mean():.4f}")

    prior = train.late.mean()
    rng = np.random.default_rng(0)
    scores = {}  # name -> (val score, test score)
    scores["Random"] = (rng.random(len(val)), rng.random(len(test)))
    # Smoothed courier history only (prior weight = 20 jobs).
    hist = lambda d: (d.past_late_rate.fillna(prior) * d.past_n + prior * 20) / (d.past_n + 20)
    scores["Courier past late rate only"] = (hist(val).to_numpy(), hist(test).to_numpy())

    job_model = fit_xgb(train, val, JOB_FEATURES)
    scores["XGBoost, job features only"] = (
        job_model.predict_proba(val[JOB_FEATURES])[:, 1], job_model.predict_proba(test[JOB_FEATURES])[:, 1])

    all_features = JOB_FEATURES + COURIER_FEATURES
    full_model = fit_xgb(train, val, all_features)
    scores["XGBoost, job + courier features"] = (
        full_model.predict_proba(val[all_features])[:, 1], full_model.predict_proba(test[all_features])[:, 1])

    rows = [{"model": "Always on-time (never flag late)", "pr_auc": prior, "roc_auc": 0.5,
             "precision": 0.0, "recall": 0.0, "f1": 0.0, "recall_in_top10pct": 0.1}]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (s_val, s_test) in scores.items():
        rows.append(evaluate(name, y_test, s_test, best_f1_threshold(y_val, s_val)))
        p, r, _ = precision_recall_curve(y_test, s_test)
        ax.plot(r, p, label=name)
    ax.axhline(y_test.mean(), color="grey", ls="--", label="base rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="LaDe-P late pickup: test PR curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/lade_pr_curves.png", dpi=150)

    # Test-set risk scores, used by the dashboard.
    test.assign(p_late=scores["XGBoost, job + courier features"][1],
                p_history=scores["Courier past late rate only"][1])[
        ["order_id", "courier_id", "city", "accept_time", "late", "past_n", "open_jobs", "p_late", "p_history"]
    ].to_parquet("data/lade_test_predictions.parquet")

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/lade_results.csv", index=False)
    print(res.round(4).to_string(index=False))

    imp = pd.Series(full_model.feature_importances_, index=all_features).sort_values(ascending=False)
    print("\nXGBoost (job + courier) feature importance:\n", imp.round(3).to_string())
    print("best iteration job-only / full:", job_model.best_iteration, full_model.best_iteration)


if __name__ == "__main__":
    main()
