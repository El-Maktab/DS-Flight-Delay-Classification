"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Better evaluation with business metrics
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from flight_delay_classification.features import TARGET_COLUMN

app = typer.Typer()

CLASS_ORDER = ["on_time", "minor_delay", "major_delay", "cancelled"]

# NOTE: those define the penalty cost for misclassifying each class.
# As we said in phase 1 report misclassifying a major delay or a cancellation
# is the major concern and more costly than on time or minor_delay
MISCLASS_COST = {
    "on_time": 0.5,
    "minor_delay": 5,
    "major_delay": 8,
    "cancelled": 10,
}


def read_labels(labels_path: Path) -> pd.Series:
    return pd.read_csv(labels_path)[TARGET_COLUMN]


def compute_core_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # NOTE: balanced accuracy is more informative for imbalanced datasets (like ours)
        # it is the average recall across classes
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # NOTE: gives each class equal importance
        # F1 computed per class then averaged equally across classes
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # NOTE: complements macro view by reflecting real class frequency mix
        # F1 per class weighted by class support
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_cost_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    miss_cost = [
        MISCLASS_COST[truth] for truth, pred in zip(y_true, y_pred) if truth != pred
    ]
    total_cost = float(sum(miss_cost))
    avg_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    baseline_pred = pd.Series(["on_time"] * len(y_true))
    baseline_miss_cost = [
        MISCLASS_COST[truth]
        for truth, pred in zip(y_true, baseline_pred)
        if truth != pred
    ]
    baseline_avg_cost = (
        float(sum(baseline_miss_cost)) / len(y_true) if len(y_true) > 0 else 0.0
    )

    return {
        # NOTE: caculates the cost based on the penalties we defined above
        # lower cost is better
        "average_misclassification_cost": avg_cost,
        # NOTE: this the average cost if considered that all flights are on time
        "baseline_on_time_average_cost": baseline_avg_cost,
        # NOTE: how much cost we save compared to the baseline of predicting all flights as on time
        # higher is better, positive means we are saving cost and this model is worth using :)
        "cost_reduction_vs_on_time_baseline": baseline_avg_cost - avg_cost,
    }


def evaluate_model(
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    report_path: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    X_test = pd.read_csv(test_features_path)
    y_true = read_labels(test_labels_path)

    with model_path.open("rb") as f:
        model = pickle.load(f)

    y_pred = pd.Series(model.predict(X_test), name="y_pred")

    report = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        predictions_path=predictions_path,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    predictions_path: Path,
) -> dict[str, Any]:
    core_metrics = compute_core_metrics(y_true, y_pred)
    cost_metrics = compute_cost_metrics(y_true, y_pred)
    per_class = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    eval_df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values})

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(predictions_path, index=False)

    report = {
        "core_metrics": core_metrics,
        "cost_metrics": cost_metrics,
        "per_class": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1_score": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"]),
            }
            for label in CLASS_ORDER
        },
        "artifacts": {
            "predictions_path": str(predictions_path),
        },
    }
    return report


@app.command()
def main(
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    report_path: Path = REPORTS_DIR / "evaluation" / "evaluation_report.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    summary = evaluate_model(
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        report_path=report_path,
        predictions_path=predictions_path,
    )

    logger.success("Evaluation complete.")
    logger.info("Core metrics: {}", summary["core_metrics"])
    logger.info("Cost metrics: {}", summary["cost_metrics"])
    logger.info("Report saved to {}", report_path)


if __name__ == "__main__":
    app()
