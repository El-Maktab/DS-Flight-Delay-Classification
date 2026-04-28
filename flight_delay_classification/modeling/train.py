"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic Model training
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from flight_delay_classification.evaluation.evaluate import evaluate_predictions

app = typer.Typer()

TARGET_COLUMN = "DELAY_CATEGORY"
CLASS_ORDER = ["on_time", "minor_delay", "major_delay", "cancelled"]
DEFAULT_EXPERIMENT_NAME = "flight-delay-baseline-better-eval"
MLFLOW_DB_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
RANDOM_STATE = 42
MODEL_MODES = (
    "logreg_balanced",
    "logreg_unbalanced",
    "majority_baseline",
    "random_forest",
)


def read_labels(labels_path: Path) -> pd.Series:
    return pd.read_csv(labels_path)[TARGET_COLUMN]


def train_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
    class_weight: str | None,
) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),  # NOTE: scaling is important for logistic regression
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,  # NOTE: "balanced" is important for class imbalance (it tells the model to focus on minority classes)
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    class_weight: str | None,
    min_samples_leaf: int,
    max_depth: int | None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,  # NOTE: can be 'balanced', 'balanced_subsample', 'balanced_subsample' is the recomended.
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def build_evaluation_outputs(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),  # NOTE: from docs "The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class."
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,  # NOTE: this prevents metrics from being NaN when there are no samples for a class in y_true or y_pred
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASS_ORDER),
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    )
    return metrics, report, confusion


def configure_mlflow(tracking_uri: str | None, experiment_name: str) -> str:
    resolved_uri = tracking_uri or MLFLOW_DB_URI
    mlflow.set_tracking_uri(resolved_uri)

    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        if resolved_uri.startswith("sqlite:///"):
            db_path = Path(resolved_uri.removeprefix("sqlite:///"))
            artifact_dir = db_path.parent / "mlartifacts" / experiment_name
        else:
            artifact_dir = PROJ_ROOT / "mlartifacts" / experiment_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.resolve().as_uri(),
        )

    mlflow.set_experiment(experiment_name)
    return resolved_uri


def train_and_log_model(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    evaluation_report_path: Path = REPORTS_DIR
    / "evaluation"
    / "evaluation_report.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    model_mode: str = "logreg_balanced",
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    if model_mode not in MODEL_MODES:
        raise ValueError(
            f"Invalid model_mode '{model_mode}'. Use one of: {MODEL_MODES}"
        )

    X_train = pd.read_csv(features_path)
    y_train = read_labels(labels_path)
    X_test = pd.read_csv(test_features_path)
    y_test = read_labels(test_labels_path)

    model: Any | None = None
    class_weight: str | None = None
    if model_mode == "logreg_balanced":
        class_weight = "balanced"
        model = train_logistic_model(
            X_train, y_train, max_iter, random_state, class_weight=class_weight
        )
        y_pred = pd.Series(model.predict(X_test), name=TARGET_COLUMN)
    elif model_mode == "logreg_unbalanced":
        model = train_logistic_model(
            X_train, y_train, max_iter, random_state, class_weight=None
        )
        y_pred = pd.Series(model.predict(X_test), name=TARGET_COLUMN)
    elif model_mode == "random_forest":
        model = train_random_forest_model(
            X_train,
            y_train,
            random_state=random_state,
            n_estimators=rf_n_estimators,
            class_weight=rf_class_weight,
            min_samples_leaf=rf_min_samples_leaf,
            max_depth=rf_max_depth,
        )
        y_pred = pd.Series(model.predict(X_test), name=TARGET_COLUMN)
    else:
        majority_class = y_train.value_counts().idxmax()
        y_pred = pd.Series([majority_class] * len(y_test), name=TARGET_COLUMN)

    evaluation_report = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        predictions_path=predictions_path,
    )
    metrics = evaluation_report["core_metrics"]
    cost_metrics = evaluation_report["cost_metrics"]

    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.write_text(
        json.dumps(evaluation_report, indent=2),
        encoding="utf-8",
    )

    resolved_model_path: str | None = None
    if model is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        resolved_model_path = str(model_path)

    resolved_uri = configure_mlflow(tracking_uri, experiment_name)
    sample = X_train.head(5)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "baseline_training",
                "model_family": (
                    "majority_baseline"
                    if model_mode == "majority_baseline"
                    else (
                        "random_forest"
                        if model_mode == "random_forest"
                        else "logistic_regression"
                    )
                ),
            }
        )
        mlflow.log_params(
            {
                "model_mode": model_mode,
                "algorithm": (
                    "majority_baseline"
                    if model_mode == "majority_baseline"
                    else (
                        "RandomForestClassifier"
                        if model_mode == "random_forest"
                        else "LogisticRegression"
                    )
                ),
                "preprocessing": (
                    "StandardScaler" if model_mode.startswith("logreg") else "none"
                ),
                "class_weight": class_weight,
                "max_iter": max_iter,
                "rf_n_estimators": rf_n_estimators,
                "rf_class_weight": rf_class_weight,
                "rf_min_samples_leaf": rf_min_samples_leaf,
                "rf_max_depth": rf_max_depth,
                "train_rows": len(X_train),
                "feature_columns": len(X_train.columns),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cost_metrics)
        if model is not None:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=infer_signature(sample, model.predict(sample)),
                input_example=sample,
            )
        mlflow.log_dict(evaluation_report, "evaluation/evaluation_report.json")
        mlflow.log_artifact(str(predictions_path), "evaluation")

    if resolved_model_path is not None:
        logger.info("Saved model to {}", resolved_model_path)
    logger.info("MLflow run {}", run.info.run_id)
    logger.info("Metrics: {}", metrics)
    return {
        "model_path": resolved_model_path,
        "run_id": run.info.run_id,
        "tracking_uri": resolved_uri,
        "metrics": metrics,
        "class_order": CLASS_ORDER,
    }


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    evaluation_report_path: Path = REPORTS_DIR
    / "evaluation"
    / "evaluation_report.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    model_mode: str = "logreg_balanced",
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    random_state: int = RANDOM_STATE,
) -> None:
    summary = train_and_log_model(
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        evaluation_report_path=evaluation_report_path,
        predictions_path=predictions_path,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        model_mode=model_mode,
        max_iter=max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_class_weight=rf_class_weight,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_depth=rf_max_depth,
        random_state=random_state,
    )

    logger.success("Training complete : {}", summary)


if __name__ == "__main__":
    app()
