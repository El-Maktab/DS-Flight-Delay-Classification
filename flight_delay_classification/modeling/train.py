"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic Model training
"""

from __future__ import annotations

import logging
import json
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# NOTE: StratifiedKFold is a cross validation technique that preserves the class distribution
from sklearn.model_selection import StratifiedKFold, cross_validate
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from flight_delay_classification.evaluation.evaluate import evaluate_predictions
from flight_delay_classification.evaluation.evaluate import (
    compute_core_metrics,
    compute_cost_metrics,
)
from flight_delay_classification.features import (
    DEFAULT_FEATURE_SELECTION_METHOD,
    DEFAULT_MIN_MUTUAL_INFO,
    adapt_features_for_model_mode,
    apply_smote,
    select_informative_features,
)
from flight_delay_classification.modeling.registry import (
    ModelTrainingRequest,
    train_model_for_mode,
)

app = typer.Typer()

TARGET_COLUMN = "DELAY_CATEGORY"
CLASS_ORDER = ["on_time", "minor_delay", "major_delay", "cancelled"]
DEFAULT_EXPERIMENT_NAME = "flight-delay-baseline-better-eval"
MLFLOW_DB_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
RANDOM_STATE = 42
MLFLOW_INTEGER_SCHEMA_WARNING = r"Hint: Inferred schema contains integer column\(s\)\."


@dataclass(frozen=True)
class _PreparedTrainingData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    selected_columns: list[str]


@dataclass(frozen=True)
class _TrainingEvaluation:
    train_metrics: dict[str, float]
    train_cost_metrics: dict[str, float]
    cv_scores: dict[str, float]
    evaluation_report: dict[str, Any]
    metrics: dict[str, float]
    cost_metrics: dict[str, float]


def read_labels(labels_path: Path) -> pd.Series:
    return pd.read_csv(labels_path)[TARGET_COLUMN]


def load_modeling_artifacts(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    return (
        pd.read_csv(features_path),
        read_labels(labels_path),
        pd.read_csv(test_features_path),
        read_labels(test_labels_path),
    )


def build_cv_splitter(
    y_train: pd.Series,
    X_train: pd.DataFrame,
) -> StratifiedKFold:
    # NOTE: between 2 and 5
    n_splits = max(2, min(5, len(X_train) // max(y_train.nunique(), 2)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def evaluate_cv_scores(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    skf = build_cv_splitter(y_train, X_train)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    cv_results = cross_validate(
        model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False
    )
    return {
        "cv_accuracy": cv_results["test_accuracy"].mean(),
        "cv_balanced_accuracy": cv_results["test_balanced_accuracy"].mean(),
        "cv_f1_macro": cv_results["test_f1_macro"].mean(),
        "cv_f1_weighted": cv_results["test_f1_weighted"].mean(),
    }


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


@contextmanager
def suppress_mlflow_model_warnings() -> Iterator[None]:
    sklearn_logger = logging.getLogger("mlflow.sklearn")
    previous_level = sklearn_logger.level
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=MLFLOW_INTEGER_SCHEMA_WARNING,
            category=UserWarning,
            module=r"mlflow\.types\.utils",
        )
        sklearn_logger.setLevel(max(previous_level, logging.ERROR))
        try:
            yield
        finally:
            sklearn_logger.setLevel(previous_level)


def _resolve_modeling_data(
    *,
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    use_smote: bool,
    random_state: int,
    X_train: pd.DataFrame | None,
    y_train: pd.Series | None,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    using_preloaded_data = not (
        X_train is None or y_train is None or X_test is None or y_test is None
    )

    if not using_preloaded_data:
        X_train, y_train, X_test, y_test = load_modeling_artifacts(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
        )

    if use_smote and not using_preloaded_data:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    assert X_train is not None and y_train is not None
    assert X_test is not None and y_test is not None
    return X_train, y_train, X_test, y_test


def _prepare_training_data(
    *,
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_mode: str,
    use_smote: bool,
    feature_selection_method: str,
    min_mutual_info: float,
    random_state: int,
    X_train: pd.DataFrame | None,
    y_train: pd.Series | None,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
) -> _PreparedTrainingData:
    resolved_X_train, resolved_y_train, resolved_X_test, resolved_y_test = (
        _resolve_modeling_data(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
            use_smote=use_smote,
            random_state=random_state,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )

    selected_X_train, selected_X_test, selected_columns = select_informative_features(
        train_features=resolved_X_train,
        test_features=resolved_X_test,
        y_train=resolved_y_train,
        method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )
    adapted_X_train, adapted_X_test, dropped_columns = adapt_features_for_model_mode(
        train_features=selected_X_train,
        test_features=selected_X_test,
        model_mode=model_mode,
    )
    if dropped_columns:
        selected_columns = [
            column for column in selected_columns if column not in dropped_columns
        ]

    return _PreparedTrainingData(
        X_train=adapted_X_train,
        y_train=resolved_y_train,
        X_test=adapted_X_test,
        y_test=resolved_y_test,
        selected_columns=selected_columns,
    )


def _build_train_predictions(
    *,
    model: BaseEstimator | None,
    model_mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    if model is not None:
        return pd.Series(model.predict(X_train), name=TARGET_COLUMN)

    if model_mode == "majority_baseline":
        majority_class = y_train.value_counts().idxmax()
        return pd.Series([majority_class] * len(X_train), name=TARGET_COLUMN)

    raise ValueError(
        f"Model mode '{model_mode}' returned no fitted estimator, so train predictions cannot be computed"
    )


def _assemble_training_evaluation(
    *,
    model_mode: str,
    model: BaseEstimator | None,
    y_pred: pd.Series,
    prepared_data: _PreparedTrainingData,
    predictions_path: Path,
) -> _TrainingEvaluation:
    train_predictions = _build_train_predictions(
        model=model,
        model_mode=model_mode,
        X_train=prepared_data.X_train,
        y_train=prepared_data.y_train,
    )
    train_metrics = compute_core_metrics(prepared_data.y_train, train_predictions)
    train_cost_metrics = compute_cost_metrics(prepared_data.y_train, train_predictions)
    cv_scores = (
        evaluate_cv_scores(model, prepared_data.X_train, prepared_data.y_train)
        if model is not None
        else {}
    )

    evaluation_report = evaluate_predictions(
        y_true=prepared_data.y_test,
        y_pred=y_pred,
        predictions_path=predictions_path,
    )
    evaluation_report["train_core_metrics"] = train_metrics
    evaluation_report["train_cost_metrics"] = train_cost_metrics

    return _TrainingEvaluation(
        train_metrics=train_metrics,
        train_cost_metrics=train_cost_metrics,
        cv_scores=cv_scores,
        evaluation_report=evaluation_report,
        metrics=evaluation_report["core_metrics"],
        cost_metrics=evaluation_report["cost_metrics"],
    )


def _write_evaluation_report(
    evaluation_report: dict[str, Any],
    evaluation_report_path: Path,
) -> None:
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.write_text(
        json.dumps(evaluation_report, indent=2),
        encoding="utf-8",
    )


def _persist_model(model: BaseEstimator | None, model_path: Path) -> str | None:
    if model is None:
        return None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    return str(model_path)


def _log_training_run(
    *,
    experiment_name: str,
    run_name: str | None,
    tracking_uri: str | None,
    model_mode: str,
    use_smote: bool,
    model: BaseEstimator | None,
    class_weight: str | None,
    max_iter: int,
    rf_n_estimators: int,
    rf_class_weight: str | None,
    rf_min_samples_leaf: int,
    rf_max_depth: int | None,
    hgb_learning_rate: float,
    hgb_max_leaf_nodes: int,
    hgb_min_samples_leaf: int,
    hgb_l2_regularization: float,
    hgb_max_depth: int | None,
    feature_selection_method: str,
    min_mutual_info: float,
    prepared_data: _PreparedTrainingData,
    training_result: Any,
    evaluation: _TrainingEvaluation,
    predictions_path: Path,
) -> tuple[str, str]:
    resolved_uri = configure_mlflow(tracking_uri, experiment_name)
    sample = prepared_data.X_train.head(5)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "baseline_training",
                "model_family": training_result.model_family,
            }
        )
        mlflow.log_params(
            {
                "model_mode": model_mode,
                "algorithm": training_result.algorithm,
                "preprocessing": training_result.preprocessing,
                "use_smote": use_smote,
                "class_weight": class_weight,
                "max_iter": max_iter,
                "rf_n_estimators": rf_n_estimators,
                "rf_class_weight": rf_class_weight,
                "rf_min_samples_leaf": rf_min_samples_leaf,
                "rf_max_depth": rf_max_depth,
                "hgb_learning_rate": hgb_learning_rate,
                "hgb_max_leaf_nodes": hgb_max_leaf_nodes,
                "hgb_min_samples_leaf": hgb_min_samples_leaf,
                "hgb_l2_regularization": hgb_l2_regularization,
                "hgb_max_depth": hgb_max_depth,
                "feature_selection_method": feature_selection_method,
                "min_mutual_info": min_mutual_info,
                "train_rows": len(prepared_data.X_train),
                "feature_columns": len(prepared_data.X_train.columns),
                "selected_feature_columns": len(prepared_data.selected_columns),
            }
        )
        mlflow.log_metrics(
            {f"train_{key}": value for key, value in evaluation.train_metrics.items()}
        )
        mlflow.log_metrics(
            {
                f"train_{key}": value
                for key, value in evaluation.train_cost_metrics.items()
            }
        )
        mlflow.log_metrics(evaluation.metrics)
        mlflow.log_metrics(evaluation.cost_metrics)
        mlflow.log_metrics(evaluation.cv_scores)
        if model is not None:
            with suppress_mlflow_model_warnings():
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=infer_signature(sample, model.predict(sample)),
                    input_example=sample,
                )
        mlflow.log_dict(
            evaluation.evaluation_report, "evaluation/evaluation_report.json"
        )
        mlflow.log_artifact(str(predictions_path), "evaluation")

    return resolved_uri, run.info.run_id


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
    use_smote: bool = False,
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    hgb_learning_rate: float = 0.1,
    hgb_max_leaf_nodes: int = 31,
    hgb_min_samples_leaf: int = 20,
    hgb_l2_regularization: float = 0.0,
    hgb_max_depth: int | None = None,
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
    random_state: int = RANDOM_STATE,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict[str, Any]:
    prepared_data = _prepare_training_data(
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_mode=model_mode,
        use_smote=use_smote,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    training_result = train_model_for_mode(
        model_mode,
        ModelTrainingRequest(
            X_train=prepared_data.X_train,
            y_train=prepared_data.y_train,
            X_test=prepared_data.X_test,
            max_iter=max_iter,
            rf_n_estimators=rf_n_estimators,
            rf_class_weight=rf_class_weight,
            rf_min_samples_leaf=rf_min_samples_leaf,
            rf_max_depth=rf_max_depth,
            random_state=random_state,
            hgb_learning_rate=hgb_learning_rate,
            hgb_max_leaf_nodes=hgb_max_leaf_nodes,
            hgb_min_samples_leaf=hgb_min_samples_leaf,
            hgb_l2_regularization=hgb_l2_regularization,
            hgb_max_depth=hgb_max_depth,
        ),
    )
    model = training_result.model
    class_weight = training_result.class_weight
    evaluation = _assemble_training_evaluation(
        model_mode=model_mode,
        model=model,
        y_pred=training_result.y_pred,
        prepared_data=prepared_data,
        predictions_path=predictions_path,
    )
    _write_evaluation_report(evaluation.evaluation_report, evaluation_report_path)

    resolved_model_path = _persist_model(model, model_path)
    resolved_uri, run_id = _log_training_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        model_mode=model_mode,
        use_smote=use_smote,
        model=model,
        class_weight=class_weight,
        max_iter=max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_class_weight=rf_class_weight,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_depth=rf_max_depth,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
        hgb_min_samples_leaf=hgb_min_samples_leaf,
        hgb_l2_regularization=hgb_l2_regularization,
        hgb_max_depth=hgb_max_depth,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        prepared_data=prepared_data,
        training_result=training_result,
        evaluation=evaluation,
        predictions_path=predictions_path,
    )

    if resolved_model_path is not None:
        logger.info("Saved model to {}", resolved_model_path)
    logger.info("MLflow run {}", run_id)
    logger.info("Metrics: {}", evaluation.metrics)
    return {
        "model_path": resolved_model_path,
        "evaluation_report_path": str(evaluation_report_path),
        "run_id": run_id,
        "tracking_uri": resolved_uri,
        "train_metrics": evaluation.train_metrics,
        "train_cost_metrics": evaluation.train_cost_metrics,
        "metrics": evaluation.metrics,
        "cost_metrics": evaluation.cost_metrics,
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
    use_smote: bool = False,
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    hgb_learning_rate: float = 0.1,
    hgb_max_leaf_nodes: int = 31,
    hgb_min_samples_leaf: int = 20,
    hgb_l2_regularization: float = 0.0,
    hgb_max_depth: int | None = None,
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
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
        use_smote=use_smote,
        max_iter=max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_class_weight=rf_class_weight,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_depth=rf_max_depth,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
        hgb_min_samples_leaf=hgb_min_samples_leaf,
        hgb_l2_regularization=hgb_l2_regularization,
        hgb_max_depth=hgb_max_depth,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
    )

    logger.success("Training complete : {}", summary)


if __name__ == "__main__":
    app()
