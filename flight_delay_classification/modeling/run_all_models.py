"""
Author: Amir Anwar
Date: 2026-05-01

Description:
    Runs all models
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from flight_delay_classification.features import apply_smote
from flight_delay_classification.modeling.registry import MODEL_MODES
from flight_delay_classification.modeling.train import (
    DEFAULT_EXPERIMENT_NAME,
    RANDOM_STATE,
    load_modeling_artifacts,
    train_and_log_model,
)

app = typer.Typer()


def build_model_descriptor(model_mode: str, use_smote: bool) -> str:
    descriptor = model_mode
    if use_smote:
        descriptor = f"{descriptor}-smote"
    return descriptor


def build_run_name(run_prefix: str, model_mode: str, use_smote: bool) -> str:
    return f"{run_prefix}-{build_model_descriptor(model_mode, use_smote)}"


def build_artifact_stem(run_prefix: str, model_mode: str, use_smote: bool) -> str:
    # NOTE: keep file names simple
    return re.sub(
        r"[^a-zA-Z0-9_-]+", "_", build_run_name(run_prefix, model_mode, use_smote)
    )


def run_all_models(
    experiment_name: str,
    run_prefix: str,
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    tracking_uri: str | None = None,
    use_smote: bool = False,
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    model_output_dir = MODELS_DIR / "batch_runs"
    evaluation_output_dir = REPORTS_DIR / "evaluation" / "batch_runs"
    X_train, y_train, X_test, y_test = load_modeling_artifacts(
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
    )

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    for model_mode in MODEL_MODES:
        artifact_stem = build_artifact_stem(run_prefix, model_mode, use_smote)
        run_name = build_run_name(run_prefix, model_mode, use_smote)
        summary = train_and_log_model(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
            model_path=model_output_dir / f"{artifact_stem}.pkl",
            evaluation_report_path=evaluation_output_dir / f"{artifact_stem}.json",
            predictions_path=PROCESSED_DATA_DIR / f"preds_{artifact_stem}.csv",
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
            random_state=random_state,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        summaries.append(
            {
                **summary,
                "model_mode": model_mode,
                "run_name": run_name,
            }
        )
        logger.info("Completed {} as {}", model_mode, run_name)

    return summaries


@app.command()
def main(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_prefix: str = "batch",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    tracking_uri: str | None = None,
    use_smote: bool = False,
    max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_class_weight: str | None = "balanced_subsample",
    rf_min_samples_leaf: int = 5,
    rf_max_depth: int | None = 25,
    random_state: int = RANDOM_STATE,
) -> None:
    summaries = run_all_models(
        experiment_name=experiment_name,
        run_prefix=run_prefix,
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        tracking_uri=tracking_uri,
        use_smote=use_smote,
        max_iter=max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_class_weight=rf_class_weight,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_depth=rf_max_depth,
        random_state=random_state,
    )

    logger.success("Completed {} model runs", len(summaries))


if __name__ == "__main__":
    app()
