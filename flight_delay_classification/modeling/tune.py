"""
Author: Amir Anwar
Date: 2026-05-02

Description:
    Focused hyperparameter tuning for selected model families
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
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from flight_delay_classification.evaluation.evaluate import evaluate_predictions
from flight_delay_classification.features import TARGET_COLUMN, apply_smote
from flight_delay_classification.modeling.registry import (
    build_hist_gradient_boosting_estimator,
    build_hierarchical_hist_gradient_boosting_estimator,
    build_random_forest_estimator,
)
from flight_delay_classification.modeling.train import (
    RANDOM_STATE,
    build_cv_splitter,
    configure_mlflow,
    load_modeling_artifacts,
)

app = typer.Typer()

DEFAULT_EXPERIMENT_NAME = "flight-delay-hyperparameter-tuning"
SUPPORTED_TUNING_MODES = (
    "hist_gradient_boosting",
    "random_forest",
    "hierarchical_hist_gradient_boosting",
)
DEFAULT_TUNING_OUTPUTS: dict[str, dict[str, Path | str]] = {
    "hist_gradient_boosting": {
        "model_path": MODELS_DIR / "hist_gradient_boosting_tuned.pkl",
        "tuning_report_path": REPORTS_DIR
        / "evaluation"
        / "eval_hist_gradient_boosting_tuned.json",
        "predictions_path": PROCESSED_DATA_DIR
        / "preds_hist_gradient_boosting_tuned.csv",
        "run_name": "hist-gradient-boosting-random-search",
    },
    "random_forest": {
        "model_path": MODELS_DIR / "random_forest_tuned.pkl",
        "tuning_report_path": REPORTS_DIR
        / "evaluation"
        / "eval_random_forest_tuned.json",
        "predictions_path": PROCESSED_DATA_DIR / "preds_random_forest_tuned.csv",
        "run_name": "random-forest-random-search",
    },
    "hierarchical_hist_gradient_boosting": {
        "model_path": MODELS_DIR / "hierarchical_hist_gradient_boosting_tuned.pkl",
        "tuning_report_path": REPORTS_DIR
        / "evaluation"
        / "eval_hierarchical_hist_gradient_boosting_tuned.json",
        "predictions_path": PROCESSED_DATA_DIR
        / "preds_hierarchical_hist_gradient_boosting_tuned.csv",
        "run_name": "hierarchical-hist-gradient-boosting-random-search",
    },
}


def build_hist_gradient_boosting_search_space() -> dict[str, list[Any]]:
    # NOTE: keep the search focused on the parameters already called out in the plan.
    return {
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "max_iter": [300, 600, 1000, 1500, 2000],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 40, 80],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "max_depth": [None, 3, 5, 7, 10],
    }


def build_random_forest_search_space() -> dict[str, list[Any]]:
    # NOTE: keep the tree search narrow enough to compare cleanly against boosting.
    return {
        "n_estimators": [200, 300, 500, 700],
        "class_weight": ["balanced", "balanced_subsample"],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_depth": [10, 20, 30, None],
    }


def build_hierarchical_hist_gradient_boosting_search_space() -> dict[str, list[Any]]:
    # NOTE: the hierarchical model shares the same boosting controls as the single-stage version.
    return build_hist_gradient_boosting_search_space()


def resolve_tuning_defaults(
    model_mode: str,
    model_path: Path | None,
    tuning_report_path: Path | None,
    predictions_path: Path | None,
    run_name: str | None,
) -> tuple[Path, Path, Path, str]:
    if model_mode not in SUPPORTED_TUNING_MODES:
        raise typer.BadParameter(
            f"Unsupported model_mode '{model_mode}'. Use one of: {SUPPORTED_TUNING_MODES}"
        )

    defaults = DEFAULT_TUNING_OUTPUTS[model_mode]
    return (
        model_path or defaults["model_path"],
        tuning_report_path or defaults["tuning_report_path"],
        predictions_path or defaults["predictions_path"],
        run_name or str(defaults["run_name"]),
    )


def load_tuning_artifacts(
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

    return X_train, y_train, X_test, y_test


def build_top_trials(
    search: RandomizedSearchCV, limit: int = 5
) -> list[dict[str, Any]]:
    results = pd.DataFrame(search.cv_results_)
    ranked = results.sort_values("rank_test_score").head(limit)
    trials: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        trials.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_test_score": float(row["mean_test_score"]),
                "std_test_score": float(row["std_test_score"]),
                "params": {
                    key: value.item() if hasattr(value, "item") else value
                    for key, value in row["params"].items()
                },
            }
        )
    return trials


def tune_and_log_search(
    *,
    model_mode: str,
    model_family: str,
    algorithm: str,
    estimator: BaseEstimator,
    param_distributions: dict[str, list[Any]],
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    tuning_report_path: Path,
    predictions_path: Path,
    experiment_name: str,
    run_name: str | None,
    tracking_uri: str | None,
    use_smote: bool,
    n_iter: int,
    search_verbose: int,
    random_state: int,
    X_train: pd.DataFrame | None,
    y_train: pd.Series | None,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
) -> dict[str, Any]:
    X_train, y_train, X_test, y_test = load_tuning_artifacts(
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

    cv_splitter = build_cv_splitter(y_train, X_train)
    total_fits = n_iter * cv_splitter.get_n_splits(X_train, y_train)
    logger.info(
        "Starting {} randomized search with {} candidates across {} CV fits",
        model_mode,
        n_iter,
        total_fits,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=cv_splitter,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=search_verbose,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = pd.Series(best_model.predict(X_test), name=TARGET_COLUMN)
    evaluation_report = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        predictions_path=predictions_path,
    )
    top_trials = build_top_trials(search)
    tuning_report = {
        "search_strategy": "randomized_search",
        "primary_metric": "balanced_accuracy",
        "model_mode": model_mode,
        "algorithm": algorithm,
        "best_params": search.best_params_,
        "best_cv_balanced_accuracy": float(search.best_score_),
        "top_trials": top_trials,
        "holdout_report": evaluation_report,
    }
    logger.info(
        "{} randomized search finished. Best CV balanced accuracy: {:.4f}",
        model_mode,
        float(search.best_score_),
    )

    tuning_report_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_report_path.write_text(
        json.dumps(tuning_report, indent=2),
        encoding="utf-8",
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(best_model, f)

    resolved_uri = configure_mlflow(tracking_uri, experiment_name)
    sample = X_train.head(5)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "hyperparameter_tuning",
                "model_family": model_family,
            }
        )
        mlflow.log_params(
            {
                "model_mode": model_mode,
                "algorithm": algorithm,
                "search_strategy": "randomized_search",
                "primary_metric": "balanced_accuracy",
                "use_smote": use_smote,
                "n_iter": n_iter,
                "train_rows": len(X_train),
                "feature_columns": len(X_train.columns),
                **{f"best_{key}": value for key, value in search.best_params_.items()},
            }
        )
        mlflow.log_metrics(evaluation_report["core_metrics"])
        mlflow.log_metrics(evaluation_report["cost_metrics"])
        mlflow.log_metric("best_cv_balanced_accuracy", float(search.best_score_))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=infer_signature(sample, best_model.predict(sample)),
            input_example=sample,
        )
        mlflow.log_dict(tuning_report, "tuning/tuning_report.json")
        mlflow.log_artifact(str(predictions_path), "evaluation")

    logger.info("Saved tuned {} model to {}", model_mode, model_path)
    logger.info("MLflow run {}", run.info.run_id)
    logger.info("Best params: {}", search.best_params_)
    logger.info("Holdout metrics: {}", evaluation_report["core_metrics"])
    return {
        "model_mode": model_mode,
        "model_path": str(model_path),
        "run_id": run.info.run_id,
        "tracking_uri": resolved_uri,
        "best_params": search.best_params_,
        "best_cv_balanced_accuracy": float(search.best_score_),
        "metrics": evaluation_report["core_metrics"],
    }


def tune_and_log_hist_gradient_boosting(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    tuning_report_path: Path = REPORTS_DIR
    / "evaluation"
    / "eval_hist_gradient_boosting_tuned.json",
    predictions_path: Path = PROCESSED_DATA_DIR
    / "preds_hist_gradient_boosting_tuned.csv",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    use_smote: bool = False,
    n_iter: int = 30,
    search_verbose: int = 2,
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict[str, Any]:
    return tune_and_log_search(
        model_mode="hist_gradient_boosting",
        model_family="hist_gradient_boosting",
        algorithm="HistGradientBoostingClassifier",
        estimator=build_hist_gradient_boosting_estimator(
            class_weight="balanced",
            learning_rate=0.1,
            max_depth=None,
            max_iter=2000,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=random_state,
        ),
        param_distributions=param_distributions
        or build_hist_gradient_boosting_search_space(),
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        tuning_report_path=tuning_report_path,
        predictions_path=predictions_path,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        use_smote=use_smote,
        n_iter=n_iter,
        search_verbose=search_verbose,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def tune_and_log_random_forest(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    tuning_report_path: Path = REPORTS_DIR
    / "evaluation"
    / "eval_random_forest_tuned.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "preds_random_forest_tuned.csv",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    use_smote: bool = False,
    n_iter: int = 30,
    search_verbose: int = 2,
    rf_class_weight: str | None = "balanced_subsample",
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict[str, Any]:
    return tune_and_log_search(
        model_mode="random_forest",
        model_family="random_forest",
        algorithm="RandomForestClassifier",
        estimator=build_random_forest_estimator(
            random_state=random_state,
            n_estimators=300,
            class_weight=rf_class_weight,
            min_samples_leaf=5,
            max_depth=25,
        ),
        param_distributions=param_distributions or build_random_forest_search_space(),
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        tuning_report_path=tuning_report_path,
        predictions_path=predictions_path,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        use_smote=use_smote,
        n_iter=n_iter,
        search_verbose=search_verbose,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def tune_and_log_hierarchical_hist_gradient_boosting(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    model_path: Path,
    tuning_report_path: Path = REPORTS_DIR
    / "evaluation"
    / "eval_hierarchical_hist_gradient_boosting_tuned.json",
    predictions_path: Path = PROCESSED_DATA_DIR
    / "preds_hierarchical_hist_gradient_boosting_tuned.csv",
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    use_smote: bool = False,
    n_iter: int = 30,
    search_verbose: int = 2,
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> dict[str, Any]:
    return tune_and_log_search(
        model_mode="hierarchical_hist_gradient_boosting",
        model_family="hierarchical_hist_gradient_boosting",
        algorithm="HierarchicalDelayClassifier",
        estimator=build_hierarchical_hist_gradient_boosting_estimator(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_depth=None,
        ),
        param_distributions=param_distributions
        or build_hierarchical_hist_gradient_boosting_search_space(),
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        tuning_report_path=tuning_report_path,
        predictions_path=predictions_path,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        use_smote=use_smote,
        n_iter=n_iter,
        search_verbose=search_verbose,
        random_state=random_state,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_mode: str = "hist_gradient_boosting",
    model_path: Path | None = None,
    tuning_report_path: Path | None = None,
    predictions_path: Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    use_smote: bool = False,
    n_iter: int = 30,
    search_verbose: int = 2,
    rf_class_weight: str | None = "balanced_subsample",
    random_state: int = RANDOM_STATE,
) -> None:
    (
        resolved_model_path,
        resolved_tuning_report_path,
        resolved_predictions_path,
        resolved_run_name,
    ) = resolve_tuning_defaults(
        model_mode=model_mode,
        model_path=model_path,
        tuning_report_path=tuning_report_path,
        predictions_path=predictions_path,
        run_name=run_name,
    )

    if model_mode == "hist_gradient_boosting":
        summary = tune_and_log_hist_gradient_boosting(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
            model_path=resolved_model_path,
            tuning_report_path=resolved_tuning_report_path,
            predictions_path=resolved_predictions_path,
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            tracking_uri=tracking_uri,
            use_smote=use_smote,
            n_iter=n_iter,
            search_verbose=search_verbose,
            random_state=random_state,
        )
    elif model_mode == "random_forest":
        summary = tune_and_log_random_forest(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
            model_path=resolved_model_path,
            tuning_report_path=resolved_tuning_report_path,
            predictions_path=resolved_predictions_path,
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            tracking_uri=tracking_uri,
            use_smote=use_smote,
            n_iter=n_iter,
            search_verbose=search_verbose,
            rf_class_weight=rf_class_weight,
            random_state=random_state,
        )
    elif model_mode == "hierarchical_hist_gradient_boosting":
        summary = tune_and_log_hierarchical_hist_gradient_boosting(
            features_path=features_path,
            labels_path=labels_path,
            test_features_path=test_features_path,
            test_labels_path=test_labels_path,
            model_path=resolved_model_path,
            tuning_report_path=resolved_tuning_report_path,
            predictions_path=resolved_predictions_path,
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            tracking_uri=tracking_uri,
            use_smote=use_smote,
            n_iter=n_iter,
            search_verbose=search_verbose,
            random_state=random_state,
        )
    else:
        raise typer.BadParameter(
            f"Unsupported model_mode '{model_mode}'. Use one of: {SUPPORTED_TUNING_MODES}"
        )

    logger.success("Tuning complete : {}", summary)


if __name__ == "__main__":
    app()
