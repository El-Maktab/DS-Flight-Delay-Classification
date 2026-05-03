"""
Author: Amir Anwar
Date: 2026-05-02

Description:
    Focused hyperparameter tuning for selected model families
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from statistics import mean, pstdev
from typing import Any

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler
import typer

from flight_delay_classification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from flight_delay_classification.evaluation.evaluate import (
    compute_cost_metrics,
    evaluate_predictions,
)
from flight_delay_classification.features import (
    DEFAULT_FEATURE_SELECTION_METHOD,
    DEFAULT_MIN_MUTUAL_INFO,
    TARGET_COLUMN,
    adapt_features_for_model_mode,
    apply_smote,
    build_feature_matrices,
    split_dataset,
)
from flight_delay_classification.modeling.registry import (
    build_hierarchical_hist_gradient_boosting_estimator,
    build_hist_gradient_boosting_estimator,
    build_random_forest_estimator,
)
from flight_delay_classification.modeling.train import (
    RANDOM_STATE,
    build_cv_splitter,
    configure_mlflow,
    suppress_mlflow_model_warnings,
)

app = typer.Typer()

DEFAULT_EXPERIMENT_NAME = "flight-delay-hyperparameter-tuning"
DEFAULT_TUNING_TEST_SIZE = 0.2
SUPPORTED_TUNING_MODES = (
    "hist_gradient_boosting",
    "random_forest",
    "hierarchical_hist_gradient_boosting",
)
SUPPORTED_PRIMARY_METRICS = ("balanced_accuracy", "cost")
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
        "class_weight": [None, "balanced"],
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


def compute_average_misclassification_cost(
    y_true: pd.Series | list[str] | Any,
    y_pred: pd.Series | list[str] | Any,
) -> float:
    return compute_cost_metrics(pd.Series(y_true), pd.Series(y_pred))[
        "average_misclassification_cost"
    ]


def compute_balanced_accuracy(
    y_true: pd.Series | list[str] | Any,
    y_pred: pd.Series | list[str] | Any,
) -> float:
    return float(balanced_accuracy_score(pd.Series(y_true), pd.Series(y_pred)))


def resolve_search_scoring(
    primary_metric: str,
) -> tuple[str | Any, str, float]:
    if primary_metric == "balanced_accuracy":
        return compute_balanced_accuracy, "best_cv_balanced_accuracy", 1.0

    if primary_metric == "cost":
        return (
            compute_average_misclassification_cost,
            "best_cv_average_misclassification_cost",
            -1.0,
        )

    raise typer.BadParameter(
        f"Unsupported primary_metric '{primary_metric}'. Use one of: {SUPPORTED_PRIMARY_METRICS}"
    )


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


def load_honest_tuning_frames(
    input_path: Path,
    test_size: float,
    random_state: int,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_df is not None and test_df is not None:
        return (
            train_df.reset_index(drop=True).copy(),
            test_df.reset_index(drop=True).copy(),
        )

    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded cleaned dataset from {} with shape {}", input_path, df.shape)
    return split_dataset(
        df=df,
        target_column=TARGET_COLUMN,
        test_size=test_size,
        random_state=random_state,
    )


def build_modeling_matrices_from_raw_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_selection_method: str,
    min_mutual_info: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train_df, X_test, y_test_df = build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        target_column=TARGET_COLUMN,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )
    return (
        X_train,
        y_train_df[TARGET_COLUMN],
        X_test,
        y_test_df[TARGET_COLUMN],
    )


def evaluate_parameter_candidate_with_honest_cv(
    *,
    model_mode: str = "",
    estimator: BaseEstimator,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    use_smote: bool,
    feature_selection_method: str,
    min_mutual_info: float,
    scoring: Any,
    score_multiplier: float,
    random_state: int,
) -> dict[str, Any]:
    cv_splitter = build_cv_splitter(train_df[TARGET_COLUMN], train_df)
    fold_scores: list[float] = []

    for fold_index, (fold_train_indices, fold_valid_indices) in enumerate(
        cv_splitter.split(train_df, train_df[TARGET_COLUMN]),
        start=1,
    ):
        fold_train_df = train_df.iloc[fold_train_indices].reset_index(drop=True)
        fold_valid_df = train_df.iloc[fold_valid_indices].reset_index(drop=True)
        X_fold_train, y_fold_train, X_fold_valid, y_fold_valid = (
            build_modeling_matrices_from_raw_frames(
                fold_train_df,
                fold_valid_df,
                feature_selection_method,
                min_mutual_info,
            )
        )
        X_fold_train, X_fold_valid, _ = adapt_features_for_model_mode(
            train_features=X_fold_train,
            test_features=X_fold_valid,
            model_mode=model_mode,
        )

        if use_smote:
            X_fold_train, y_fold_train = apply_smote(
                X_fold_train,
                y_fold_train,
                random_state,
            )

        candidate = clone(estimator)
        candidate.set_params(**params)
        candidate.fit(X_fold_train, y_fold_train)
        fold_predictions = pd.Series(
            candidate.predict(X_fold_valid), name=TARGET_COLUMN
        )
        fold_score = float(scoring(y_fold_valid, fold_predictions))
        logger.debug(
            "Scored params {} on fold {} with {}={:.6f}",
            params,
            fold_index,
            scoring.__name__,
            fold_score,
        )
        fold_scores.append(fold_score)

    return {
        "params": params,
        "mean_score": float(mean(fold_scores)),
        "std_score": float(pstdev(fold_scores)),
        "selection_score": float(mean(fold_scores) * score_multiplier),
    }


def score_parameter_candidates_with_honest_cv(
    *,
    model_mode: str = "",
    estimator: BaseEstimator,
    parameter_candidates: list[dict[str, Any]],
    train_df: pd.DataFrame,
    primary_metric: str,
    use_smote: bool,
    feature_selection_method: str,
    min_mutual_info: float,
    random_state: int,
) -> tuple[list[dict[str, Any]], int]:
    scoring, _, score_multiplier = resolve_search_scoring(primary_metric)
    cv_splitter = build_cv_splitter(train_df[TARGET_COLUMN], train_df)
    scored_trials = [
        evaluate_parameter_candidate_with_honest_cv(
            model_mode=model_mode,
            estimator=estimator,
            params=params,
            train_df=train_df,
            use_smote=use_smote,
            feature_selection_method=feature_selection_method,
            min_mutual_info=min_mutual_info,
            scoring=scoring,
            score_multiplier=score_multiplier,
            random_state=random_state,
        )
        for params in parameter_candidates
    ]
    return scored_trials, cv_splitter.get_n_splits(train_df, train_df[TARGET_COLUMN])


def build_top_trials(
    scored_trials: list[dict[str, Any]],
    primary_metric: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        scored_trials,
        key=lambda trial: trial["selection_score"],
        reverse=True,
    )[:limit]
    score_label = (
        "mean_cv_average_misclassification_cost"
        if primary_metric == "cost"
        else "mean_cv_balanced_accuracy"
    )
    std_label = (
        "std_cv_average_misclassification_cost"
        if primary_metric == "cost"
        else "std_cv_balanced_accuracy"
    )
    trials: list[dict[str, Any]] = []
    for rank, trial in enumerate(ranked, start=1):
        trials.append(
            {
                "rank": rank,
                score_label: float(trial["mean_score"]),
                std_label: float(trial["std_score"]),
                "params": trial["params"],
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
    input_path: Path,
    test_size: float,
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
    primary_metric: str,
    feature_selection_method: str,
    min_mutual_info: float,
    random_state: int,
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    X_train: pd.DataFrame | None,
    y_train: pd.Series | None,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
) -> dict[str, Any]:
    _ = (
        features_path,
        labels_path,
        test_features_path,
        test_labels_path,
        search_verbose,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    raw_train_df, raw_test_df = load_honest_tuning_frames(
        input_path=input_path,
        test_size=test_size,
        random_state=random_state,
        train_df=train_df,
        test_df=test_df,
    )

    _, best_score_key, _ = resolve_search_scoring(primary_metric)
    parameter_candidates = list(
        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state,
        )
    )
    scored_trials, cv_folds = score_parameter_candidates_with_honest_cv(
        model_mode=model_mode,
        estimator=estimator,
        parameter_candidates=parameter_candidates,
        train_df=raw_train_df,
        primary_metric=primary_metric,
        use_smote=use_smote,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
    )
    total_fits = len(parameter_candidates) * cv_folds
    logger.info(
        "Starting honest {} parameter search with {} candidates across {} fold-local CV fits",
        model_mode,
        len(parameter_candidates),
        total_fits,
    )

    best_trial = max(
        scored_trials,
        key=lambda trial: trial["selection_score"],
    )
    best_cv_score = float(best_trial["mean_score"])
    best_params = best_trial["params"]

    X_train, y_train, X_test, y_test = build_modeling_matrices_from_raw_frames(
        raw_train_df,
        raw_test_df,
        feature_selection_method,
        min_mutual_info,
    )
    X_train, X_test, _ = adapt_features_for_model_mode(
        train_features=X_train,
        test_features=X_test,
        model_mode=model_mode,
    )
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    best_model = clone(estimator)
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = pd.Series(best_model.predict(X_test), name=TARGET_COLUMN)
    evaluation_report = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        predictions_path=predictions_path,
    )
    top_trials = build_top_trials(scored_trials, primary_metric=primary_metric)
    tuning_report = {
        "search_strategy": "parameter_sampler",
        "tuning_strategy": "honest_fold_rebuild",
        "primary_metric": primary_metric,
        "model_mode": model_mode,
        "algorithm": algorithm,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "feature_selection_method": feature_selection_method,
        "selected_feature_columns": len(X_train.columns),
        "cv_folds": cv_folds,
        "top_trials": top_trials,
        "holdout_report": evaluation_report,
    }
    tuning_report[best_score_key] = best_cv_score
    logger.info(
        "{} randomized search finished. Best CV {}: {:.4f}",
        model_mode,
        primary_metric,
        best_cv_score,
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
                "search_strategy": "parameter_sampler",
                "tuning_strategy": "honest_fold_rebuild",
                "primary_metric": primary_metric,
                "use_smote": use_smote,
                "n_iter": n_iter,
                "feature_selection_method": feature_selection_method,
                "min_mutual_info": min_mutual_info,
                "train_rows": len(raw_train_df),
                "holdout_rows": len(raw_test_df),
                "feature_columns": len(X_train.columns),
                "selected_feature_columns": len(X_train.columns),
                "cv_folds": cv_folds,
                **{f"best_{key}": value for key, value in best_params.items()},
            }
        )
        mlflow.log_metrics(evaluation_report["core_metrics"])
        mlflow.log_metrics(evaluation_report["cost_metrics"])
        mlflow.log_metric("best_cv_score", best_cv_score)
        mlflow.log_metric(best_score_key, best_cv_score)
        with suppress_mlflow_model_warnings():
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
    logger.info("Best params: {}", best_params)
    logger.info("Holdout metrics: {}", evaluation_report["core_metrics"])
    return {
        "model_mode": model_mode,
        "model_path": str(model_path),
        "run_id": run.info.run_id,
        "tracking_uri": resolved_uri,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "metrics": evaluation_report["core_metrics"],
        "cost_metrics": evaluation_report["cost_metrics"],
    }


def tune_and_log_hist_gradient_boosting(
    input_path: Path,
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
    primary_metric: str = "balanced_accuracy",
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
    test_size: float = DEFAULT_TUNING_TEST_SIZE,
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
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
        input_path=input_path,
        test_size=test_size,
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
        primary_metric=primary_metric,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def tune_and_log_random_forest(
    input_path: Path,
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
    primary_metric: str = "balanced_accuracy",
    rf_class_weight: str | None = "balanced_subsample",
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
    test_size: float = DEFAULT_TUNING_TEST_SIZE,
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
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
        input_path=input_path,
        test_size=test_size,
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
        primary_metric=primary_metric,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def tune_and_log_hierarchical_hist_gradient_boosting(
    input_path: Path,
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
    primary_metric: str = "balanced_accuracy",
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
    test_size: float = DEFAULT_TUNING_TEST_SIZE,
    random_state: int = RANDOM_STATE,
    param_distributions: dict[str, list[Any]] | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
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
        input_path=input_path,
        test_size=test_size,
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
        primary_metric=primary_metric,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
        random_state=random_state,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "flights_cleaned.csv",
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
    primary_metric: str = "balanced_accuracy",
    rf_class_weight: str | None = "balanced_subsample",
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
    test_size: float = DEFAULT_TUNING_TEST_SIZE,
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
            input_path=input_path,
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
            primary_metric=primary_metric,
            feature_selection_method=feature_selection_method,
            min_mutual_info=min_mutual_info,
            test_size=test_size,
            random_state=random_state,
        )
    elif model_mode == "random_forest":
        summary = tune_and_log_random_forest(
            input_path=input_path,
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
            primary_metric=primary_metric,
            rf_class_weight=rf_class_weight,
            feature_selection_method=feature_selection_method,
            min_mutual_info=min_mutual_info,
            test_size=test_size,
            random_state=random_state,
        )
    elif model_mode == "hierarchical_hist_gradient_boosting":
        summary = tune_and_log_hierarchical_hist_gradient_boosting(
            input_path=input_path,
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
            primary_metric=primary_metric,
            feature_selection_method=feature_selection_method,
            min_mutual_info=min_mutual_info,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        raise typer.BadParameter(
            f"Unsupported model_mode '{model_mode}'. Use one of: {SUPPORTED_TUNING_MODES}"
        )

    logger.success("Tuning complete : {}", summary)


if __name__ == "__main__":
    app()
