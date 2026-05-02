"""
Author: Amir Anwar
Date: 2026-05-01

Description:
    Model registry for training mode selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

TARGET_COLUMN = "DELAY_CATEGORY"


@dataclass(frozen=True)
class ModelTrainingRequest:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    max_iter: int
    rf_n_estimators: int
    rf_class_weight: str | None
    rf_min_samples_leaf: int
    rf_max_depth: int | None
    random_state: int


@dataclass(frozen=True)
class ModelTrainingResult:
    model: BaseEstimator | None
    y_pred: pd.Series
    class_weight: str | None
    model_family: str
    algorithm: str
    preprocessing: str


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


def train_hist_gradient_boosting_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
    class_weight: str | dict[str, float] | None,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_extra_trees_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    class_weight: str | None,
    min_samples_leaf: int,
    max_depth: int | None,
) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
    class_weight: str | None,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    sample_weight = None
    if class_weight is not None:
        sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def build_logreg_balanced(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_logistic_model(
        request.X_train,
        request.y_train,
        request.max_iter,
        request.random_state,
        class_weight="balanced",
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight="balanced",
        model_family="logistic_regression",
        algorithm="LogisticRegression",
        preprocessing="StandardScaler",
    )


def build_logreg_unbalanced(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_logistic_model(
        request.X_train,
        request.y_train,
        request.max_iter,
        request.random_state,
        class_weight=None,
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight=None,
        model_family="logistic_regression",
        algorithm="LogisticRegression",
        preprocessing="StandardScaler",
    )


def build_random_forest(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_random_forest_model(
        request.X_train,
        request.y_train,
        random_state=request.random_state,
        n_estimators=request.rf_n_estimators,
        class_weight=request.rf_class_weight,
        min_samples_leaf=request.rf_min_samples_leaf,
        max_depth=request.rf_max_depth,
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight=request.rf_class_weight,
        model_family="random_forest",
        algorithm="RandomForestClassifier",
        preprocessing="none",
    )


# NOTE: HistGradientBoosting
# NOTE: boosting can learn interactions without adding sparse encoding complexity
def build_hist_gradient_boosting(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_hist_gradient_boosting_model(
        request.X_train,
        request.y_train,
        max_iter=request.max_iter,
        random_state=request.random_state,
        class_weight="balanced",
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight="balanced",
        model_family="hist_gradient_boosting",
        algorithm="HistGradientBoostingClassifier",
        preprocessing="none",
    )


# NOTE: ExtraTrees is a cheap high-variance tree ensemble
def build_extra_trees(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_extra_trees_model(
        request.X_train,
        request.y_train,
        random_state=request.random_state,
        n_estimators=request.rf_n_estimators,
        class_weight=request.rf_class_weight,
        min_samples_leaf=request.rf_min_samples_leaf,
        max_depth=request.rf_max_depth,
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight=request.rf_class_weight,
        model_family="extra_trees",
        algorithm="ExtraTreesClassifier",
        preprocessing="none",
    )


# NOTE: Multi-layer Perceptron (Deep Learning model )
def build_mlp_balanced(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_mlp_model(
        request.X_train,
        request.y_train,
        max_iter=request.max_iter,
        random_state=request.random_state,
        class_weight="balanced",
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight="balanced",
        model_family="mlp",
        algorithm="MLPClassifier",
        preprocessing="StandardScaler",
    )


def build_majority_baseline(request: ModelTrainingRequest) -> ModelTrainingResult:
    majority_class = request.y_train.value_counts().idxmax()
    return ModelTrainingResult(
        model=None,
        y_pred=pd.Series([majority_class] * len(request.X_test), name=TARGET_COLUMN),
        class_weight=None,
        model_family="majority_baseline",
        algorithm="majority_baseline",
        preprocessing="none",
    )


ModelTrainer = Callable[[ModelTrainingRequest], ModelTrainingResult]

# NOTE: add a new model by implementing one builder and registering it here.
MODEL_BUILDERS: dict[str, ModelTrainer] = {
    "logreg_balanced": build_logreg_balanced,
    "logreg_unbalanced": build_logreg_unbalanced,
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "extra_trees": build_extra_trees,
    "majority_baseline": build_majority_baseline,
    "random_forest": build_random_forest,
    "mlp_balanced": build_mlp_balanced,
}
MODEL_MODES = tuple(MODEL_BUILDERS)


def train_model_for_mode(
    model_mode: str,
    request: ModelTrainingRequest,
) -> ModelTrainingResult:
    trainer = MODEL_BUILDERS.get(model_mode)
    if trainer is None:
        raise ValueError(
            f"Invalid model_mode '{model_mode}'. Use one of: {MODEL_MODES}"
        )
    return trainer(request)
