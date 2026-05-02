"""
Author: Amir Anwar
Date: 2026-05-01

Description:
    Model registry for training mode selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

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
    hgb_learning_rate: float = 0.1
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 0.0
    hgb_max_depth: int | None = None


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
    model = build_random_forest_estimator(
        n_estimators=n_estimators,  # NOTE: number of trees in the forest
        random_state=random_state,
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return model


def build_random_forest_estimator(
    *,
    random_state: int,
    n_estimators: int,
    class_weight: str | None,
    min_samples_leaf: int,
    max_depth: int | None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,  # NOTE: number of trees in the forest
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,  # NOTE: stops leaves from becoming too small
        max_depth=max_depth,  # NOTE: limits how deep each tree can grow
        random_state=random_state,
        n_jobs=-1,  # NOTE: uses all CPU cores
    )


def build_hist_gradient_boosting_estimator(
    *,
    max_iter: int,
    random_state: int,
    class_weight: str | dict[str, float] | None,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_depth: int | None,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight=class_weight,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


def train_hist_gradient_boosting_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
    class_weight: str | dict[str, float] | None,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_depth: int | None,
) -> HistGradientBoostingClassifier:
    model = build_hist_gradient_boosting_estimator(
        class_weight=class_weight,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
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
        n_estimators=n_estimators,  # NOTE: number of trees
        class_weight=class_weight,
        min_samples_leaf=min_samples_leaf,  # NOTE: stops leaves from becoming too small
        max_depth=max_depth,  # NOTE: limits how deep each tree can grow
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
                    hidden_layer_sizes=(
                        128,
                        64,
                    ),  # NOTE: two hidden layers for a small neural net
                    max_iter=max_iter,  # NOTE: how many training passes to allow
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


class XGBoostDelayClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int,
        max_depth: int | None,
        random_state: int,
        class_weight: str | None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.label_encoder_: LabelEncoder | None = None
        self.model_: XGBClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostDelayClassifier:
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        sample_weight = None
        if self.class_weight is not None:
            sample_weight = compute_sample_weight(
                class_weight=self.class_weight,
                y=y,
            )

        self.model_ = XGBClassifier(
            objective="multi:softprob",  # NOTE: predicts class probabilities for all delay classes
            num_class=len(
                self.label_encoder_.classes_
            ),  # NOTE: tells XGBoost how many labels exist
            n_estimators=self.n_estimators,  # NOTE: number of boosting trees
            max_depth=self.max_depth or 6,  # NOTE: limits how deep each tree can grow
            learning_rate=0.05,  # NOTE: smaller steps usually make boosting steadier
            subsample=0.9,  # NOTE: trains each round on most row
            colsample_bytree=0.9,  # NOTE: trains each tree on most, not all
            tree_method="hist",  # NOTE: uses the fast histogram tree builder
            eval_metric="mlogloss",  # NOTE: measures multiclass probability error while training
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y_encoded, sample_weight=sample_weight, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred.astype(int))


class HierarchicalDelayClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iter: int,
        random_state: int,
        class_weight: str | dict[str, float] | None,
        learning_rate: float,
        max_leaf_nodes: int,
        min_samples_leaf: int,
        l2_regularization: float,
        max_depth: int | None,
    ) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_depth = max_depth
        self.stage_one_model_: HistGradientBoostingClassifier | None = None
        self.stage_two_model_: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HierarchicalDelayClassifier:
        self.stage_one_model_ = build_hist_gradient_boosting_estimator(
            class_weight=self.class_weight,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.stage_two_model_ = build_hist_gradient_boosting_estimator(
            class_weight=self.class_weight,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

        delayed_mask = y != "on_time"
        self.stage_one_model_.fit(X, delayed_mask)
        self.stage_two_model_.fit(X.loc[delayed_mask], y.loc[delayed_mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        delayed_pred = self.stage_one_model_.predict(X).astype(bool)
        predictions = np.full(len(X), "on_time", dtype=object)
        if delayed_pred.any():
            predictions[delayed_pred] = self.stage_two_model_.predict(
                X.loc[delayed_pred]
            )
        return predictions


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    class_weight: str | None,
    max_depth: int | None,
) -> XGBoostDelayClassifier:
    model = XGBoostDelayClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def train_hierarchical_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
    class_weight: str | dict[str, float] | None,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_depth: int | None,
) -> HierarchicalDelayClassifier:
    model = build_hierarchical_hist_gradient_boosting_estimator(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return model


def build_hierarchical_hist_gradient_boosting_estimator(
    *,
    max_iter: int,
    random_state: int,
    class_weight: str | dict[str, float] | None,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_depth: int | None,
) -> HierarchicalDelayClassifier:
    return HierarchicalDelayClassifier(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_depth=max_depth,
    )


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
        learning_rate=request.hgb_learning_rate,
        max_leaf_nodes=request.hgb_max_leaf_nodes,
        min_samples_leaf=request.hgb_min_samples_leaf,
        l2_regularization=request.hgb_l2_regularization,
        max_depth=request.hgb_max_depth,
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


# NOTE: XGBoost one of the strongest standard learners
def build_xgboost_balanced(request: ModelTrainingRequest) -> ModelTrainingResult:
    model = train_xgboost_model(
        request.X_train,
        request.y_train,
        random_state=request.random_state,
        n_estimators=request.rf_n_estimators,
        class_weight="balanced",
        max_depth=request.rf_max_depth,
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight="balanced",
        model_family="xgboost",
        algorithm="XGBClassifier",
        preprocessing="none",
    )


# NOTE: A hierarchical classifier is a good future-facing option because the
# NOTE: hardest part of this dataset is separating delayed flights from the big
# NOTE: on_time class before splitting the minority classes from each other.
def build_hierarchical_hist_gradient_boosting(
    request: ModelTrainingRequest,
) -> ModelTrainingResult:
    model = train_hierarchical_model(
        request.X_train,
        request.y_train,
        max_iter=request.max_iter,
        random_state=request.random_state,
        class_weight="balanced",
        learning_rate=request.hgb_learning_rate,
        max_leaf_nodes=request.hgb_max_leaf_nodes,
        min_samples_leaf=request.hgb_min_samples_leaf,
        l2_regularization=request.hgb_l2_regularization,
        max_depth=request.hgb_max_depth,
    )
    return ModelTrainingResult(
        model=model,
        y_pred=pd.Series(model.predict(request.X_test), name=TARGET_COLUMN),
        class_weight="balanced",
        model_family="hierarchical_hist_gradient_boosting",
        algorithm="HierarchicalDelayClassifier",
        preprocessing="none",
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
    "hierarchical_hist_gradient_boosting": build_hierarchical_hist_gradient_boosting,
    "extra_trees": build_extra_trees,
    "majority_baseline": build_majority_baseline,
    "random_forest": build_random_forest,
    "xgboost_balanced": build_xgboost_balanced,
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
