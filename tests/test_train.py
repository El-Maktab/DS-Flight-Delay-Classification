"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic tests for the train model

"""

from pathlib import Path

import pandas as pd
import pytest

from flight_delay_classification.modeling.registry import (
    MODEL_MODES,
    ModelTrainingRequest,
    train_model_for_mode,
)
from flight_delay_classification.modeling.run_all_models import (
    build_run_name,
    run_all_models,
)
from flight_delay_classification.modeling.train import (
    build_evaluation_outputs,
    train_and_log_model,
)


def write_modeling_artifacts(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    train_features = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1, 15.0, 15.1],
            "feature_b": [0.0, 0.2, 5.0, 5.2, 10.0, 10.2, 15.0, 15.2],
        }
    )
    train_labels = pd.DataFrame(
        {
            "DELAY_CATEGORY": [
                "on_time",
                "on_time",
                "minor_delay",
                "minor_delay",
                "major_delay",
                "major_delay",
                "cancelled",
                "cancelled",
            ]
        }
    )
    test_features = pd.DataFrame(
        {
            "feature_a": [0.05, 5.05, 10.05, 15.05],
            "feature_b": [0.1, 5.1, 10.1, 15.1],
        }
    )
    test_labels = pd.DataFrame(
        {
            "DELAY_CATEGORY": [
                "on_time",
                "minor_delay",
                "major_delay",
                "cancelled",
            ]
        }
    )

    features_path = base_dir / "features.csv"
    labels_path = base_dir / "labels.csv"
    test_features_path = base_dir / "test_features.csv"
    test_labels_path = base_dir / "test_labels.csv"

    train_features.to_csv(features_path, index=False)
    train_labels.to_csv(labels_path, index=False)
    test_features.to_csv(test_features_path, index=False)
    test_labels.to_csv(test_labels_path, index=False)

    return features_path, labels_path, test_features_path, test_labels_path


def test_build_evaluation_outputs_returns_expected_shapes() -> None:
    y_true = pd.Series(["on_time", "minor_delay", "major_delay", "cancelled"])
    y_pred = pd.Series(["on_time", "minor_delay", "on_time", "cancelled"])
    class_order = ["on_time", "minor_delay", "major_delay", "cancelled"]

    metrics, report, confusion = build_evaluation_outputs(y_true, y_pred)

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    }
    assert confusion.shape == (4, 4)
    assert confusion.index.tolist() == class_order
    assert confusion.columns.tolist() == class_order
    assert report["on_time"]["recall"] == 1.0


def test_train_and_log_model_saves_model_and_mlfow_run(tmp_path: Path) -> None:
    features_path, labels_path, test_features_path, test_labels_path = (
        write_modeling_artifacts(tmp_path)
    )
    tracking_db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db_path.resolve().as_posix()}"
    model_path = tmp_path / "models" / "model.pkl"

    summary = train_and_log_model(
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        model_path=model_path,
        experiment_name="pytest-flight-delay-baseline",
        run_name="baseline-test-run",
        tracking_uri=tracking_uri,
        max_iter=500,
        random_state=42,
    )

    assert model_path.exists()
    assert summary["run_id"]
    assert summary["tracking_uri"] == tracking_uri
    assert summary["metrics"]["macro_f1"] >= 0.0
    assert tracking_db_path.exists()
    assert (tmp_path / "mlartifacts").exists()


def test_train_model_for_mode_uses_registry_for_majority_baseline() -> None:
    X_train = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0], "feature_b": [0.0, 1.0, 2.0]})
    y_train = pd.Series(["on_time", "on_time", "minor_delay"])
    X_test = pd.DataFrame({"feature_a": [3.0, 4.0], "feature_b": [3.0, 4.0]})

    result = train_model_for_mode(
        "majority_baseline",
        ModelTrainingRequest(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            max_iter=100,
            rf_n_estimators=10,
            rf_class_weight="balanced_subsample",
            rf_min_samples_leaf=1,
            rf_max_depth=3,
            random_state=42,
        ),
    )

    assert "majority_baseline" in MODEL_MODES
    assert result.model is None
    assert result.algorithm == "majority_baseline"
    assert result.y_pred.tolist() == ["on_time", "on_time"]


@pytest.mark.parametrize(
    ("model_mode", "expected_algorithm"),
    [
        (
            "hierarchical_hist_gradient_boosting",
            "HierarchicalDelayClassifier",
        ),
        ("extra_trees", "ExtraTreesClassifier"),
        ("hist_gradient_boosting", "HistGradientBoostingClassifier"),
        ("xgboost_balanced", "XGBClassifier"),
    ],
)
def test_train_model_for_mode_supports_new_models(
    model_mode: str,
    expected_algorithm: str,
) -> None:
    X_train = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1, 15.0, 15.1],
            "feature_b": [0.0, 0.2, 5.0, 5.2, 10.0, 10.2, 15.0, 15.2],
        }
    )
    y_train = pd.Series(
        [
            "on_time",
            "on_time",
            "minor_delay",
            "minor_delay",
            "major_delay",
            "major_delay",
            "cancelled",
            "cancelled",
        ]
    )
    X_test = pd.DataFrame(
        {
            "feature_a": [0.05, 5.05, 10.05, 15.05],
            "feature_b": [0.1, 5.1, 10.1, 15.1],
        }
    )

    result = train_model_for_mode(
        model_mode,
        ModelTrainingRequest(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            max_iter=50,
            rf_n_estimators=20,
            rf_class_weight="balanced_subsample",
            rf_min_samples_leaf=1,
            rf_max_depth=3,
            random_state=42,
        ),
    )

    assert model_mode in MODEL_MODES
    assert result.model is not None
    assert result.algorithm == expected_algorithm
    assert len(result.y_pred) == len(X_test)


def test_train_model_for_mode_rejects_unknown_mode() -> None:
    request = ModelTrainingRequest(
        X_train=pd.DataFrame({"feature_a": [0.0], "feature_b": [0.0]}),
        y_train=pd.Series(["on_time"]),
        X_test=pd.DataFrame({"feature_a": [1.0], "feature_b": [1.0]}),
        max_iter=100,
        rf_n_estimators=10,
        rf_class_weight="balanced_subsample",
        rf_min_samples_leaf=1,
        rf_max_depth=3,
        random_state=42,
    )

    with pytest.raises(ValueError, match="Invalid model_mode"):
        train_model_for_mode("does_not_exist", request)


def test_build_run_name_appends_model_descriptor() -> None:
    assert (
        build_run_name("feature-refresh", "random_forest", use_smote=False)
        == "feature-refresh-random_forest"
    )
    assert (
        build_run_name("feature-refresh", "random_forest", use_smote=True)
        == "feature-refresh-random_forest-smote"
    )


def test_run_all_models_runs_each_registered_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_path, labels_path, test_features_path, test_labels_path = (
        write_modeling_artifacts(tmp_path)
    )
    calls: list[dict[str, object]] = []

    def fake_train_and_log_model(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "model_path": str(kwargs["model_path"]),
            "run_id": f"run-{kwargs['model_mode']}",
            "tracking_uri": kwargs["tracking_uri"],
            "metrics": {"macro_f1": 0.5},
            "class_order": ["on_time"],
        }

    monkeypatch.setattr(
        "flight_delay_classification.modeling.run_all_models.train_and_log_model",
        fake_train_and_log_model,
    )

    summaries = run_all_models(
        experiment_name="exp-123",
        run_prefix="feature-refresh",
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        tracking_uri="sqlite:///tmp.db",
    )

    assert [call["model_mode"] for call in calls] == list(MODEL_MODES)
    assert [summary["model_mode"] for summary in summaries] == list(MODEL_MODES)
    assert [summary["run_name"] for summary in summaries] == [
        build_run_name("feature-refresh", model_mode, use_smote=False)
        for model_mode in MODEL_MODES
    ]


def test_run_all_models_applies_smote_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_path, labels_path, test_features_path, test_labels_path = (
        write_modeling_artifacts(tmp_path)
    )
    smote_calls = 0
    train_calls: list[dict[str, object]] = []

    def fake_load_modeling_artifacts(
        **_: object,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        return (
            pd.read_csv(features_path),
            pd.read_csv(labels_path)["DELAY_CATEGORY"],
            pd.read_csv(test_features_path),
            pd.read_csv(test_labels_path)["DELAY_CATEGORY"],
        )

    def fake_apply_smote(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        nonlocal smote_calls
        assert random_state == 42
        smote_calls += 1
        return X_train.assign(smote_marker=1.0), y_train

    def fake_train_and_log_model(**kwargs: object) -> dict[str, object]:
        train_calls.append(kwargs)
        return {
            "model_path": str(kwargs["model_path"]),
            "run_id": f"run-{kwargs['model_mode']}",
            "tracking_uri": kwargs["tracking_uri"],
            "metrics": {"macro_f1": 0.5},
            "class_order": ["on_time"],
        }

    monkeypatch.setattr(
        "flight_delay_classification.modeling.run_all_models.load_modeling_artifacts",
        fake_load_modeling_artifacts,
    )
    monkeypatch.setattr(
        "flight_delay_classification.modeling.run_all_models.apply_smote",
        fake_apply_smote,
    )
    monkeypatch.setattr(
        "flight_delay_classification.modeling.run_all_models.train_and_log_model",
        fake_train_and_log_model,
    )

    run_all_models(
        experiment_name="exp-123",
        run_prefix="feature-refresh",
        features_path=features_path,
        labels_path=labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        tracking_uri="sqlite:///tmp.db",
        use_smote=True,
    )

    assert smote_calls == 1
    assert len(train_calls) == len(MODEL_MODES)
    assert all("smote_marker" in call["X_train"].columns for call in train_calls)
