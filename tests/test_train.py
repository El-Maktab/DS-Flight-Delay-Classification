"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic tests for the train model

"""

from pathlib import Path

import pandas as pd

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
