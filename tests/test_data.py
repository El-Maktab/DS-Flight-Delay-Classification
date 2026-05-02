"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic tests for the feature pipeline

"""

from pathlib import Path

import pandas as pd

from flight_delay_classification.features import (
    build_feature_matrices,
    prepare_feature_artifacts,
)


EXPECTED_HISTORICAL_COLUMNS = [
    "airline_historical_delay_rate",
    "airline_historical_severe_rate",
    "airline_historical_cancelled_rate",
    "origin_airport_historical_delay_rate",
    "origin_airport_historical_severe_rate",
    "origin_airport_historical_cancelled_rate",
    "destination_airport_historical_delay_rate",
    "destination_airport_historical_severe_rate",
    "destination_airport_historical_cancelled_rate",
    "route_historical_delay_rate",
    "route_historical_severe_rate",
    "route_historical_cancelled_rate",
]


def test_build_feature_matrices_aligns_test_columns() -> None:
    train_df = pd.DataFrame(
        {
            "YEAR": [2015, 2015],
            "AIRLINE": ["AA", "DL"],
            "ORIGIN_AIRPORT": ["ATL", "JFK"],
            "DESTINATION_AIRPORT": ["LAX", "SFO"],
            "DIVERTED": [0, 0],
            "CANCELLATION_REASON": [None, None],
            "DELAY_CATEGORY": ["on_time", "minor_delay"],
        }
    )
    test_df = pd.DataFrame(
        {
            "YEAR": [2015],
            "AIRLINE": ["UA"],
            "ORIGIN_AIRPORT": ["LAX"],
            "DESTINATION_AIRPORT": ["SEA"],
            "DIVERTED": [0],
            "CANCELLATION_REASON": [None],
            "DELAY_CATEGORY": ["major_delay"],
        }
    )

    train_features, train_labels, test_features, test_labels = build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
    )

    assert list(train_features.columns) == list(test_features.columns)
    assert "DIVERTED" not in train_features.columns
    assert "CANCELLATION_REASON" not in train_features.columns
    assert "AIRLINE" not in train_features.columns
    assert "ORIGIN_AIRPORT" not in train_features.columns
    assert "DESTINATION_AIRPORT" not in train_features.columns
    assert "ROUTE" not in train_features.columns
    for column in EXPECTED_HISTORICAL_COLUMNS:
        assert column in train_features.columns
        assert column in test_features.columns
    assert test_features[EXPECTED_HISTORICAL_COLUMNS].notna().all().all()
    assert train_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert test_labels.columns.tolist() == ["DELAY_CATEGORY"]


def test_prepare_feature_artifacts_writes_expected_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "YEAR": [2015] * 8,
            "MONTH": [1, 1, 2, 2, 3, 3, 4, 4],
            "AIRLINE": ["AA", "AA", "DL", "DL", "UA", "UA", "WN", "WN"],
            "ORIGIN_AIRPORT": ["ATL", "ATL", "JFK", "JFK", "ORD", "ORD", "BNA", "BNA"],
            "DESTINATION_AIRPORT": [
                "LAX",
                "LAX",
                "SFO",
                "SFO",
                "SEA",
                "SEA",
                "DAL",
                "DAL",
            ],
            "DISTANCE": [100, 110, 200, 210, 300, 310, 400, 410],
            "DIVERTED": [0] * 8,
            "CANCELLATION_REASON": [None, None, None, None, "A", "A", None, None],
            "DELAY_CATEGORY": [
                "on_time",
                "on_time",
                "minor_delay",
                "minor_delay",
                "major_delay",
                "major_delay",
                "cancelled",
                "cancelled",
            ],
        }
    )
    input_path = tmp_path / "flights_cleaned.csv"
    train_features_path = tmp_path / "features.csv"
    train_labels_path = tmp_path / "labels.csv"
    test_features_path = tmp_path / "test_features.csv"
    test_labels_path = tmp_path / "test_labels.csv"
    df.to_csv(input_path, index=False)

    summary = prepare_feature_artifacts(
        input_path=input_path,
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        test_size=0.5,
        random_state=42,
    )

    written_train_features = pd.read_csv(train_features_path)
    written_test_features = pd.read_csv(test_features_path)
    written_train_labels = pd.read_csv(train_labels_path)
    written_test_labels = pd.read_csv(test_labels_path)

    assert train_features_path.exists()
    assert train_labels_path.exists()
    assert test_features_path.exists()
    assert test_labels_path.exists()
    assert list(written_train_features.columns) == list(written_test_features.columns)
    for column in EXPECTED_HISTORICAL_COLUMNS:
        assert column in written_train_features.columns
        assert column in written_test_features.columns
    assert written_train_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert written_test_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert summary["train_rows"] == len(written_train_features)
    assert summary["test_rows"] == len(written_test_features)
