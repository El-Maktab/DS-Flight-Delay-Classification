"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Basic tests for the feature pipeline

"""

from pathlib import Path

import pandas as pd
import pytest

from flight_delay_classification.features import (
    add_smoothed_historical_rate_features,
    adapt_features_for_model_mode,
    build_feature_matrices,
    prepare_feature_artifacts,
    select_informative_features,
)

EXPECTED_HISTORICAL_COLUMNS = [
    "airline_historical_delay_rate",
    "airline_historical_on_time_rate",
    "airline_historical_minor_delay_rate",
    "airline_historical_severe_rate",
    "airline_historical_major_delay_rate",
    "airline_historical_cancelled_rate",
    "origin_airport_historical_delay_rate",
    "origin_airport_historical_on_time_rate",
    "origin_airport_historical_minor_delay_rate",
    "origin_airport_historical_severe_rate",
    "origin_airport_historical_major_delay_rate",
    "origin_airport_historical_cancelled_rate",
    "destination_airport_historical_delay_rate",
    "destination_airport_historical_on_time_rate",
    "destination_airport_historical_minor_delay_rate",
    "destination_airport_historical_severe_rate",
    "destination_airport_historical_major_delay_rate",
    "destination_airport_historical_cancelled_rate",
    "route_historical_delay_rate",
    "route_historical_on_time_rate",
    "route_historical_minor_delay_rate",
    "route_historical_severe_rate",
    "route_historical_major_delay_rate",
    "route_historical_cancelled_rate",
]

EXPECTED_TEMPORAL_COLUMNS = [
    "is_weekend",
    "is_holiday",
    "scheduled_departure_hour_sin",
    "scheduled_departure_hour_cos",
    "scheduled_arrival_hour_sin",
    "scheduled_arrival_hour_cos",
    "month_sin",
    "month_cos",
    "is_morning_peak_bank",
    "is_evening_peak_bank",
]

EXPECTED_OPERATIONAL_INTERACTION_COLUMNS = [
    "departure_hour_bucket_code",
    "origin_hourly_departure_count",
    "destination_hourly_arrival_count",
    "origin_departure_bank_count",
    "destination_departure_bank_count",
    "origin_congestion_ratio",
    "destination_congestion_ratio",
    "origin_departure_bank_ratio",
    "destination_departure_bank_ratio",
    "origin_weather_intensity",
    "destination_weather_intensity",
    "origin_congestion_weather_score",
    "destination_congestion_weather_score",
    "route_congestion_weather_score",
    "origin_departure_bank_weather_score",
    "destination_departure_bank_weather_score",
]

EXPECTED_INTERACTION_HISTORICAL_COLUMNS = [
    "airline_departure_bank_historical_delay_rate",
    "airline_departure_bank_historical_on_time_rate",
    "airline_departure_bank_historical_minor_delay_rate",
    "airline_departure_bank_historical_severe_rate",
    "airline_departure_bank_historical_major_delay_rate",
    "airline_departure_bank_historical_cancelled_rate",
    "origin_departure_bank_historical_delay_rate",
    "origin_departure_bank_historical_on_time_rate",
    "origin_departure_bank_historical_minor_delay_rate",
    "origin_departure_bank_historical_severe_rate",
    "origin_departure_bank_historical_major_delay_rate",
    "origin_departure_bank_historical_cancelled_rate",
    "origin_day_of_week_historical_delay_rate",
    "origin_day_of_week_historical_on_time_rate",
    "origin_day_of_week_historical_minor_delay_rate",
    "origin_day_of_week_historical_severe_rate",
    "origin_day_of_week_historical_major_delay_rate",
    "origin_day_of_week_historical_cancelled_rate",
    "airline_route_historical_delay_rate",
    "airline_route_historical_on_time_rate",
    "airline_route_historical_minor_delay_rate",
    "airline_route_historical_severe_rate",
    "airline_route_historical_major_delay_rate",
    "airline_route_historical_cancelled_rate",
    "route_departure_bank_historical_delay_rate",
    "route_departure_bank_historical_on_time_rate",
    "route_departure_bank_historical_minor_delay_rate",
    "route_departure_bank_historical_severe_rate",
    "route_departure_bank_historical_major_delay_rate",
    "route_departure_bank_historical_cancelled_rate",
    "route_day_of_week_historical_delay_rate",
    "route_day_of_week_historical_on_time_rate",
    "route_day_of_week_historical_minor_delay_rate",
    "route_day_of_week_historical_severe_rate",
    "route_day_of_week_historical_major_delay_rate",
    "route_day_of_week_historical_cancelled_rate",
]


def test_build_feature_matrices_aligns_test_columns() -> None:
    train_df = pd.DataFrame(
        {
            "YEAR": [2015, 2015],
            "MONTH": [1, 7],
            "DAY": [1, 4],
            "DAY_OF_WEEK": [4, 6],
            "SCHEDULED_DEPARTURE": [530, 1845],
            "SCHEDULED_ARRIVAL": [745, 2105],
            "temperature_c": [-2.0, 18.0],
            "precipitation_mm": [6.0, 0.0],
            "rain_mm": [6.0, 0.0],
            "snowfall_cm": [0.0, 0.0],
            "wind_speed_kmh": [10.0, 15.0],
            "wind_gusts_kmh": [45.0, 18.0],
            "dest_temperature_c": [15.0, -3.0],
            "dest_precipitation_mm": [0.0, 3.0],
            "dest_rain_mm": [0.0, 3.0],
            "dest_snowfall_cm": [0.0, 1.0],
            "dest_wind_speed_kmh": [12.0, 42.0],
            "dest_wind_gusts_kmh": [20.0, 55.0],
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
            "MONTH": [11],
            "DAY": [26],
            "DAY_OF_WEEK": [4],
            "SCHEDULED_DEPARTURE": [830],
            "SCHEDULED_ARRIVAL": [1130],
            "temperature_c": [-5.0],
            "precipitation_mm": [4.0],
            "rain_mm": [4.0],
            "snowfall_cm": [2.0],
            "wind_speed_kmh": [35.0],
            "wind_gusts_kmh": [50.0],
            "dest_temperature_c": [-1.0],
            "dest_precipitation_mm": [3.0],
            "dest_rain_mm": [3.0],
            "dest_snowfall_cm": [0.0],
            "dest_wind_speed_kmh": [10.0],
            "dest_wind_gusts_kmh": [25.0],
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
    for column in EXPECTED_TEMPORAL_COLUMNS:
        assert column in train_features.columns
        assert column in test_features.columns
    for column in EXPECTED_OPERATIONAL_INTERACTION_COLUMNS:
        assert column in train_features.columns
        assert column in test_features.columns
    for column in EXPECTED_HISTORICAL_COLUMNS:
        assert column in train_features.columns
        assert column in test_features.columns
    for column in EXPECTED_INTERACTION_HISTORICAL_COLUMNS:
        assert column in train_features.columns
        assert column in test_features.columns
    assert test_features[EXPECTED_TEMPORAL_COLUMNS].notna().all().all()
    assert test_features[EXPECTED_OPERATIONAL_INTERACTION_COLUMNS].notna().all().all()
    assert test_features[EXPECTED_HISTORICAL_COLUMNS].notna().all().all()
    assert test_features[EXPECTED_INTERACTION_HISTORICAL_COLUMNS].notna().all().all()
    assert train_features.loc[0, "is_holiday"] == 1
    assert train_features.loc[1, "is_weekend"] == 1
    assert train_features.loc[1, "is_evening_peak_bank"] == 1
    assert test_features.loc[0, "is_weekend"] == 0
    assert test_features.loc[0, "is_holiday"] == 1
    assert test_features.loc[0, "is_morning_peak_bank"] == 1
    assert test_features.loc[0, "scheduled_departure_hour_sin"] == pytest.approx(
        0.8660254038
    )
    assert test_features.loc[0, "scheduled_departure_hour_cos"] == pytest.approx(-0.5)
    assert train_features.loc[0, "month_sin"] == pytest.approx(0.0)
    assert train_features.loc[0, "month_cos"] == pytest.approx(1.0)
    assert train_features.loc[0, "origin_hourly_departure_count"] == 1
    assert train_features.loc[1, "destination_hourly_arrival_count"] == 1
    assert test_features.loc[0, "origin_weather_intensity"] == 4.0
    assert test_features.loc[0, "destination_weather_intensity"] == 2.0
    assert test_features.loc[0, "origin_congestion_weather_score"] == pytest.approx(4.0)
    assert test_features.loc[
        0, "destination_congestion_weather_score"
    ] == pytest.approx(2.0)
    assert test_features.loc[0, "route_congestion_weather_score"] == pytest.approx(12.0)
    assert test_features.loc[0, "airline_historical_on_time_rate"] == 0.5
    assert test_features.loc[0, "airline_historical_minor_delay_rate"] == 0.5
    assert test_features.loc[0, "airline_historical_major_delay_rate"] == 0.0
    assert test_features.loc[0, "airline_route_historical_on_time_rate"] == 0.5
    assert (
        test_features.loc[0, "route_departure_bank_historical_minor_delay_rate"] == 0.5
    )
    assert test_features.loc[0, "route_day_of_week_historical_major_delay_rate"] == 0.0
    assert test_features.loc[0, "route_historical_cancelled_rate"] == 0.0
    assert train_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert test_labels.columns.tolist() == ["DELAY_CATEGORY"]


def test_prepare_feature_artifacts_writes_expected_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "YEAR": [2015] * 8,
            "MONTH": [1, 1, 2, 2, 3, 3, 4, 4],
            "DAY": [1, 2, 3, 4, 5, 6, 7, 8],
            "DAY_OF_WEEK": [4, 5, 6, 7, 1, 2, 3, 4],
            "SCHEDULED_DEPARTURE": [600, 615, 930, 945, 1230, 1245, 1815, 1830],
            "SCHEDULED_ARRIVAL": [815, 830, 1145, 1200, 1445, 1500, 2030, 2045],
            "temperature_c": [-1.0, 4.0, 10.0, 14.0, 20.0, 22.0, 6.0, 8.0],
            "precipitation_mm": [5.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 7.0],
            "rain_mm": [5.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 7.0],
            "snowfall_cm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "wind_speed_kmh": [18.0, 12.0, 30.0, 20.0, 10.0, 15.0, 28.0, 35.0],
            "wind_gusts_kmh": [44.0, 18.0, 41.0, 25.0, 20.0, 22.0, 45.0, 50.0],
            "dest_temperature_c": [8.0, 9.0, 2.0, 1.0, -2.0, 0.0, 7.0, 6.0],
            "dest_precipitation_mm": [0.0, 0.0, 2.5, 0.0, 4.0, 3.0, 0.0, 2.0],
            "dest_rain_mm": [0.0, 0.0, 2.5, 0.0, 4.0, 3.0, 0.0, 2.0],
            "dest_snowfall_cm": [0.0, 0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0],
            "dest_wind_speed_kmh": [12.0, 15.0, 32.0, 18.0, 14.0, 28.0, 20.0, 30.0],
            "dest_wind_gusts_kmh": [18.0, 22.0, 46.0, 26.0, 24.0, 43.0, 25.0, 44.0],
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
    for column in EXPECTED_TEMPORAL_COLUMNS:
        assert column in written_train_features.columns
        assert column in written_test_features.columns
    for column in EXPECTED_OPERATIONAL_INTERACTION_COLUMNS:
        assert column in written_train_features.columns
        assert column in written_test_features.columns
    for column in EXPECTED_HISTORICAL_COLUMNS:
        assert column in written_train_features.columns
        assert column in written_test_features.columns
    for column in EXPECTED_INTERACTION_HISTORICAL_COLUMNS:
        assert column in written_train_features.columns
        assert column in written_test_features.columns
    assert written_train_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert written_test_labels.columns.tolist() == ["DELAY_CATEGORY"]
    assert summary["train_rows"] == len(written_train_features)
    assert summary["test_rows"] == len(written_test_features)


def test_select_informative_features_keeps_only_signal_columns() -> None:
    train_features = pd.DataFrame(
        {
            "signal_feature": [0, 0, 0, 0, 1, 1, 1, 1],
            "weak_noise": [0, 1, 0, 1, 0, 1, 0, 1],
            "constant_feature": [1] * 8,
        }
    )
    test_features = pd.DataFrame(
        {
            "signal_feature": [0, 1],
            "weak_noise": [1, 0],
            "constant_feature": [1, 1],
        }
    )
    y_train = pd.Series(
        [
            "on_time",
            "on_time",
            "on_time",
            "on_time",
            "major_delay",
            "major_delay",
            "major_delay",
            "major_delay",
        ]
    )

    selected_train, selected_test, selected_columns = select_informative_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        method="mutual_info",
        min_mutual_info=0.2,
    )

    assert selected_columns == ["signal_feature"]
    assert selected_train.columns.tolist() == ["signal_feature"]
    assert selected_test.columns.tolist() == ["signal_feature"]


def test_smoothed_historical_rate_features_use_leave_one_out_on_train() -> None:
    train_features = pd.DataFrame({"AIRLINE": ["AA", "AA", "UA"]})
    test_features = pd.DataFrame({"AIRLINE": ["AA", "UA", "DL"]})
    y_train = pd.Series(["on_time", "major_delay", "minor_delay"])

    encoded_train, encoded_test = add_smoothed_historical_rate_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        encoding_columns={"AIRLINE": "airline"},
    )

    assert encoded_train.loc[0, "airline_historical_on_time_rate"] == pytest.approx(
        25 / 78
    )
    assert encoded_train.loc[1, "airline_historical_on_time_rate"] == pytest.approx(
        28 / 78
    )
    assert encoded_train.loc[2, "airline_historical_on_time_rate"] == pytest.approx(
        1 / 3
    )
    assert encoded_test.loc[0, "airline_historical_on_time_rate"] == pytest.approx(
        28 / 81
    )
    assert encoded_test.loc[1, "airline_historical_on_time_rate"] == pytest.approx(
        25 / 78
    )
    assert encoded_test.loc[2, "airline_historical_on_time_rate"] == pytest.approx(
        1 / 3
    )


def test_adapt_features_for_model_mode_drops_historical_features_for_hist_gradient() -> (
    None
):
    train_features = pd.DataFrame(
        {
            "signal_feature": [0.0, 1.0],
            "route_historical_on_time_rate": [0.2, 0.8],
            "airline_route_historical_major_delay_rate": [0.1, 0.9],
        }
    )
    test_features = pd.DataFrame(
        {
            "signal_feature": [0.5],
            "route_historical_on_time_rate": [0.4],
            "airline_route_historical_major_delay_rate": [0.6],
        }
    )

    adapted_train, adapted_test, dropped_columns = adapt_features_for_model_mode(
        train_features=train_features,
        test_features=test_features,
        model_mode="hist_gradient_boosting",
    )

    assert adapted_train.columns.tolist() == ["signal_feature"]
    assert adapted_test.columns.tolist() == ["signal_feature"]
    assert dropped_columns == [
        "route_historical_on_time_rate",
        "airline_route_historical_major_delay_rate",
    ]


def test_adapt_features_for_model_mode_keeps_historical_features_for_logistic() -> None:
    train_features = pd.DataFrame(
        {
            "signal_feature": [0.0, 1.0],
            "route_historical_on_time_rate": [0.2, 0.8],
        }
    )
    test_features = pd.DataFrame(
        {
            "signal_feature": [0.5],
            "route_historical_on_time_rate": [0.4],
        }
    )

    adapted_train, adapted_test, dropped_columns = adapt_features_for_model_mode(
        train_features=train_features,
        test_features=test_features,
        model_mode="logreg_balanced",
    )

    assert adapted_train.columns.tolist() == train_features.columns.tolist()
    assert adapted_test.columns.tolist() == test_features.columns.tolist()
    assert dropped_columns == []
