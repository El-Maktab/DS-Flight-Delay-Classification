import numpy as np
import pandas as pd
import pytest

from flight_delay_classification.preprocessing.preprocess import (
    LEAKAGE_COLUMNS,
    OUTLIER_COLS,
    cap_outliers,
    create_target,
    drop_leakage_columns,
    drop_numeric_airport_rows,
    drop_unlabellable_rows,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "FLIGHT_NUMBER": [123, 456, 789, 101, 102],
            "ORIGIN_AIRPORT": ["ATL", "12345", "JFK", "LAX", "ORD"],
            "DESTINATION_AIRPORT": ["JFK", "ORD", "67890", "SFO", "MIA"],
            "DEPARTURE_DELAY": [-5, 10, 20, 60, np.nan],
            "CANCELLED": [0, 0, 0, 0, 1],
            "DISTANCE": [500, 1000, 1500, 10000, 200],
            "ARRIVAL_DELAY": [0, 5, 10, 50, np.nan],
            "DAY_OF_WEEK": [1, 2, 3, 4, 5],
        }
    )


def test_drop_leakage_columns(sample_df):
    """Test that leakage and identifier columns are removed."""
    df_out = drop_leakage_columns(sample_df)

    for col in LEAKAGE_COLUMNS:
        assert col not in df_out.columns

    assert "ORIGIN_AIRPORT" in df_out.columns
    assert "DEPARTURE_DELAY" in df_out.columns


def test_drop_numeric_airport_rows(sample_df):
    """Test that rows with numeric airport codes are dropped."""
    df_out = drop_numeric_airport_rows(sample_df)

    assert len(df_out) == 3
    assert "12345" not in df_out["ORIGIN_AIRPORT"].values
    assert "67890" not in df_out["DESTINATION_AIRPORT"].values


def test_create_target():
    """Test that DELAY_CATEGORY is correctly created based on business rules."""
    df = pd.DataFrame(
        {
            "DEPARTURE_DELAY": [-10, 0, 15, 16, 45, 46, 100, np.nan, np.nan],
            "CANCELLED": [0, 0, 0, 0, 0, 0, 0, 1, 0],
        }
    )

    df_out = create_target(df)

    expected = [
        "on_time",
        "on_time",
        "on_time",
        "minor_delay",
        "minor_delay",
        "major_delay",
        "major_delay",
        "cancelled",
        np.nan,
    ]

    result_list = df_out["DELAY_CATEGORY"].fillna("missing").tolist()
    expected_list = ["missing" if pd.isna(x) else x for x in expected]

    assert result_list == expected_list


def test_drop_unlabellable_rows():
    """Test that rows with null DELAY_CATEGORY are dropped."""
    df = pd.DataFrame({"DELAY_CATEGORY": ["on_time", "minor_delay", np.nan, "major_delay"]})

    df_out = drop_unlabellable_rows(df)

    assert len(df_out) == 3
    assert df_out["DELAY_CATEGORY"].isna().sum() == 0


def test_cap_outliers():
    """Test that outliers are capped at the 1st and 99th percentiles."""
    data = list(range(1, 100)) + [-1000, 10000]
    df = pd.DataFrame({"DISTANCE": data})

    df["NON_OUTLIER"] = data

    assert "DISTANCE" in OUTLIER_COLS

    df_out = cap_outliers(df)

    assert df_out["DISTANCE"].max() < 10000
    assert df_out["DISTANCE"].min() > -1000

    assert df_out["NON_OUTLIER"].max() == 10000
    assert df_out["NON_OUTLIER"].min() == -1000
