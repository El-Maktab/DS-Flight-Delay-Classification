"""
Author: Akram Hany
Date: 2026-04-18

Description:
    Preprocessing pipeline for the flights_weather dataset.

"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

# Import validation checks so we can re-run them after cleaning
from flight_delay_classification.validation.validate import (
    check_completeness,
    check_ranges,
)

log = logging.getLogger(__name__)

# Constants

# Columns to drop because they are post-departure (target leakage) or identifiers
LEAKAGE_COLUMNS: list[str] = [
    "FLIGHT_NUMBER",  # row identifier - not a predictive feature
    "TAIL_NUMBER",  # aircraft identifier - not predictive, also 0.22% null
    "DEPARTURE_TIME",  # actual departure - only known after pushback
    "ARRIVAL_TIME",  # actual arrival - only known after landing
    "WHEELS_OFF",  # actual wheels-off - only known after pushback
    "WHEELS_ON",  # actual wheels-on - only known after landing
    "ELAPSED_TIME",  # actual elapsed - only known after landing
    "AIR_TIME",  # actual airborne - only known after landing
    "TAXI_OUT",  # actual taxi-out - only known after pushback
    "TAXI_IN",  # actual taxi-in - only known after landing
    "ARRIVAL_DELAY",  # used only to derive target; not a predictor
    "AIR_SYSTEM_DELAY",  # post-departure delay breakdown - leakage
    "SECURITY_DELAY",  # post-departure delay breakdown - leakage
    "AIRLINE_DELAY",  # post-departure delay breakdown - leakage
    "LATE_AIRCRAFT_DELAY",  # post-departure delay breakdown - leakage
    "WEATHER_DELAY",  # post-departure delay breakdown - leakage
]

# Columns where outlier capping is meaningful (numeric, model-relevant)
OUTLIER_COLS: list[str] = [
    "DISTANCE",
    "SCHEDULED_TIME",
    "TAXI_OUT",
    "TAXI_IN",
    "AIR_TIME",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kmh",
    "wind_gusts_kmh",
    "dest_temperature_c",
    "dest_precipitation_mm",
    "dest_wind_speed_kmh",
    "dest_wind_gusts_kmh",
]

RANDOM_STATE: int = 42


# 1. Drop leakage / identifier columns


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all post-departure and identifier columns that would cause
    target leakage or are not useful as predictive features.
    """
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(
        "[step 1 | drop_leakage_columns] Dropped %d columns: %s",
        len(cols_to_drop),
        cols_to_drop,
    )
    return df


# 2. Drop rows with numeric airport codes


def drop_numeric_airport_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where ORIGIN_AIRPORT or DESTINATION_AIRPORT contains a numeric
    FAA station ID instead of an IATA code.
    """
    before = len(df)

    origin_numeric = df["ORIGIN_AIRPORT"].str.isnumeric()
    dest_numeric = df["DESTINATION_AIRPORT"].str.isnumeric()

    df = df[~(origin_numeric | dest_numeric)].copy()

    dropped = before - len(df)
    log.info(
        "[step 2 | drop_numeric_airport_rows] Dropped %d rows (%.2f%%) "
        "with numeric FAA airport codes — no IATA match → no weather data.",
        dropped,
        dropped / before * 100,
    )
    return df


# 3. Create target variable


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build DELAY_CATEGORY target column:
        on_time     : DEPARTURE_DELAY <= 15 min  (FAA on-time standard)
        minor_delay : 15 < DEPARTURE_DELAY <= 45 min
        major_delay : DEPARTURE_DELAY > 45 min
        cancelled   : CANCELLED == 1

    DEPARTURE_DELAY and CANCELLED columns are dropped after the target is created.
    """
    df = df.copy()

    # Cancelled flights get their own class regardless of DEPARTURE_DELAY
    cancelled_mask = df["CANCELLED"] == 1
    delay_mask = ~cancelled_mask & df["DEPARTURE_DELAY"].notna()

    df.loc[delay_mask, "DELAY_CATEGORY"] = pd.cut(
        df.loc[delay_mask, "DEPARTURE_DELAY"],
        bins=[-float("inf"), 15, 45, float("inf")],
        labels=["on_time", "minor_delay", "major_delay"],
    ).astype(str)

    df.loc[cancelled_mask, "DELAY_CATEGORY"] = "cancelled"

    class_counts = df["DELAY_CATEGORY"].value_counts(dropna=False).to_dict()
    log.info(
        "[step 3 | create_target] DELAY_CATEGORY created. Distribution: %s",
        class_counts,
    )

    # Drop columns as they are no longer needed
    df = df.drop(columns=["DEPARTURE_DELAY", "CANCELLED"])
    log.info("[step 3 | create_target] Dropped DEPARTURE_DELAY and CANCELLED columns.")
    return df


# 4. Drop unlabellable rows


def drop_unlabellable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any row where DELAY_CATEGORY is still null after target creation.
    """
    before = len(df)
    df = df.dropna(
        subset=["DELAY_CATEGORY"]
    )  # drop na in column DELAY_CATEGORY (should not happen)
    dropped = before - len(df)
    log.info(
        "[step 4 | drop_unlabellable_rows] Dropped %d rows with no assignable target class.",
        dropped,
    )
    return df


# 5. Cap outliers


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set extreme values (outliers) to the 1st and 99th percentiles.
    """
    df = df.copy()
    total_affected = 0

    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)

        below = (df[col] < lower).sum()
        above = (df[col] > upper).sum()
        affected = int(below + above)

        if affected > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_affected += affected
            log.info(
                "[step 5 | cap_outliers] %-30s capped %4d rows "
                "(below %.2f: %d, above %.2f: %d)",
                col,
                affected,
                lower,
                int(below),
                upper,
                int(above),
            )

    log.info(
        "[step 5 | cap_outliers] Total rows affected by capping: %d",
        total_affected,
    )
    return df


# 6. Drop remaining nulls


def drop_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any row that still has a null in a feature column that is not expected to be null.
    """
    before = len(df)

    EXPECTED_NULL_COLS = {
        "CANCELLATION_REASON",  # only populated for cancelled flights
    }

    check_cols = [c for c in df.columns if c not in EXPECTED_NULL_COLS]
    df = df.dropna(subset=check_cols)

    dropped = before - len(df)
    log.info(
        "[step 6 | drop_remaining_nulls] Dropped %d rows with residual "
        "null values in non-expected feature columns.",
        dropped,
    )
    return df


# 7. Validate after cleaning


def validate_after_cleaning(df: pd.DataFrame) -> dict:
    """
    Re-run a subset of validation checks on the cleaned DataFrame to confirm
    that the known issues have been resolved.
    """
    log.info("[step 7 | validate_after_cleaning] Re-running validation checks...")

    results = {
        "completeness": check_completeness(df),
        "ranges": check_ranges(df),
    }

    for check, result in results.items():
        icon = "\u2713" if result["passed"] else "\u2717"
        log.info(
            "[step 7 | validate_after_cleaning]   [%s] %-20s %s",
            icon,
            check,
            result["summary"],
        )

    all_passed = all(r["passed"] for r in results.values())
    if not all_passed:
        log.warning("[step 7 | validate_after_cleaning] Some checks did not pass.")
    else:
        log.info(
            "[step 7 | validate_after_cleaning] All checks passed on cleaned data."
        )

    return results


# 8. Save cleaned data to CSV


def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned DataFrame to a CSV file so downstream stages
    (feature engineering, modeling) can load it without re-running
    preprocessing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(
        "[step 8 | save_to_csv] Saved %d rows × %d columns to %s",
        *df.shape,
        output_path,
    )


# 9. Before / after summary


def summarize(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> dict:
    """
    Build a summary dict comparing the dataset before and after cleaning.
    """

    def _missing(df: pd.DataFrame) -> int:
        return int(df.isnull().sum().sum())

    summary = {
        "before": {
            "rows": len(df_before),
            "columns": len(df_before.columns),
            "missing_cells": _missing(df_before),
        },
        "after": {
            "rows": len(df_after),
            "columns": len(df_after.columns),
            "missing_cells": _missing(df_after),
            "rows_dropped": len(df_before) - len(df_after),
            "columns_dropped": len(df_before.columns) - len(df_after.columns),
        },
    }

    log.info(
        "[step 9 | summarize] Before: %d rows × %d cols, %d missing cells",
        summary["before"]["rows"],
        summary["before"]["columns"],
        summary["before"]["missing_cells"],
    )
    log.info(
        "[step 9 | summarize] After:  %d rows × %d cols, %d missing cells "
        "(%d rows dropped, %d cols dropped)",
        summary["after"]["rows"],
        summary["after"]["columns"],
        summary["after"]["missing_cells"],
        summary["after"]["rows_dropped"],
        summary["after"]["columns_dropped"],
    )

    return summary


# Main pipeline


def run_pipeline(
    df: pd.DataFrame, output_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline on a copy of the input DataFrame.
    The original DataFrame is never mutated.

    Steps:
        1.  drop_leakage_columns      - remove post-departure & identifier cols
        2.  drop_numeric_airport_rows - remove rows with no IATA/weather match
        3.  create_target             - build 4-class DELAY_CATEGORY
        4.  drop_unlabellable_rows    - remove rows with no assignable class
        5.  cap_outliers              - 1st-99th percentile capping
        6.  drop_remaining_nulls      - drop any residual unexpected nulls
        7.  validate_after_cleaning   - re-run validation checks
        8.  save to CSV               - write cleaned data to output_path
        9.  summarize                 - before/after summary dict

    Data splitting should be done after preprocessing, during the feature engineering stage.

    Args:
        df:          Input DataFrame (flights_weather.csv).
        output_path: Where to save the cleaned CSV.
                     Defaults to data/processed/flights_cleaned.csv.

    Returns:
        (df_cleaned, summary)
    """
    if output_path is None:
        output_path = (
            Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
            / "flights_cleaned.csv"
        )

    log.info("=" * 60)
    log.info("PREPROCESSING PIPELINE - START")
    log.info("Input: %d rows * %d columns", *df.shape)
    log.info("=" * 60)

    df_before = df.copy()  # keep an unmodified snapshot for the summary
    df = df.copy()  # work on a copy to not mutate the original df

    df = drop_leakage_columns(df)
    df = drop_numeric_airport_rows(df)
    df = create_target(df)
    df = drop_unlabellable_rows(df)
    df = cap_outliers(df)
    df = drop_remaining_nulls(df)

    validation_results = validate_after_cleaning(df)

    save_to_csv(df, output_path)

    summary = summarize(df_before, df)
    summary["validation_after_cleaning"] = {
        k: {"passed": v["passed"], "summary": v["summary"]}
        for k, v in validation_results.items()
    }

    log.info("=" * 60)
    log.info("PREPROCESSING PIPELINE DONE")
    log.info("=" * 60)

    return df, summary


# Entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    data_path = processed_dir / "flights_weather.csv"

    log.info("Loading %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    log.info("Loaded %d rows * %d columns", *df.shape)

    _, summary = run_pipeline(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(
        f"  Before : {summary['before']['rows']:>7,} rows * {summary['before']['columns']} columns"
    )
    print(
        f"  After  : {summary['after']['rows']:>7,} rows * {summary['after']['columns']} columns"
    )
    print(
        f"  Dropped: {summary['after']['rows_dropped']:>7,} rows, {summary['after']['columns_dropped']} columns"
    )
    print(f"  Missing cells after: {summary['after']['missing_cells']}")
    print()
    print("  Post-cleaning validation:")
    for check, res in summary["validation_after_cleaning"].items():
        icon = "✓" if res["passed"] else "✗"
        print(f"    [{icon}] {check:<20}  {res['summary']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
