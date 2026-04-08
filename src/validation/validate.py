"""
Author: Akram Hany
Date: 2026-04-08

Description:
    Data validation module for the flights_weather dataset.
    Each check returns a dict with:
        - passed  : bool
        - summary : short human-readable string
        - details : dict of supporting numbers / DataFrames
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ── Expected schema ────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
    "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY",
    "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME",
    "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN",
    "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY",
    "DIVERTED", "CANCELLED", "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    "departure_hour", "scheduled_arrival_hour",
    "temperature_c", "precipitation_mm", "rain_mm", "snowfall_cm",
    "wind_speed_kmh", "wind_direction_deg", "wind_gusts_kmh",
    "cloud_cover_pct", "weather_code", "relative_humidity_pct", "pressure_msl_hpa",
    "dest_temperature_c", "dest_precipitation_mm", "dest_rain_mm", "dest_snowfall_cm",
    "dest_wind_speed_kmh", "dest_wind_direction_deg", "dest_wind_gusts_kmh",
    "dest_cloud_cover_pct", "dest_weather_code", "dest_relative_humidity_pct",
    "dest_pressure_msl_hpa",
]

EXPECTED_DTYPES: dict[str, list[str]] = {
    "YEAR":                 ["int64", "int32"],
    "MONTH":                ["int64", "int32"],
    "DAY":                  ["int64", "int32"],
    "DAY_OF_WEEK":          ["int64", "int32"],
    "FLIGHT_NUMBER":        ["int64", "int32"],
    "SCHEDULED_DEPARTURE":  ["int64", "int32"],
    "SCHEDULED_ARRIVAL":    ["int64", "int32"],
    "DISTANCE":             ["int64", "int32"],
    "DIVERTED":             ["int64", "int32"],
    "CANCELLED":            ["int64", "int32"],
    "AIRLINE":              ["object"],
    "TAIL_NUMBER":          ["object"],
    "ORIGIN_AIRPORT":       ["object"],
    "DESTINATION_AIRPORT":  ["object"],
    "CANCELLATION_REASON":  ["object"],
}

# ── Valid ranges ───────────────────────────────────────────────────────────────

COLUMN_RANGES: dict[str, tuple] = {
    "MONTH":                  (1, 12),
    "DAY":                    (1, 31),
    "DAY_OF_WEEK":            (1, 7),
    "SCHEDULED_DEPARTURE":    (0, 2359),
    "SCHEDULED_ARRIVAL":      (0, 2359),
    "DISTANCE":               (1, 10000),
    "DIVERTED":               (0, 1),
    "CANCELLED":              (0, 1),
    "departure_hour":         (0, 23),
    "scheduled_arrival_hour": (0, 23),
    "temperature_c":          (-60, 60),
    "precipitation_mm":       (0, 500),
    "rain_mm":                (0, 500),
    "snowfall_cm":            (0, 200),
    "wind_speed_kmh":         (0, 400),
    "wind_gusts_kmh":         (0, 400),
    "cloud_cover_pct":        (0, 100),
    "relative_humidity_pct":  (0, 100),
    "pressure_msl_hpa":       (850, 1100),
    "dest_temperature_c":          (-60, 60),
    "dest_precipitation_mm":       (0, 500),
    "dest_rain_mm":                (0, 500),
    "dest_snowfall_cm":            (0, 200),
    "dest_wind_speed_kmh":         (0, 400),
    "dest_wind_gusts_kmh":         (0, 400),
    "dest_cloud_cover_pct":        (0, 100),
    "dest_relative_humidity_pct":  (0, 100),
    "dest_pressure_msl_hpa":       (850, 1100),
}


# ── 1. Schema ──────────────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> dict:
    """Verify all expected columns are present with correct dtypes."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    dtype_issues = {}
    for col, allowed in EXPECTED_DTYPES.items():
        if col in df.columns and str(df[col].dtype) not in allowed:
            dtype_issues[col] = {"actual": str(df[col].dtype), "expected": allowed}

    passed = not missing_cols and not dtype_issues
    summary = (
        "Schema valid."
        if passed
        else f"{len(missing_cols)} missing columns, {len(dtype_issues)} dtype mismatches."
    )
    log.info("[schema] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "dtype_issues": dtype_issues,
            "total_rows": len(df),
            "total_columns": len(df.columns),
        },
    }


# ── 2. Completeness ────────────────────────────────────────────────────────────

# Columns where high nulls are expected and not a data quality issue
EXPECTED_NULLS = {
    "CANCELLATION_REASON",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF",
    "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON", "TAXI_IN",
    "ARRIVAL_TIME", "ARRIVAL_DELAY",
}


def check_completeness(df: pd.DataFrame) -> dict:
    """Report missing value counts and percentages per column."""
    total = len(df)
    missing = df.isna().sum()
    missing_pct = (missing / total * 100).round(2)

    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    unexpected_nulls = report[~report.index.isin(EXPECTED_NULLS)]
    passed = len(unexpected_nulls) == 0

    summary = (
        "All missing values are in expected columns."
        if passed
        else f"{len(unexpected_nulls)} columns have unexpected nulls."
    )
    log.info("[completeness] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {
            "missing_report": report,
            "unexpected_nulls": unexpected_nulls,
        },
    }


# ── 3. Duplicates ──────────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for full-row duplicates and flight-level duplicates."""
    full_dupes = df.duplicated().sum()

    # A flight is uniquely identified by airline + flight number + date
    flight_keys = ["AIRLINE", "FLIGHT_NUMBER", "MONTH", "DAY"]
    flight_dupes = df.duplicated(subset=flight_keys).sum()

    passed = full_dupes == 0
    summary = (
        "No full-row duplicates."
        if passed
        else f"{full_dupes} full-row duplicates found."
    )
    if flight_dupes > 0:
        summary += f" {flight_dupes} flight-level duplicates (same airline+flight+date) — expected for codeshares."

    log.info("[duplicates] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {
            "full_row_duplicates": int(full_dupes),
            "flight_level_duplicates": int(flight_dupes),
        },
    }


# ── 4. Validity / Range checks ─────────────────────────────────────────────────

def check_ranges(df: pd.DataFrame) -> dict:
    """Check that numeric columns fall within expected ranges."""
    violations: dict[str, dict] = {}

    for col, (low, high) in COLUMN_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_range = series[(series < low) | (series > high)]
        if not out_of_range.empty:
            violations[col] = {
                "count": len(out_of_range),
                "min": float(out_of_range.min()),
                "max": float(out_of_range.max()),
                "expected": (low, high),
            }

    # Check for numeric (non-IATA) airport codes
    numeric_origins = df["ORIGIN_AIRPORT"].str.isnumeric().sum()
    numeric_dests = df["DESTINATION_AIRPORT"].str.isnumeric().sum()

    passed = not violations
    summary = (
        "All values within expected ranges."
        if passed
        else f"{len(violations)} columns have out-of-range values."
    )
    log.info("[ranges] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {
            "range_violations": violations,
            "numeric_origin_airports": int(numeric_origins),
            "numeric_dest_airports": int(numeric_dests),
        },
    }


# ── 5. Consistency checks ──────────────────────────────────────────────────────

def check_consistency(df: pd.DataFrame) -> dict:
    """Cross-column logic checks."""
    issues: dict[str, dict] = {}

    # Cancelled flights should have no departure time
    cancelled = df[df["CANCELLED"] == 1]
    cancelled_with_departure = cancelled["DEPARTURE_TIME"].notna().sum()
    if cancelled_with_departure > 0:
        issues["cancelled_but_has_departure_time"] = {
            "count": int(cancelled_with_departure),
            "description": "Cancelled flights that still have a DEPARTURE_TIME value.",
        }

    # Non-cancelled, non-diverted flights: ELAPSED_TIME ≈ TAXI_OUT + AIR_TIME + TAXI_IN
    active = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    active = active.dropna(subset=["ELAPSED_TIME", "TAXI_OUT", "AIR_TIME", "TAXI_IN"])
    computed = active["TAXI_OUT"] + active["AIR_TIME"] + active["TAXI_IN"]
    elapsed_mismatch = (abs(active["ELAPSED_TIME"] - computed) > 5).sum()
    if elapsed_mismatch > 0:
        issues["elapsed_time_mismatch"] = {
            "count": int(elapsed_mismatch),
            "description": "Rows where ELAPSED_TIME differs from TAXI_OUT+AIR_TIME+TAXI_IN by more than 5 min.",
        }

    # Diverted and cancelled should not both be 1
    both_flags = ((df["CANCELLED"] == 1) & (df["DIVERTED"] == 1)).sum()
    if both_flags > 0:
        issues["cancelled_and_diverted"] = {
            "count": int(both_flags),
            "description": "Rows flagged as both CANCELLED and DIVERTED.",
        }

    # ARRIVAL_DELAY should be null when flight is cancelled
    cancelled_with_arrival = df[(df["CANCELLED"] == 1) & df["ARRIVAL_DELAY"].notna()]
    if not cancelled_with_arrival.empty:
        issues["cancelled_but_has_arrival_delay"] = {
            "count": len(cancelled_with_arrival),
            "description": "Cancelled flights that still have an ARRIVAL_DELAY value.",
        }

    passed = not issues
    summary = "All consistency checks passed." if passed else f"{len(issues)} consistency issues found."
    log.info("[consistency] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": issues,
    }


# ── 6. Target readiness ────────────────────────────────────────────────────────

TARGET_BINS = [-float("inf"), 15, 45, float("inf")]
TARGET_LABELS = ["on_time", "minor_delay", "major_delay"]


def check_target(df: pd.DataFrame) -> dict:
    """
    Assess readiness of ARRIVAL_DELAY as the classification target.
    Target definition:
        on_time      : ARRIVAL_DELAY <= 15 min
        minor_delay  : 15 < ARRIVAL_DELAY <= 45 min
        major_delay  : ARRIVAL_DELAY > 45 min
        cancelled    : CANCELLED == 1
    """
    total = len(df)
    cancelled_count = int((df["CANCELLED"] == 1).sum())
    arrival_null = int(df["ARRIVAL_DELAY"].isna().sum())

    # Non-cancelled flights with a valid ARRIVAL_DELAY
    active = df[(df["CANCELLED"] == 0) & df["ARRIVAL_DELAY"].notna()].copy()
    active["target"] = pd.cut(
        active["ARRIVAL_DELAY"],
        bins=TARGET_BINS,
        labels=TARGET_LABELS,
    )

    class_counts = active["target"].value_counts().to_dict()
    class_counts["cancelled"] = cancelled_count

    class_pct = {k: round(v / total * 100, 2) for k, v in class_counts.items()}

    # Rows that can't be labelled (non-cancelled but missing ARRIVAL_DELAY)
    unlabellable = int(
        ((df["CANCELLED"] == 0) & df["ARRIVAL_DELAY"].isna()).sum()
    )

    passed = unlabellable == 0
    summary = (
        "Target variable is fully labellable."
        if passed
        else f"{unlabellable} non-cancelled rows have no ARRIVAL_DELAY — cannot be labelled."
    )
    log.info("[target] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {
            "total_rows": total,
            "cancelled_count": cancelled_count,
            "arrival_delay_nulls": arrival_null,
            "unlabellable_rows": unlabellable,
            "class_counts": class_counts,
            "class_pct": class_pct,
            "arrival_delay_stats": df["ARRIVAL_DELAY"].describe().to_dict(),
        },
    }


# ── 7. Outlier detection ───────────────────────────────────────────────────────

OUTLIER_COLS = [
    "DISTANCE", "SCHEDULED_TIME",
    "ARRIVAL_DELAY", "DEPARTURE_DELAY",
    "TAXI_OUT", "TAXI_IN", "AIR_TIME",
    "temperature_c", "precipitation_mm", "wind_speed_kmh", "wind_gusts_kmh",
    "dest_temperature_c", "dest_precipitation_mm",
    "dest_wind_speed_kmh", "dest_wind_gusts_kmh",
]


def check_outliers(df: pd.DataFrame) -> dict:
    """
    Detect statistical outliers using the IQR method (1.5 × IQR fence).
    Only flags outliers in columns relevant to the model — not operational
    columns like WHEELS_OFF that are expected to have wide distributions.
    """
    results: dict[str, dict] = {}

    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if not outliers.empty:
            results[col] = {
                "count": len(outliers),
                "pct": round(len(outliers) / len(series) * 100, 2),
                "lower_fence": round(lower, 2),
                "upper_fence": round(upper, 2),
                "min_outlier": round(float(outliers.min()), 2),
                "max_outlier": round(float(outliers.max()), 2),
            }

    passed = not results
    summary = (
        "No statistical outliers detected."
        if passed
        else f"Outliers found in {len(results)} columns."
    )
    log.info("[outliers] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": results,
    }


# ── 8. Referential integrity ───────────────────────────────────────────────────

def check_referential_integrity(
    df: pd.DataFrame,
    airlines_path: Path,
    airports_path: Path,
) -> dict:
    """
    Check that AIRLINE codes and IATA airport codes exist in the reference files.
    Numeric airport codes are excluded — they are a known acquisition gap,
    not a referential integrity failure.
    """
    issues: dict[str, dict] = {}

    # Airlines
    known_airlines = set(pd.read_csv(airlines_path)["IATA_CODE"].dropna())
    unknown_airlines = set(df["AIRLINE"].dropna().unique()) - known_airlines
    if unknown_airlines:
        issues["unknown_airlines"] = {
            "count": int(df["AIRLINE"].isin(unknown_airlines).sum()),
            "values": sorted(unknown_airlines),
            "description": "AIRLINE codes not found in airlines.csv.",
        }

    # Airports — only check IATA (non-numeric) codes
    known_airports = set(pd.read_csv(airports_path)["IATA_CODE"].dropna())
    iata_origins = df.loc[~df["ORIGIN_AIRPORT"].str.isnumeric(), "ORIGIN_AIRPORT"].dropna()
    iata_dests = df.loc[~df["DESTINATION_AIRPORT"].str.isnumeric(), "DESTINATION_AIRPORT"].dropna()

    unknown_origins = set(iata_origins.unique()) - known_airports
    unknown_dests = set(iata_dests.unique()) - known_airports

    if unknown_origins:
        issues["unknown_origin_airports"] = {
            "count": int(iata_origins.isin(unknown_origins).sum()),
            "values": sorted(unknown_origins),
            "description": "ORIGIN_AIRPORT IATA codes not found in airports.csv.",
        }
    if unknown_dests:
        issues["unknown_destination_airports"] = {
            "count": int(iata_dests.isin(unknown_dests).sum()),
            "values": sorted(unknown_dests),
            "description": "DESTINATION_AIRPORT IATA codes not found in airports.csv.",
        }

    passed = not issues
    summary = "All codes match reference files." if passed else f"{len(issues)} referential integrity issue(s) found."
    log.info("[referential_integrity] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": issues,
    }


# ── 9. Format / pattern validation ────────────────────────────────────────────

def check_formats(df: pd.DataFrame) -> dict:
    """
    Validate format of coded fields:
    - IATA airport/airline codes: 2–3 uppercase letters
    - HHMM time fields: minutes part must be 00–59
    - YEAR: all rows must be 2015
    - CANCELLATION_REASON: must be one of A, B, C, D (or null)
    """
    issues: dict[str, dict] = {}

    # IATA code format (non-numeric only)
    for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        iata = df.loc[~df[col].str.isnumeric(), col].dropna()
        bad = iata[~iata.str.match(r"^[A-Z]{3}$")]
        if not bad.empty:
            issues[f"{col}_bad_iata_format"] = {
                "count": len(bad),
                "examples": bad.unique()[:5].tolist(),
                "description": f"{col} values that are not 3 uppercase letters.",
            }

    # HHMM minute validity (minutes digit = last two digits, must be 00-59)
    for col in ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]:
        minutes = df[col].dropna() % 100
        bad_minutes = df.loc[(df[col].notna()) & (minutes >= 60), col]
        if not bad_minutes.empty:
            issues[f"{col}_invalid_minutes"] = {
                "count": len(bad_minutes),
                "examples": bad_minutes.unique()[:5].tolist(),
                "description": f"{col} values with minutes ≥ 60.",
            }

    # YEAR must be 2015
    wrong_year = df[df["YEAR"] != 2015]
    if not wrong_year.empty:
        issues["wrong_year"] = {
            "count": len(wrong_year),
            "values": wrong_year["YEAR"].unique().tolist(),
            "description": "Rows with YEAR ≠ 2015.",
        }

    # CANCELLATION_REASON must be A, B, C, D or null
    valid_reasons = {"A", "B", "C", "D"}
    actual_reasons = df["CANCELLATION_REASON"].dropna()
    bad_reasons = actual_reasons[~actual_reasons.isin(valid_reasons)]
    if not bad_reasons.empty:
        issues["invalid_cancellation_reason"] = {
            "count": len(bad_reasons),
            "values": bad_reasons.unique().tolist(),
            "description": "CANCELLATION_REASON values outside {A, B, C, D}.",
        }

    passed = not issues
    summary = "All format checks passed." if passed else f"{len(issues)} format issue(s) found."
    log.info("[formats] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": issues,
    }


# ── 10. Temporal validity ──────────────────────────────────────────────────────

def check_temporal(df: pd.DataFrame) -> dict:
    """
    Validate that (YEAR, MONTH, DAY) combinations form real calendar dates.
    E.g. February 30 or April 31 are invalid.
    """
    import calendar

    invalid_dates = []
    for _, row in df[["MONTH", "DAY"]].drop_duplicates().iterrows():
        month, day = int(row["MONTH"]), int(row["DAY"])
        max_day = calendar.monthrange(2015, month)[1]
        if day > max_day:
            count = int(((df["MONTH"] == month) & (df["DAY"] == day)).sum())
            invalid_dates.append({"month": month, "day": day, "max_valid_day": max_day, "count": count})

    passed = not invalid_dates
    summary = (
        "All dates are valid calendar dates."
        if passed
        else f"{len(invalid_dates)} invalid (month, day) combination(s) found."
    )
    log.info("[temporal] %s", summary)
    return {
        "passed": passed,
        "summary": summary,
        "details": {"invalid_dates": invalid_dates},
    }


# ── 11. Cardinality ────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "CANCELLATION_REASON", "DAY_OF_WEEK", "MONTH",
]


def check_cardinality(df: pd.DataFrame) -> dict:
    """Report unique value counts for categorical columns."""
    report = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        n_unique = int(df[col].nunique(dropna=True))
        top_values = df[col].value_counts(dropna=True).head(5).to_dict()
        report[col] = {"unique_count": n_unique, "top_5": top_values}

    log.info("[cardinality] Checked %d categorical columns.", len(report))
    return {
        "passed": True,
        "summary": f"Cardinality report for {len(report)} categorical columns.",
        "details": report,
    }


# ── 12. Statistical summary ────────────────────────────────────────────────────

SUMMARY_COLS = [
    "DISTANCE", "SCHEDULED_TIME", "ARRIVAL_DELAY", "DEPARTURE_DELAY",
    "TAXI_OUT", "TAXI_IN", "AIR_TIME",
    "temperature_c", "precipitation_mm", "wind_speed_kmh",
    "wind_gusts_kmh", "cloud_cover_pct", "relative_humidity_pct", "pressure_msl_hpa",
    "dest_temperature_c", "dest_wind_speed_kmh",
]


def check_statistics(df: pd.DataFrame) -> dict:
    """Descriptive statistics for key numeric columns."""
    available = [c for c in SUMMARY_COLS if c in df.columns]
    stats = df[available].describe().round(2)
    log.info("[statistics] Descriptive stats computed for %d columns.", len(available))
    return {
        "passed": True,
        "summary": f"Descriptive statistics for {len(available)} numeric columns.",
        "details": {"stats": stats},
    }


# ── Runner / CLI ──────────────────────────────────────────────────────────────

def run_all(
    df: pd.DataFrame,
    airlines_path: Path | None = None,
    airports_path: Path | None = None,
) -> dict[str, dict]:
    """Run all validation checks and return a combined results dict."""
    checks = {
        "schema":        check_schema(df),
        "completeness":  check_completeness(df),
        "duplicates":    check_duplicates(df),
        "ranges":        check_ranges(df),
        "consistency":   check_consistency(df),
        "outliers":      check_outliers(df),
        "formats":       check_formats(df),
        "temporal":      check_temporal(df),
        "cardinality":   check_cardinality(df),
        "statistics":    check_statistics(df),
        "target":        check_target(df),
    }
    if airlines_path and airports_path:
        checks["referential_integrity"] = check_referential_integrity(df, airlines_path, airports_path)
    return checks


def main() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

    processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    external_dir  = Path(os.getenv("EXTERNAL_DATA_DIR",  "data/external"))

    data_path     = processed_dir / "flights_weather.csv"
    airlines_path = external_dir  / "airlines.csv"
    airports_path = external_dir  / "airports.csv"

    log.info("Loading %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    log.info("Loaded %d rows, %d columns", *df.shape)

    results = run_all(df, airlines_path=airlines_path, airports_path=airports_path)

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    passed_all = all(r["passed"] for r in results.values())
    for check, result in results.items():
        icon = "✓" if result["passed"] else "✗"
        print(f"  [{icon}] {check:<25}  {result['summary']}")
    print("=" * 60)
    print(f"  Overall: {'PASSED' if passed_all else 'ISSUES FOUND'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
