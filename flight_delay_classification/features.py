"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Feature pipeline
"""

from pathlib import Path

from imblearn.over_sampling import SMOTE
from loguru import logger
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.feature_selection import mutual_info_classif
import typer

from flight_delay_classification.config import PROCESSED_DATA_DIR

app = (
    typer.Typer()
)  # NOTE: typer is a tool that comes with the templates, it is used for building CLIs

TARGET_COLUMN = "DELAY_CATEGORY"
RANDOM_STATE = 42
NON_PREDICTIVE_COLUMNS = [  # NOTE: if we used this features they will leak the target
    "DIVERTED",  # Diverted is if a plane changed it's direction
    "CANCELLATION_REASON",  # Cancellation reason is only filled if it was cancelled
]

HIGH_CARDINALITY_ROUTE_COLUMNS = [
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "AIRLINE",
]

HISTORICAL_ENCODING_COLUMNS = {
    "AIRLINE": "airline",
    "ORIGIN_AIRPORT": "origin_airport",
    "DESTINATION_AIRPORT": "destination_airport",
    "ROUTE": "route",
}
HISTORICAL_SMOOTHING = 25.0
INTERACTION_HISTORICAL_ENCODING_COLUMNS = {
    "AIRLINE_DEPARTURE_BANK": "airline_departure_bank",
    "ORIGIN_AIRPORT_DEPARTURE_BANK": "origin_departure_bank",
    "ORIGIN_AIRPORT_DAY_OF_WEEK": "origin_day_of_week",
    "AIRLINE_ROUTE": "airline_route",
    "ROUTE_DEPARTURE_BANK": "route_departure_bank",
    "ROUTE_DAY_OF_WEEK": "route_day_of_week",
}
PEAK_TRAVEL_BANKS = {
    "is_morning_peak_bank": (6, 9),
    "is_evening_peak_bank": (16, 19),
}
DEPARTURE_BANK_BOUNDS = [0, 6, 10, 14, 18, 22, 24]
DEPARTURE_BANK_LABELS = [
    "overnight",
    "morning",
    "midday",
    "afternoon",
    "evening",
    "late_night",
]
WEATHER_INTENSITY_THRESHOLDS = {
    "precipitation_mm": 2.0,
    "rain_mm": 2.0,
    "snowfall_cm": 0.0,
    "wind_speed_kmh": 25.0,
    "wind_gusts_kmh": 40.0,
}
DEFAULT_FEATURE_SELECTION_METHOD = "none"
DEFAULT_MIN_MUTUAL_INFO = 0.001
HIST_GRADIENT_MODEL_MODES = {
    "hist_gradient_boosting",
    "hierarchical_hist_gradient_boosting",
}


def _build_route_column(features: pd.DataFrame) -> pd.Series:
    return features["ORIGIN_AIRPORT"] + "_" + features["DESTINATION_AIRPORT"]


def _build_historical_targets(y_train: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "historical_delay_rate": (y_train != "on_time").astype(float),
            "historical_on_time_rate": (y_train == "on_time").astype(float),
            "historical_minor_delay_rate": (y_train == "minor_delay").astype(float),
            "historical_severe_rate": (y_train == "major_delay").astype(float),
            "historical_major_delay_rate": (y_train == "major_delay").astype(float),
            "historical_cancelled_rate": (y_train == "cancelled").astype(float),
        }
    )


def _add_frequency_encoded_columns(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in columns:
        # NOTE: Frequency maps are learned on train only to avoid leakage.
        frequency_map = train_features[column].value_counts(normalize=True)
        train_features[f"{column}_freq"] = train_features[column].map(frequency_map)
        test_features[f"{column}_freq"] = (
            test_features[column].map(frequency_map).fillna(0.0)
        )

    return train_features, test_features


def _add_group_count_feature(
    features: pd.DataFrame,
    group_columns: list[str | pd.Series],
    anchor_column: str,
    feature_name: str,
) -> None:
    features[feature_name] = features.groupby(group_columns, dropna=False)[
        anchor_column
    ].transform("size")


def _add_group_ratio_feature(
    features: pd.DataFrame,
    numerator_column: str,
    group_columns: list[str],
    feature_name: str,
) -> None:
    features[feature_name] = features[numerator_column] / features.groupby(
        group_columns,
        dropna=False,
    )[numerator_column].transform("mean")


def _compute_weather_intensity(features: pd.DataFrame, prefix: str = "") -> pd.Series:
    precipitation_column = f"{prefix}precipitation_mm"
    rain_column = f"{prefix}rain_mm"
    snowfall_column = f"{prefix}snowfall_cm"
    wind_speed_column = f"{prefix}wind_speed_kmh"
    wind_gusts_column = f"{prefix}wind_gusts_kmh"
    temperature_column = f"{prefix}temperature_c"

    weather_intensity = (
        (
            features[precipitation_column]
            >= WEATHER_INTENSITY_THRESHOLDS["precipitation_mm"]
        )
        | (features[rain_column] >= WEATHER_INTENSITY_THRESHOLDS["rain_mm"])
    ).astype(float)
    weather_intensity += (
        features[snowfall_column] > WEATHER_INTENSITY_THRESHOLDS["snowfall_cm"]
    ).astype(float)
    weather_intensity += (
        (features[wind_speed_column] >= WEATHER_INTENSITY_THRESHOLDS["wind_speed_kmh"])
        | (
            features[wind_gusts_column]
            >= WEATHER_INTENSITY_THRESHOLDS["wind_gusts_kmh"]
        )
    ).astype(float)
    weather_intensity += (features[temperature_column] <= 0).astype(float)

    return weather_intensity


def _drop_feature_engineering_categoricals(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_columns = [
        column
        for column in [
            *HISTORICAL_ENCODING_COLUMNS,
            *INTERACTION_HISTORICAL_ENCODING_COLUMNS,
            "departure_hour_bucket",
        ]
        if column in train_features.columns and column in test_features.columns
    ]
    if not categorical_columns:
        return train_features, test_features

    return (
        train_features.drop(columns=categorical_columns),
        test_features.drop(columns=categorical_columns),
    )


def _prepare_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic features that do not depend on training labels."""
    features = add_temporal_features(features)
    features = add_congestion_weather_interaction_features(features)
    features["ROUTE"] = _build_route_column(features)
    return add_dense_interaction_keys(features)


def _apply_train_only_feature_enrichments(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    feature_selection_method: str,
    min_mutual_info: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply train-fitted encodings and optional selection to both splits."""
    route_columns = [
        column
        for column in HIGH_CARDINALITY_ROUTE_COLUMNS
        if column in train_features.columns and column in test_features.columns
    ]
    train_features, test_features = _add_frequency_encoded_columns(
        train_features,
        test_features,
        route_columns,
    )

    train_features, test_features = add_smoothed_historical_rate_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
    )
    train_features, test_features = add_smoothed_historical_rate_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        encoding_columns=INTERACTION_HISTORICAL_ENCODING_COLUMNS,
    )
    train_features, test_features = _drop_feature_engineering_categoricals(
        train_features,
        test_features,
    )

    train_features, test_features, _ = select_informative_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train,
        method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )
    return train_features, test_features


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_indices = []

    for _, group in df.groupby(
        target_column, dropna=False
    ):  # NOTE: this splits by class
        if len(group) <= 1:
            continue

        test_rows = int(round(len(group) * test_size))
        test_rows = max(1, min(test_rows, len(group) - 1))
        # NOTE: group.sample is random sampling but reproducable because of the random_statae/seed
        sampled_group = group.sample(n=test_rows, random_state=random_state)
        test_indices.extend(sampled_group.index.tolist())

    train_df = df.drop(index=test_indices).sample(frac=1, random_state=random_state)
    test_df = df.loc[test_indices].sample(frac=1, random_state=random_state)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    logger.info(
        "SMOTE applied: {} -> {} samples",
        len(X_train),
        len(X_resampled),
    )
    return X_resampled, y_resampled


def add_smoothed_historical_rate_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    encoding_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add smoothed train-only target encodings for the supplied categorical keys."""
    # NOTE: Dense target-rate features avoid one-hot expansion on very wide keys.
    train_targets = _build_historical_targets(y_train)
    global_rates = train_targets.mean().to_dict()
    resolved_encoding_columns = encoding_columns or HISTORICAL_ENCODING_COLUMNS

    for column, prefix in resolved_encoding_columns.items():
        grouped = pd.concat([train_features[[column]], train_targets], axis=1).groupby(
            column,
            dropna=False,
        )
        group_size = grouped.size()
        group_sums = grouped.sum()

        for rate_name, global_rate in global_rates.items():
            feature_name = f"{prefix}_{rate_name}"
            train_group_size = train_features[column].map(group_size)
            train_group_sum = train_features[column].map(group_sums[rate_name])
            leave_one_out_numerator = (
                train_group_sum - train_targets[rate_name]
            ) + HISTORICAL_SMOOTHING * global_rate
            leave_one_out_denominator = (train_group_size - 1) + HISTORICAL_SMOOTHING
            train_features[feature_name] = (
                leave_one_out_numerator / leave_one_out_denominator
            ).fillna(global_rate)
            test_rates = (
                group_sums[rate_name] + HISTORICAL_SMOOTHING * global_rate
            ) / (group_size + HISTORICAL_SMOOTHING)
            test_features[feature_name] = (
                test_features[column].map(test_rates).fillna(global_rate)
            )

    return train_features, test_features


def add_temporal_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and clock-based features from scheduled departure metadata."""
    departure_hour = features["SCHEDULED_DEPARTURE"] // 100
    arrival_hour = features["SCHEDULED_ARRIVAL"] // 100
    flight_dates = pd.to_datetime(
        features[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    )
    holiday_dates = USFederalHolidayCalendar().holidays(
        start=flight_dates.min(),
        end=flight_dates.max(),
    )

    features["is_weekend"] = features["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    features["is_holiday"] = flight_dates.isin(holiday_dates).astype(int)

    # NOTE: cyclical encoding keeps midnight close to 23:00 and January close to December.
    departure_angle = 2 * np.pi * departure_hour / 24
    arrival_angle = 2 * np.pi * arrival_hour / 24
    month_angle = 2 * np.pi * (features["MONTH"] - 1) / 12

    features["scheduled_departure_hour_sin"] = np.sin(departure_angle)
    features["scheduled_departure_hour_cos"] = np.cos(departure_angle)
    features["scheduled_arrival_hour_sin"] = np.sin(arrival_angle)
    features["scheduled_arrival_hour_cos"] = np.cos(arrival_angle)
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)
    features["departure_hour_bucket"] = pd.cut(
        departure_hour,
        bins=DEPARTURE_BANK_BOUNDS,
        labels=DEPARTURE_BANK_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    features["departure_hour_bucket_code"] = features["departure_hour_bucket"].map(
        {label: index for index, label in enumerate(DEPARTURE_BANK_LABELS)}
    )

    for column_name, (start_hour, end_hour) in PEAK_TRAVEL_BANKS.items():
        features[column_name] = departure_hour.between(start_hour, end_hour).astype(int)

    return features


def add_congestion_weather_interaction_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine airport congestion signals with origin and destination weather intensity."""
    departure_hour = (features["SCHEDULED_DEPARTURE"] // 100).rename("departure_hour")
    arrival_hour = (features["SCHEDULED_ARRIVAL"] // 100).rename("arrival_hour")

    _add_group_count_feature(
        features,
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK", departure_hour],
        "ORIGIN_AIRPORT",
        "origin_hourly_departure_count",
    )
    _add_group_count_feature(
        features,
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK", arrival_hour],
        "DESTINATION_AIRPORT",
        "destination_hourly_arrival_count",
    )
    _add_group_count_feature(
        features,
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK", "departure_hour_bucket"],
        "ORIGIN_AIRPORT",
        "origin_departure_bank_count",
    )
    _add_group_count_feature(
        features,
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK", "departure_hour_bucket"],
        "DESTINATION_AIRPORT",
        "destination_departure_bank_count",
    )

    _add_group_ratio_feature(
        features,
        "origin_hourly_departure_count",
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK"],
        "origin_congestion_ratio",
    )
    _add_group_ratio_feature(
        features,
        "destination_hourly_arrival_count",
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK"],
        "destination_congestion_ratio",
    )
    _add_group_ratio_feature(
        features,
        "origin_departure_bank_count",
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK"],
        "origin_departure_bank_ratio",
    )
    _add_group_ratio_feature(
        features,
        "destination_departure_bank_count",
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK"],
        "destination_departure_bank_ratio",
    )

    features["origin_weather_intensity"] = _compute_weather_intensity(features)
    features["destination_weather_intensity"] = _compute_weather_intensity(
        features,
        prefix="dest_",
    )

    # NOTE: These interaction terms emphasize busy airports during disruptive weather.
    features["origin_congestion_weather_score"] = (
        features["origin_congestion_ratio"] * features["origin_weather_intensity"]
    )
    features["destination_congestion_weather_score"] = (
        features["destination_congestion_ratio"]
        * features["destination_weather_intensity"]
    )
    features["route_congestion_weather_score"] = (
        features["origin_congestion_ratio"] + features["destination_congestion_ratio"]
    ) * (
        features["origin_weather_intensity"] + features["destination_weather_intensity"]
    )
    features["origin_departure_bank_weather_score"] = (
        features["origin_departure_bank_ratio"] * features["origin_weather_intensity"]
    )
    features["destination_departure_bank_weather_score"] = (
        features["destination_departure_bank_ratio"]
        * features["destination_weather_intensity"]
    )

    return features


def add_dense_interaction_keys(features: pd.DataFrame) -> pd.DataFrame:
    features["AIRLINE_DEPARTURE_BANK"] = (
        features["AIRLINE"] + "_" + features["departure_hour_bucket"]
    )
    features["ORIGIN_AIRPORT_DEPARTURE_BANK"] = (
        features["ORIGIN_AIRPORT"] + "_" + features["departure_hour_bucket"]
    )
    features["ORIGIN_AIRPORT_DAY_OF_WEEK"] = (
        features["ORIGIN_AIRPORT"] + "_" + features["DAY_OF_WEEK"].astype(str)
    )
    features["AIRLINE_ROUTE"] = features["AIRLINE"] + "_" + features["ROUTE"]
    features["ROUTE_DEPARTURE_BANK"] = (
        features["ROUTE"] + "_" + features["departure_hour_bucket"]
    )
    features["ROUTE_DAY_OF_WEEK"] = (
        features["ROUTE"] + "_" + features["DAY_OF_WEEK"].astype(str)
    )
    return features


def select_informative_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Optionally filter the feature matrix with a simple mutual-information rule."""
    if method == "none":
        return train_features, test_features, train_features.columns.tolist()

    if method != "mutual_info":
        raise ValueError(
            f"Unsupported feature_selection_method '{method}'. Use 'none' or 'mutual_info'."
        )

    varying_columns = [
        column
        for column in train_features.columns
        if train_features[column].nunique() > 1
    ]
    if not varying_columns:
        fallback_column = train_features.columns[0]
        return (
            train_features[[fallback_column]].copy(),
            test_features[[fallback_column]].copy(),
            [fallback_column],
        )

    filtered_train_features = train_features[varying_columns]
    filtered_test_features = test_features[varying_columns]
    mi_scores = mutual_info_classif(
        filtered_train_features,
        y_train,
        random_state=RANDOM_STATE,
    )
    selected_columns = [
        column
        for column, score in zip(varying_columns, mi_scores, strict=False)
        if score >= min_mutual_info
    ]
    if not selected_columns:
        fallback_index = int(np.argmax(mi_scores))
        selected_columns = [varying_columns[fallback_index]]

    logger.info(
        "Feature selection kept {} of {} columns using {} (min_mutual_info={})",
        len(selected_columns),
        len(train_features.columns),
        method,
        min_mutual_info,
    )
    return (
        filtered_train_features[selected_columns].copy(),
        filtered_test_features[selected_columns].copy(),
        selected_columns,
    )


def adapt_features_for_model_mode(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    model_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop known-problematic encodings for model families that overfit them."""
    if model_mode not in HIST_GRADIENT_MODEL_MODES:
        return train_features, test_features, []

    dropped_columns = [
        column
        for column in train_features.columns
        if "_historical_" in column and column in test_features.columns
    ]
    if not dropped_columns:
        return train_features, test_features, []

    logger.info(
        "Dropping {} historical target-encoding features for {} to avoid histogram boosting collapse",
        len(dropped_columns),
        model_mode,
    )
    return (
        train_features.drop(columns=dropped_columns),
        test_features.drop(columns=dropped_columns),
        dropped_columns,
    )


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate labels and add deterministic plus train-only encoded features."""

    y_train = train_df[[target_column]].copy()
    y_test = test_df[[target_column]].copy()

    train_features = train_df.drop(columns=[target_column, *NON_PREDICTIVE_COLUMNS])
    test_features = test_df.drop(columns=[target_column, *NON_PREDICTIVE_COLUMNS])
    train_features = _prepare_model_features(train_features)
    test_features = _prepare_model_features(test_features)
    train_features, test_features = _apply_train_only_feature_enrichments(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train[target_column],
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )

    return train_features, y_train, test_features, y_test


def save_feature_artifacts(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_features: pd.DataFrame,
    test_labels: pd.DataFrame,
    train_features_path: Path,
    train_labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
) -> None:
    output_paths = [
        train_features_path,
        train_labels_path,
        test_features_path,
        test_labels_path,
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(train_features_path, index=False)
    train_labels.to_csv(train_labels_path, index=False)
    test_features.to_csv(test_features_path, index=False)
    test_labels.to_csv(test_labels_path, index=False)

    logger.info(
        "Saved train features to {} with shape {}",
        train_features_path,
        train_features.shape,
    )
    logger.info(
        "Saved train labels to {} with shape {}", train_labels_path, train_labels.shape
    )
    logger.info(
        "Saved test features to {} with shape {}",
        test_features_path,
        test_features.shape,
    )
    logger.info(
        "Saved test labels to {} with shape {}", test_labels_path, test_labels.shape
    )


def prepare_feature_artifacts(
    input_path: Path,
    train_features_path: Path,
    train_labels_path: Path,
    test_features_path: Path,
    test_labels_path: Path,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
) -> dict[str, int | float]:
    """Load, split, encode, and save the minimal baseline artifacts"""

    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded cleaned dataset from {} with shape {}", input_path, df.shape)

    train_df, test_df = split_dataset(
        df=df,
        target_column=TARGET_COLUMN,
        test_size=test_size,
        random_state=random_state,
    )

    train_features, train_labels, test_features, test_labels = build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        target_column=TARGET_COLUMN,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )

    save_feature_artifacts(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
    )

    summary = {
        "input_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_columns": len(train_features.columns),
    }

    logger.info("Feature preparation complete: {}", summary)

    return summary


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "flights_cleaned.csv",
    train_features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    train_labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    feature_selection_method: str = DEFAULT_FEATURE_SELECTION_METHOD,
    min_mutual_info: float = DEFAULT_MIN_MUTUAL_INFO,
) -> None:
    """CLI entrypoint for generating the baseline modeling artifacts."""
    prepare_feature_artifacts(
        input_path=input_path,
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        test_size=test_size,
        random_state=random_state,
        feature_selection_method=feature_selection_method,
        min_mutual_info=min_mutual_info,
    )


if __name__ == "__main__":
    app()
