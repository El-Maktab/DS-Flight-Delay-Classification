"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Feature pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
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
    """Add train-only smoothed historical rate features for the main route keys.

    The function builds historical encodings from the training labels only
    then applies the learned rates to both train and test features
    it computes three binary target rates:
    """
    # NOTE: those binary targets keep the encoding dense without one-hot expanding the categories.
    train_targets = pd.DataFrame(
        {
            "historical_delay_rate": (y_train != "on_time").astype(float),
            "historical_on_time_rate": (y_train == "on_time").astype(float),
            "historical_minor_delay_rate": (y_train == "minor_delay").astype(float),
            "historical_severe_rate": (y_train == "major_delay").astype(float),
            "historical_major_delay_rate": (y_train == "major_delay").astype(float),
            "historical_cancelled_rate": (y_train == "cancelled").astype(float),
        }
    )
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
            smoothed_rates = (
                group_size * (group_sums[rate_name] / group_size)
                + HISTORICAL_SMOOTHING * global_rate
            ) / (group_size + HISTORICAL_SMOOTHING)
            train_group_size = train_features[column].map(group_size)
            train_group_sum = train_features[column].map(group_sums[rate_name])
            leave_one_out_numerator = (
                train_group_sum - train_targets[rate_name]
            ) + HISTORICAL_SMOOTHING * global_rate
            leave_one_out_denominator = (train_group_size - 1) + HISTORICAL_SMOOTHING
            smoothed_rates = (
                leave_one_out_numerator / leave_one_out_denominator
            ).fillna(global_rate)
            train_features[feature_name] = smoothed_rates
            test_features[feature_name] = (
                test_features[column]
                .map(
                    (group_sums[rate_name] + HISTORICAL_SMOOTHING * global_rate)
                    / (group_size + HISTORICAL_SMOOTHING)
                )
                .fillna(global_rate)
            )

    return train_features, test_features


def add_temporal_features(features: pd.DataFrame) -> pd.DataFrame:
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
    """builds simple airport traffic counts from scheduled departure
    and arrival hour buckets then combines those congestion signals with small
    origin and destination weather intensity scores so busy bad-weather
    stand out more clearly.
    """
    departure_hour = (features["SCHEDULED_DEPARTURE"] // 100).rename("departure_hour")
    arrival_hour = (features["SCHEDULED_ARRIVAL"] // 100).rename("arrival_hour")

    features["origin_hourly_departure_count"] = features.groupby(
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK", departure_hour],
        dropna=False,
    )["ORIGIN_AIRPORT"].transform("size")
    features["destination_hourly_arrival_count"] = features.groupby(
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK", arrival_hour],
        dropna=False,
    )["DESTINATION_AIRPORT"].transform("size")
    features["origin_departure_bank_count"] = features.groupby(
        ["ORIGIN_AIRPORT", "DAY_OF_WEEK", "departure_hour_bucket"],
        dropna=False,
    )["ORIGIN_AIRPORT"].transform("size")
    features["destination_departure_bank_count"] = features.groupby(
        ["DESTINATION_AIRPORT", "DAY_OF_WEEK", "departure_hour_bucket"],
        dropna=False,
    )["DESTINATION_AIRPORT"].transform("size")

    features["origin_congestion_ratio"] = features[
        "origin_hourly_departure_count"
    ] / features.groupby(["ORIGIN_AIRPORT", "DAY_OF_WEEK"], dropna=False)[
        "origin_hourly_departure_count"
    ].transform(
        "mean"
    )
    features["destination_congestion_ratio"] = features[
        "destination_hourly_arrival_count"
    ] / features.groupby(["DESTINATION_AIRPORT", "DAY_OF_WEEK"], dropna=False)[
        "destination_hourly_arrival_count"
    ].transform(
        "mean"
    )
    features["origin_departure_bank_ratio"] = features[
        "origin_departure_bank_count"
    ] / features.groupby(["ORIGIN_AIRPORT", "DAY_OF_WEEK"], dropna=False)[
        "origin_departure_bank_count"
    ].transform(
        "mean"
    )
    features["destination_departure_bank_ratio"] = features[
        "destination_departure_bank_count"
    ] / features.groupby(["DESTINATION_AIRPORT", "DAY_OF_WEEK"], dropna=False)[
        "destination_departure_bank_count"
    ].transform(
        "mean"
    )

    features["origin_weather_intensity"] = (
        (
            features["precipitation_mm"]
            >= WEATHER_INTENSITY_THRESHOLDS["precipitation_mm"]
        )
        | (features["rain_mm"] >= WEATHER_INTENSITY_THRESHOLDS["rain_mm"])
    ).astype(float)
    features["origin_weather_intensity"] += (
        features["snowfall_cm"] > WEATHER_INTENSITY_THRESHOLDS["snowfall_cm"]
    ).astype(float)
    features["origin_weather_intensity"] += (
        (features["wind_speed_kmh"] >= WEATHER_INTENSITY_THRESHOLDS["wind_speed_kmh"])
        | (features["wind_gusts_kmh"] >= WEATHER_INTENSITY_THRESHOLDS["wind_gusts_kmh"])
    ).astype(float)
    features["origin_weather_intensity"] += (features["temperature_c"] <= 0).astype(
        float
    )

    features["destination_weather_intensity"] = (
        (
            features["dest_precipitation_mm"]
            >= WEATHER_INTENSITY_THRESHOLDS["precipitation_mm"]
        )
        | (features["dest_rain_mm"] >= WEATHER_INTENSITY_THRESHOLDS["rain_mm"])
    ).astype(float)
    features["destination_weather_intensity"] += (
        features["dest_snowfall_cm"] > WEATHER_INTENSITY_THRESHOLDS["snowfall_cm"]
    ).astype(float)
    features["destination_weather_intensity"] += (
        (
            features["dest_wind_speed_kmh"]
            >= WEATHER_INTENSITY_THRESHOLDS["wind_speed_kmh"]
        )
        | (
            features["dest_wind_gusts_kmh"]
            >= WEATHER_INTENSITY_THRESHOLDS["wind_gusts_kmh"]
        )
    ).astype(float)
    features["destination_weather_intensity"] += (
        features["dest_temperature_c"] <= 0
    ).astype(float)

    # NOTE: we
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
    train_features = add_temporal_features(train_features)
    test_features = add_temporal_features(test_features)
    train_features = add_congestion_weather_interaction_features(train_features)
    test_features = add_congestion_weather_interaction_features(test_features)
    train_features["ROUTE"] = (
        train_features["ORIGIN_AIRPORT"] + "_" + train_features["DESTINATION_AIRPORT"]
    )
    test_features["ROUTE"] = (
        test_features["ORIGIN_AIRPORT"] + "_" + test_features["DESTINATION_AIRPORT"]
    )
    train_features = add_dense_interaction_keys(train_features)
    test_features = add_dense_interaction_keys(test_features)

    route_columns = [
        col
        for col in HIGH_CARDINALITY_ROUTE_COLUMNS
        if col in train_features.columns and col in test_features.columns
    ]

    for column in route_columns:
        # NOTE: we calculate the frequency map on the training data to avoid leakage
        frequency_map = train_features[column].value_counts(normalize=True)
        train_features[f"{column}_freq"] = train_features[column].map(frequency_map)
        test_features[f"{column}_freq"] = (
            test_features[column].map(frequency_map).fillna(0.0)
        )

    train_features, test_features = add_smoothed_historical_rate_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train[target_column],
    )
    train_features, test_features = add_smoothed_historical_rate_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train[target_column],
        encoding_columns=INTERACTION_HISTORICAL_ENCODING_COLUMNS,
    )

    categorical_columns = [
        col
        for col in [
            *HISTORICAL_ENCODING_COLUMNS,
            *INTERACTION_HISTORICAL_ENCODING_COLUMNS,
            "departure_hour_bucket",
        ]
        if col in train_features.columns and col in test_features.columns
    ]
    if categorical_columns:
        train_features = train_features.drop(columns=categorical_columns)
        test_features = test_features.drop(columns=categorical_columns)

    train_features, test_features, _ = select_informative_features(
        train_features=train_features,
        test_features=test_features,
        y_train=y_train[target_column],
        method=feature_selection_method,
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
