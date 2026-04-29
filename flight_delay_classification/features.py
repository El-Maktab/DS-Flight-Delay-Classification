"""
Author: Amir Anwar
Date: 2026-04-28

Description:
    Feature pipeline
"""

from pathlib import Path

import pandas as pd
from loguru import logger
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


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate labels and apply train-only frequency encoding."""

    y_train = train_df[[target_column]].copy()
    y_test = test_df[[target_column]].copy()

    train_features = train_df.drop(columns=[target_column, *NON_PREDICTIVE_COLUMNS])
    test_features = test_df.drop(columns=[target_column, *NON_PREDICTIVE_COLUMNS])

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

    if route_columns:
        train_features = train_features.drop(columns=route_columns)
        test_features = test_features.drop(columns=route_columns)

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
    )


if __name__ == "__main__":
    app()
