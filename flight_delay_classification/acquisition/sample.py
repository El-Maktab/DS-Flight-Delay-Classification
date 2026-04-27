"""
Author: Akram Hany
Date: 07/04/2026

Description: Sample a reproducible 2% stratified-by-month subset from the raw flights CSV.
Output: data/raw/flights_sampled.csv
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
INPUT_FILE = RAW_DIR / "flights.csv"
OUTPUT_FILE = RAW_DIR / "flights_sampled.csv"
SAMPLE_FRACTION = 0.02
RANDOM_STATE = 42


def sample_flights(input_path: Path, output_path: Path, frac: float) -> None:
    log.info("Reading %s ...", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    sampled = df.groupby("MONTH", group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=RANDOM_STATE)).reset_index(drop=True)

    log.info(
        "Sampled %d rows (%.1f%% of original) — distribution by month:\n%s",
        len(sampled),
        frac * 100,
        sampled["MONTH"].value_counts().sort_index().to_string(),
    )

    sampled.to_csv(output_path, index=False)
    log.info("Saved sampled dataset to %s", output_path)


if __name__ == "__main__":
    sample_flights(INPUT_FILE, OUTPUT_FILE, SAMPLE_FRACTION)
