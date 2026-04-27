#!/usr/bin/env bash
# run_phase1.sh — Set up the environment and run the Phase 1 pipeline.
#
# Steps:
#   1. Install Poetry (if missing)
#   2. Install project dependencies via Poetry
#   3. Create .env from .env.example (if missing)
#   4. Download and extract the weather cache (if missing)
#   5. Run data sampling        → data/raw/flights_sampled.csv
#   6. Run weather enrichment   → data/processed/flights_weather.csv
#   7. Run validation           → prints validation report to stdout
#   8. Run preprocessing        → prints preprocessing summary to stdout

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

# ── 1. Poetry ──────────────────────────────────────────────────────────────────
step "Checking Poetry installation..."
if ! command -v poetry &>/dev/null; then
    warn "Poetry not found — installing via the official installer."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "Poetry $(poetry --version)"

# ── 2. Install dependencies ────────────────────────────────────────────────────
step "Installing project dependencies..."
poetry install --no-interaction

# ── 3. Environment file ────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    step "Creating .env from .env.example..."
    cp .env.example .env
fi

# ── 4. Weather cache ───────────────────────────────────────────────────────────
CACHE_DIR="data/raw/weather_cache"
GDRIVE_ID="1BeEICuVMeoTwS45GwCMCNiOjcwxqctsM"
ZIP_PATH="data/raw/weather_cache.zip"

if [ -d "$CACHE_DIR" ] && [ -n "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    warn "Weather cache already present at $CACHE_DIR — skipping download."
else
    step "Downloading weather cache from Google Drive..."
    mkdir -p data/raw
    poetry run gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$ZIP_PATH"

    step "Extracting weather cache..."
    unzip -q "$ZIP_PATH" -d data/raw/
    rm "$ZIP_PATH"
    echo "Weather cache extracted to $CACHE_DIR"
fi

# ── 5. Sampling ────────────────────────────────────────────────────────────────
step "Running data sampling (flight_delay_classification/acquisition/sample.py)..."
poetry run python -m flight_delay_classification.acquisition.sample

# ── 6. Weather enrichment ──────────────────────────────────────────────────────
step "Running weather enrichment (flight_delay_classification/acquisition/weather.py)..."
poetry run python -m flight_delay_classification.acquisition.weather

# ── 7. Validation ──────────────────────────────────────────────────────────────
step "Running validation (flight_delay_classification/validation/validate.py)..."
poetry run python -m flight_delay_classification.validation.validate

# ── 8. Preprocessing ───────────────────────────────────────────────────────────
step "Running preprocessing (flight_delay_classification/preprocessing/preprocess.py)..."
poetry run python -m flight_delay_classification.preprocessing.preprocess
