# DS-Flight-Delay-Classification

## Team

Team 5, Cairo University, Faculty of Engineering, Computer Engineering Department

- Ahmed Hamed Gaber Hamed
- Akram Hany Karam Salam
- Amir Anwar Bekhit Awd Kedis
- Somia Saad Esmail Elshemy

## Project description

This project predicts how severe a flight departure delay will be before pushback. We use 2015 U.S. domestic flight records, add weather data, and classify each flight into one of four outcomes:

- On Time
- Minor Delay
- Major Delay
- Cancelled

The point is simple: give airline and airport teams something useful early enough to react.

## Setup

### Prerequisites

- Python 3.11+
- Poetry

### Install dependencies

```bash
poetry install
```

### Prepare the data

Download the flight delays dataset from Kaggle:

https://www.kaggle.com/datasets/usdot/flight-delays

Then place the files like this:

- `data/raw/flights.csv`
- supporting CSV files such as `airlines.csv` and `airports.csv` in `data/external/`

If you need a local environment file, copy it from the template:

```bash
cp .env.example .env
```

## Run instructions

### Option 1. Run the phase 1 pipeline step by step

These commands sample flights, fetch weather data, validate the result, and build the processed dataset used for modeling.

```bash
poetry run python -m flight_delay_classification.acquisition.sample
poetry run python -m flight_delay_classification.acquisition.weather
poetry run python -m flight_delay_classification.validation.validate
poetry run python -m flight_delay_classification.preprocessing.preprocess
```

### Option 2. Use the helper script

On Linux, you can run the same setup and preprocessing flow with:

```bash
bash run.sh
```

### Option 3. Train and compare models

To run the final model selection sweep:

```bash
poetry run python -m flight_delay_classification.modeling.run_all_models --experiment-name "flight-delay-final-model-selection" --run-prefix "final" --primary-metric balanced_accuracy
```

If you want the full list of single-model training and tuning commands, see [run_guide.md](run_guide.md).

### Optional: start the MLflow UI

```bash
poetry run mlflow ui --backend-store-uri "sqlite:///./mlflow.db" --default-artifact-root "file:///./mlartifacts"
```

## Project layout

```text
data/                          Raw, external, interim, and processed datasets
docs/                          Proposal and planning documents
flight_delay_classification/   Source code for acquisition, validation, preprocessing, and modeling
mlartifacts/                   MLflow artifact store
models/                        Saved trained models
notebooks/                     Analysis and validation notebooks
reports/                       Evaluation outputs and report assets
tests/                         Automated tests
```
