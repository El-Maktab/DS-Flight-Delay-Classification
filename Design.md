# Flight Delay Severity Classification — Project Report
### Applied Data Science Project | Spring 2026

---

## 1. Problem Definition & Business Context

**Business Problem:**
Predict the severity of a flight delay before departure, enabling airlines to take proactive operational decisions based on expected delay magnitude.

**Stakeholder:**
Airline operations team / airport management

**Classification Justification:**
This is a supervised multi-class classification problem. Given a set of features known before or at departure time (airline, route, scheduled time, weather conditions), the model predicts which of four delay severity categories a flight will fall into.

**Decision-Making Impact:**

| Predicted Class | Operational Action |
|---|---|
| 0 — On Time (< 15 min) | No action needed |
| 1 — Minor Delay (15–60 min) | Send passenger notification, monitor connecting flights |
| 2 — Major Delay (60–180 min) | Proactively rebook connecting passengers, alert ground crew |
| 3 — Severe Delay (> 180 min) | Escalate to operations center, trigger hotel/voucher protocols, notify downstream flights |

---

## 2. Target Variable

Derived from the `DEPARTURE_DELAY` column in the flight dataset.

| Class | Label | Definition | Approx. Distribution |
|---|---|---|---|
| 0 | On Time | Delay < 15 min | ~60% |
| 1 | Minor Delay | 15–60 min | ~20% |
| 2 | Major Delay | 60–180 min | ~15% |
| 3 | Severe Delay | > 180 min | ~5% |

> **Note on Class Imbalance:** Class 3 (Severe Delay) is rare (~5%). This imbalance must be reported in the validation phase and addressed during model training using **SMOTE applied exclusively to the training split**. SMOTE must never be applied to validation or test data.

---

## 3. Data Sources

### Source 1 — CSV Dataset (Kaggle)
- **Name:** 2015 Flight Delays and Cancellations
- **URL:** https://www.kaggle.com/datasets/usdot/flight-delays
- **Provider:** US Department of Transportation (Bureau of Transportation Statistics)
- **Size:** ~5.8 million flight records
- **Key Features:** airline code, origin airport, destination airport, scheduled departure time, actual departure time, departure delay (minutes), arrival delay, cancellation flag, delay reason breakdown (carrier, weather, NAS, security, late aircraft)
- **Acquisition Method:** Direct CSV download

### Source 2 — REST API
- **Name:** Open-Meteo Historical Weather API
- **URL:** https://open-meteo.com/en/docs/historical-weather-api
- **Cost:** Free, no API key required
- **Query Parameters:** latitude, longitude, date, hourly weather variables
- **Features Retrieved:** wind speed (km/h), precipitation (mm), visibility (km), cloud cover (%), temperature (°C)
- **Acquisition Method:** HTTP GET requests per unique (airport, date) combination, with local caching to avoid redundant calls

---

## 4. Merge Strategy

**Type:** Horizontal merge (adding new feature columns to existing rows)

**Merge Key:** `origin_airport_code` → mapped to `(latitude, longitude)` via an airport coordinates lookup table, then joined with `departure_date`

**Logic:**
1. Extract all unique `(origin_airport, departure_date)` combinations from the flight CSV
2. Query the Open-Meteo API for each combination to retrieve hourly weather at departure time
3. Join the weather features back onto the flight records using `(origin_airport, departure_date, departure_hour)` as the composite key

**Expected Result:** Each flight row is enriched with weather conditions at its origin airport at the time of departure. This merge is meaningful because weather is one of the strongest predictors of delays — the merge directly improves model performance in a demonstrable way.

**Row counts before/after merging must be documented in the report** to confirm no unintended row loss or duplication.

---

## 5. Feature Engineering

The following features should be engineered after merging:

**Time-based features:**
- `hour_of_day` — extracted from scheduled departure time
- `day_of_week` — Monday=0 … Sunday=6
- `month` — 1–12
- `is_weekend` — binary flag
- `is_holiday` — binary flag based on US federal holidays calendar

**Route & traffic features:**
- `route` — concatenation of origin + destination airport codes
- `historical_route_delay_rate` — mean delay rate per route (computed from training data only)
- `airport_daily_traffic` — number of flights per origin airport per day (busyness score)

**Airline features:**
- `historical_airline_delay_rate` — mean delay rate per airline (computed from training data only)

**Weather features (from API):**
- `wind_speed_kmh`
- `precipitation_mm`
- `visibility_km`
- `cloud_cover_pct`
- `temperature_c`

**Interaction features:**
- `busy_bad_weather_score` — product of `airport_daily_traffic` and `precipitation_mm` (normalized)

> **Important:** Historical rate features (route delay rate, airline delay rate) must be computed exclusively from the training split and then mapped onto validation and test sets to prevent data leakage.

---

## 6. Preprocessing Steps

- Drop rows with missing values in critical columns (departure delay, origin, destination, airline)
- Remove cancelled and diverted flights from the dataset (they are not delay prediction targets)
- Cap extreme outlier delay values (e.g., delays > 1000 minutes) or bin them into Class 3
- Encode categorical features: `airline`, `origin_airport`, `destination_airport` using target encoding or frequency encoding (not one-hot, due to high cardinality)
- Normalize/standardize numerical features (wind speed, precipitation, etc.) for models sensitive to scale (Logistic Regression, KNN)
- Apply SMOTE on training split only after the train/validation/test split is performed

---

## 7. Train / Validation / Test Split

| Split | Proportion | Purpose |
|---|---|---|
| Training | 70% | Model fitting and SMOTE balancing |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final unbiased evaluation |

Split should be stratified by target class to preserve class distribution across all splits.

---

## 8. Models

At least five models must be built, including one baseline:

| # | Model | Role |
|---|---|---|
| 1 | Logistic Regression | Baseline |
| 2 | Decision Tree | Interpretable tree-based model |
| 3 | Random Forest | Ensemble — strong generalization |
| 4 | XGBoost or LightGBM | Primary candidate for best performance |
| 5 | K-Nearest Neighbors | Distance-based contrasting approach |

All models must be logged in MLflow with hyperparameters, metrics, and saved as artifacts.

---

## 9. Evaluation Metrics

### Standard Metrics
- **Weighted F1-Score** — primary metric; handles multi-class imbalance
- **Macro ROC-AUC (One-vs-Rest)** — measures per-class discrimination ability

### Business-Oriented Metrics
- **Severe Delay Recall (Class 3)** — catching severe delays is operationally critical; missing them has the highest cost
- **Cost-Weighted Accuracy** — assigns higher penalty to misclassifying severe delays as on-time (the worst possible operational outcome)

**Metric justification for report:** False negatives on Class 3 are the most costly outcome — an operations team that is not warned of a severe delay cannot take corrective action. Therefore, Recall on Class 3 and Cost-Weighted Accuracy are prioritized over raw accuracy.

---

## 10. MLflow Experiment Tracking

Each model run must log:
- Model name and version
- All hyperparameters used
- Weighted F1-Score
- Macro ROC-AUC
- Severe Delay Recall (Class 3)
- Cost-Weighted Accuracy
- Trained model saved as MLflow artifact

A screenshot of the MLflow experiment comparison interface showing all runs side by side must be included in the final report.

---

## 11. EDA Visualizations (Minimum 5)

1. **Delay severity distribution by airline** — bar chart showing which carriers have the highest share of severe delays
2. **Delay severity heatmap by hour × day of week** — reveals when severe delays are most likely
3. **Weather vs delay severity boxplots** — wind speed, precipitation, and visibility distributions per delay class (directly justifies the API merge)
4. **Top 20 routes by average delay severity** — horizontal bar chart or route map
5. **Monthly delay severity trends** — line chart showing seasonality (winter storms, summer thunderstorms)

---

## 12. Project Structure (Cookie-Cutter)

```
flight-delay-classification/
│
├── data/
│   ├── raw/                  # Original CSV + cached API responses
│   ├── processed/            # Cleaned & merged data
│   └── external/             # Airport coordinates lookup table
│
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── acquisition/          # Download CSV, call & cache weather API
│   ├── validation/           # Data validation checks & reports
│   ├── preprocessing/        # Cleaning, encoding, scaling, SMOTE
│   ├── features/             # Feature engineering
│   ├── models/               # Model training & hyperparameter tuning
│   └── evaluation/           # Metrics, MLflow logging, reporting
│
├── tests/                    # Pytest unit & integration tests
├── .github/workflows/        # GitHub Actions CI configuration
├── Makefile                  # Task automation
├── .env                      # Environment-specific settings
├── pyproject.toml            # Poetry dependency management
├── poetry.lock
└── README.md                 # Setup and run instructions
```

---

## 13. Technical Requirements

- **Language:** Python
- **Dependency Management:** Poetry (`pyproject.toml` + `poetry.lock`)
- **Version Control:** Git (clean, organized repository)
- **Experiment Tracking:** MLflow
- **Testing:** Pytest with coverage reporting
- **Automation:** Makefile for preprocessing, training, evaluation, and testing tasks
- **CI Pipeline:** GitHub Actions — runs tests and linting on every push and pull request to main
- **Reproducibility:** `poetry install` + Makefile commands must fully reproduce results

---

## 14. Bonus Opportunities

| Bonus | Description | Marks |
|---|---|---|
| Streamlit Dashboard | Interactive dashboard showing EDA findings, model comparisons, and a live prediction interface (user inputs route + date → model returns predicted delay severity class) | 5% |
| Public Deployment | Deploy the model as a public endpoint | 5% |

---

## 15. Key Constraints & Reminders

- SMOTE must only be applied to the **training split**, never to validation or test data
- Historical rate features (airline/route delay rates) must be computed from **training data only** and mapped onto other splits
- All data must be real-world — no synthetic datasets except SMOTE oversampling
- Dataset must have ≥ 5,000 rows and ≥ 10 features after merging
- Validation report must document and highlight the class imbalance as a finding
- All code must follow clean code practices with logging and modular structure