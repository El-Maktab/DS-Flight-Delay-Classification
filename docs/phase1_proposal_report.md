# Phase 1 — Proposal & Dataset Validation Report

### Applied Data Science Project | Spring 2026

### Team 5 | Cairo University — Faculty of Engineering, Computer Engineering Department

| Name | BN |
| --- | --- |
| Ahmed Hamed Gaber Hamed | 3 |
| Akram Hany Karam Salam | 13 |
| Amir Anwar Bekhit Awd Kedis | 14 |
| Somia Saad Esmail Elshemy | 26 |

---

## 1\. Executive Summary

Multi-class classification model to predict flight departure delay severity before pushback. Data sources: 2015 BTS domestic flight records (CSV) + Open-Meteo Historical Weather API. Four classes: **On Time**, **Minor Delay**, **Major Delay**, **Cancelled**. Final dataset: 116,381 rows × 55 columns. Validation found no full-row duplicates and only expected missing-value patterns.

---

## 2\. Problem Statement

### 2.1 Business Problem

Flight delays cause operational disruptions, increased costs, and poor passenger experience. The goal is to **predict departure delay severity before pushback**, enabling ground operations to take proportional action in advance.

### 2.2 Stakeholders

**Primary:** Airline operations team and airport management.  
**Secondary:** Passengers and customer service teams.

### 2.3 Why Classification

Delay severity is a discrete outcome, not a continuous one. Each class maps to a distinct operational response, making classification the best option to use. Multi-class over binary captures severity categories. Misclassification costs are asymmetric, missing a major delay is worse than over-alerting, which is handled through business-oriented evaluation metrics.

### 2.4 Decision-Making Impact

| Predicted Class | Label | Operational Action |
| --- | --- | --- |
| `on_time` | On Time (<= 15 min) | No action |
| `minor_delay` | Minor Delay (15-45 min) | Notify passengers; monitor connections |
| `major_delay` | Major Delay (> 45 min) | Rebook connections; alert ground crew |
| `cancelled` | Cancelled | Trigger rebooking and compensation protocols |

---

## 3\. Target Variable Definition

The classification target is derived from the `DEPARTURE_DELAY` column (minutes) and the `CANCELLED` flag.

| Class | Label | Definition | Count | % |
| --- | --- | --- | --- | --- |
| `on_time` | On Time | `DEPARTURE_DELAY ≤ 15` min (FAA standard threshold) | 94,345 | 81.1% |
| `minor_delay` | Minor Delay | `15 < DEPARTURE_DELAY ≤ 45` min | 11,433 | 9.8% |
| `major_delay` | Major Delay | `DEPARTURE_DELAY > 45` min | 8,833 | 7.6% |
| `cancelled` | Cancelled | `CANCELLED == 1` | 1,770 | 1.5% |

**Why `DEPARTURE_DELAY`:** The goal is pre-departure action, departure delay is the directly actionable outcome.

**Class thresholds:** 15-minute threshold follows the FAA on-time standard, 45-minute split is where airlines typically begin rebooking connecting passengers.

**Cancelled flights:** Retained as a distinct fourth class. They have no `DEPARTURE_DELAY` but cancellation is itself a severe outcome. Final handling deferred to the preprocessing phase.

**Class imbalance:** `on_time` dominates at 81.1%. Will be addressed using **SMOTE and other techniques while training**.

---

## 4\. Dataset Documentation

### 4.1 Source 1 — CSV (BTS / Kaggle)

| Field | Details |
| --- | --- |
| **Name** | 2015 Flight Delays and Cancellations |
| **Provider** | US Department of Transportation — Bureau of Transportation Statistics (BTS) |
| **URL** | [https://www.kaggle.com/datasets/usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays) |
| **Citation** | U.S. Department of Transportation, Bureau of Transportation Statistics. (2015). *Airline On-Time Performance Data*. Kaggle. |
| **Acquisition Method** | Direct CSV download (flat file) |
| **Raw Size** | ~5,819,079 flight records × 31 features |
| **Sampled Size** | 116,381 rows (2% stratified random sample by month, `random_state=42`) |
| **Coverage** | All domestic US flights, calendar year 2015 |
| **License** | Public domain (US government data) |

**Key Features:**

| Feature | Type | Description |
| --- | --- | --- |
| `YEAR` | int | Year of flight (2015) |
| `MONTH` | int | Month (1–12) |
| `DAY` | int | Day of month |
| `DAY_OF_WEEK` | int | 1=Monday … 7=Sunday |
| `AIRLINE` | str | IATA airline carrier code (e.g., AA, DL, UA) |
| `FLIGHT_NUMBER` | int | Airline-assigned flight number |
| `TAIL_NUMBER` | str | Aircraft tail registration number |
| `ORIGIN_AIRPORT` | str | IATA code of departure airport |
| `DESTINATION_AIRPORT` | str | IATA code of arrival airport |
| `SCHEDULED_DEPARTURE` | int | Scheduled departure time (HHMM) |
| `DEPARTURE_TIME` | float | Actual departure time (HHMM) — post-departure, excluded from features |
| `DEPARTURE_DELAY` | float | Departure delay in minutes — post-departure, excluded from features |
| `TAXI_OUT` | float | Taxi-out time in minutes — post-departure, excluded from features |
| `WHEELS_OFF` | float | Wheels-off time (HHMM) — post-departure, excluded from features |
| `SCHEDULED_TIME` | float | Scheduled flight duration (minutes) |
| `ELAPSED_TIME` | float | Actual elapsed time — post-departure, excluded from features |
| `AIR_TIME` | float | Airborne time — post-departure, excluded from features |
| `DISTANCE` | int | Flight distance (miles) |
| `WHEELS_ON` | float | Wheels-on time — post-departure, excluded from features |
| `TAXI_IN` | float | Taxi-in time — post-departure, excluded from features |
| `SCHEDULED_ARRIVAL` | int | Scheduled arrival time (HHMM) |
| `ARRIVAL_TIME` | float | Actual arrival time — post-departure, excluded from features |
| `ARRIVAL_DELAY` | float | Arrival delay in minutes — used to build target label only |
| `DIVERTED` | int | 1 if flight was diverted |
| `CANCELLED` | int | 1 if flight was cancelled |
| `CANCELLATION_REASON` | str | A=Carrier, B=Weather, C=NAS, D=Security |
| `AIR_SYSTEM_DELAY` | float | Delay minutes from NAS — post-departure, excluded from features |
| `SECURITY_DELAY` | float | Delay minutes from security — post-departure, excluded from features |
| `AIRLINE_DELAY` | float | Delay minutes from carrier — post-departure, excluded from features |
| `LATE_AIRCRAFT_DELAY` | float | Delay minutes from late aircraft — post-departure, excluded from features |
| `WEATHER_DELAY` | float | Delay minutes from weather — post-departure, excluded from features |

---

### 4.2 Source 2 — REST API (Open-Meteo Historical Weather)

| Field | Details |
| --- | --- |
| **Name** | Open-Meteo Historical Weather API |
| **Provider** | Open-Meteo |
| **URL** | [https://open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api) |
| **Cost** | Free, no API key required |
| **Acquisition Method** | HTTP GET requests, one call per unique airport per year; responses cached as JSON in `data/raw/weather_cache/{IATA}_{YEAR}.json` |
| **Query Parameters** | `latitude`, `longitude`, `start_date=2015-01-01`, `end_date=2015-12-31`, `hourly` weather variables, `timezone=America/New_York` |

**11 Weather Variables Retrieved (per airport, per hour):**

| Feature (origin) | Feature (dest) | Unit | Description |
| --- | --- | --- | --- |
| `temperature_c` | `dest_temperature_c` | °C | Air temperature at 2 m |
| `precipitation_mm` | `dest_precipitation_mm` | mm | Total precipitation |
| `rain_mm` | `dest_rain_mm` | mm | Rain component |
| `snowfall_cm` | `dest_snowfall_cm` | cm | Snowfall |
| `wind_speed_kmh` | `dest_wind_speed_kmh` | km/h | Wind speed at 10 m |
| `wind_direction_deg` | `dest_wind_direction_deg` | degrees | Wind direction |
| `wind_gusts_kmh` | `dest_wind_gusts_kmh` | km/h | Wind gusts |
| `cloud_cover_pct` | `dest_cloud_cover_pct` | % | Cloud cover |
| `weather_code` | `dest_weather_code` | WMO code | Precipitation/fog/storm type |
| `relative_humidity_pct` | `dest_relative_humidity_pct` | % | Relative humidity |
| `pressure_msl_hpa` | `dest_pressure_msl_hpa` | hPa | Sea-level pressure |

**Why destination weather?** Destination weather at scheduled arrival time is knowable before departure (weather forecasts) and directly influences whether the plane can land. Including it improves the model’s ability to predict arrival delays caused by destination-side conditions.

## 5\. Merge Strategy

**Type:** Horizontal merge, weather columns are appended to each flight row.

**Two joins performed:**

| Join | Key | Weather Columns Added |
| --- | --- | --- |
| Origin weather | `(ORIGIN_AIRPORT, MONTH, DAY, departure_hour)` | 11 columns (`temperature_c`, etc.) |
| Destination weather | `(DESTINATION_AIRPORT, MONTH, arrival_day, scheduled_arrival_hour)` | 11 columns (`dest_temperature_c`, etc.) |

**Timezone handling:** The Open-Meteo API returns timestamps in `America/New_York`. Flight schedules are in each airport’s local time. `timezonefinder` resolves each airport’s IANA timezone from its coordinates, `zoneinfo` converts local departure/arrival hours to Eastern before building the lookup key, correctly handling DST for all 2015 dates.

**Overnight flights:** Arrival day is computed by adding `SCHEDULED_TIME` (minutes) to departure hour, detecting day rollovers where the flight arrives the next calendar day.

**Row counts before/after merging:**

| Stage | Row Count |
| --- | --- |
| Raw BTS flights CSV | ~5,819,079 |
| After 2% stratified sample (by month) | 116,381 |
| After weather join (no rows dropped) | 116,381 |
| Rows with missing origin weather (numeric airport codes) | 9,837 (8.45%) |
| Rows with missing destination weather (numeric airport codes) | 9,823 (8.44%) |
| Final dataset | **116,381 rows × 55 columns** |

> The join does not drop rows, flights with no valid airport code get `NaN` weather values. These ~9,837 rows will be dropped in the cleaning phase as weather is a core feature set.

---

## 6\. Engineered Features

Features derived during acquisition (present in `flights_weather.csv`):

| Feature | Derivation | Notes |
| --- | --- | --- |
| `departure_hour` | `SCHEDULED_DEPARTURE // 100` | Hour of scheduled departure (0–23) |
| `scheduled_arrival_hour` | `SCHEDULED_ARRIVAL // 100` | Hour of scheduled arrival (0–23) |
| `dest_*` weather columns | Open-Meteo API at destination | 11 destination weather features |

Additional features to be engineered during the preprocessing phase:

-   `is_weekend`, `is_holiday`, `route` (ORIGIN + DESTINATION), historical delay rates per airline/route, `busy_bad_weather_score`.

---

## 7\. Data Validation Report

Validation was performed via `src/validation/validate.py` covering 12 checks. Results are reproduced from `notebooks/validation_report.ipynb`.

**Dataset:** `data/processed/flights_weather.csv,` 116,381 rows × 55 columns

### 7.1 Summary

| Check | Result |
| --- | --- |
| Schema | All 55 columns present, all types are correct too |
| Completeness | Weather cols 8.45% null (numeric airports, expected), delay breakdown 81.75% null (expected, only set when delayed), `TAIL_NUMBER` 0.22% null |
| Duplicates | 0 full-row duplicates, 864 flight-level duplicates (codeshares — expected) |
| Value Ranges | All numeric values within expected bounds |
| Consistency | 97 cancelled flights have a `DEPARTURE_TIME` (taxied before cancellation) |
| Outliers | Detected via IQR, delay and distance columns show expected right-skewed extremes |
| Referential Integrity | 9,723 rows use numeric FAA IDs instead of IATA codes — no coordinate match |
| Format Validation | All IATA codes, HHMM times, and YEAR values conform to expected formats |
| Temporal Validity | All (MONTH, DAY) combinations are valid calendar dates |
| Cardinality | 14 unique airlines, ~322 unique origin and destination airports |
| Statistics | Descriptive statistics within expected bounds for all numeric columns |
| Target Readiness | 340 non-cancelled rows unlabellable (no `ARRIVAL_DELAY`), 4-class distribution documented |

---

### 7.2 Completeness — Missing Values

| Column | Missing Count | Missing % | Explanation |
| --- | --- | --- | --- |
| `CANCELLATION_REASON` | 114,611 | 98.48% | Only populated for cancelled flights — expected |
| `LATE_AIRCRAFT_DELAY` | 95,143 | 81.75% | Only populated for delayed flights — expected |
| `AIR_SYSTEM_DELAY` | 95,143 | 81.75% | Same — expected |
| `SECURITY_DELAY` | 95,143 | 81.75% | Same — expected |
| `WEATHER_DELAY` | 95,143 | 81.75% | Same — expected |
| `AIRLINE_DELAY` | 95,143 | 81.75% | Same — expected |
| Origin weather cols (11) | 9,837 | 8.45% | Numeric airport codes — no IATA match |
| Dest weather cols (11) | 9,823 | 8.44% | Same (slightly different set of flights) |
| `ARRIVAL_DELAY` | 2,110 | 1.81% | Cancelled flights + 340 diverted — expected |
| `AIR_TIME` / `ELAPSED_TIME` | 2,110 | 1.81% | Cancelled/diverted flights — expected |
| `TAXI_IN` / `WHEELS_ON` / `ARRIVAL_TIME` | 1,829 | 1.57% | Flights that did not arrive |
| `WHEELS_OFF` / `TAXI_OUT` | 1,748 | 1.50% | Flights that did not depart |
| `DEPARTURE_TIME` / `DEPARTURE_DELAY` | 1,673 | 1.44% | Cancelled flights |
| `TAIL_NUMBER` | 258 | 0.22% | Missing tail registration |

---

### 7.3 Duplicates

| Check | Count | Notes |
| --- | --- | --- |
| Full-row duplicates | 0 | No exact row duplicates |
| Flight-level duplicates (AIRLINE + FLIGHT\_NUMBER + MONTH + DAY) | 864 | Codeshare flights — multiple airlines operating the same physical flight; expected |

---

### 7.4 Value Ranges

All numeric columns fall within expected bounds:

| Column | Valid Range | Violations |
| --- | --- | --- |
| `MONTH` | 1–12 | 0 |
| `DAY` | 1–31 | 0 |
| `DAY_OF_WEEK` | 1–7 | 0 |
| `SCHEDULED_DEPARTURE` | 0–2359 | 0 |
| `DISTANCE` | \> 0 | 0 |
| `SCHEDULED_TIME` | \> 0 | 0 |

Note: ~9,837 rows have numeric values in `ORIGIN_AIRPORT` (e.g., `14747`) that are FAA numeric station IDs rather than IATA codes. These are not range violations, they are a data quality issue documented under Referential Integrity.

---

### 7.5 Consistency Checks

| Check | Rows Affected | Description |
| --- | --- | --- |
| Cancelled flights with `DEPARTURE_TIME` | 97 | Flights that began taxiing before cancellation — a known data quirk, not dropped |

---

### 7.6 Outlier Detection (IQR Method, 1.5× fence)

Outliers were detected using the IQR method across key numeric columns. Extreme values in delay-related columns are expected (heavy weather events, ground stops) and will be handled during the cleaning phase by binning rather than dropping.

---

### 7.7 Referential Integrity

| Check | Unmatched Count | Unmatched Values (sample) |
| --- | --- | --- |
| `AIRLINE` codes vs `airlines.csv` | 0 | All match |
| IATA airport codes vs `airports.csv` | 9,723 rows | Numeric FAA IDs (e.g., 14747, 13830) — no IATA equivalent |

The 9,723 rows with numeric airport codes (~8.35% of the dataset) have no coordinates and therefore no weather data. **Decision:** drop in the cleaning phase, weather is a core feature set and imputation would be meaningless.

---

### 7.8 Format Validation

| Check | Violations |
| --- | --- |
| IATA codes (3 uppercase letters) | 0 (among non-numeric codes) |
| HHMM times (minutes 00–59) | 0 |
| `YEAR` = 2015 | 0 |
| `CANCELLATION_REASON` in {A, B, C, D} | 0 |

---

### 7.9 Temporal Validity

All (MONTH, DAY) combinations are valid calendar dates for 2015. No invalid dates detected.

---

### 7.10 Cardinality

| Column | Unique Values | Top Values |
| --- | --- | --- |
| `AIRLINE` | 14 | WN, DL, AA, UA, OO |
| `ORIGIN_AIRPORT` | ~322 | ATL, ORD, DFW, LAX, DEN |
| `DESTINATION_AIRPORT` | ~322 | ATL, ORD, DFW, LAX, DEN |
| `DAY_OF_WEEK` | 7 | — |
| `CANCELLATION_REASON` | 4 | B (Weather), A (Carrier), C (NAS), D (Security) |

---

### 7.11 Target Variable Readiness

| Metric | Value |
| --- | --- |
| Total rows | 116,381 |
| Cancelled flights | 1,770 |
| `DEPARTURE_DELAY` nulls (non-cancelled) | 0 |
| Unlabellable rows | 0 |
| Labellable rows | 116,381 |

**Class distribution (target: `DEPARTURE_DELAY`):**

| Class | Label | Count | % |
| --- | --- | --- | --- |
| `on_time` | On Time (≤ 15 min) | 94,345 | 81.1% |
| `minor_delay` | Minor Delay (15–45 min) | 11,433 | 9.8% |
| `major_delay` | Major Delay (> 45 min) | 8,833 | 7.6% |
| `cancelled` | Cancelled | 1,770 | 1.5% |

---

### 7.12 Known Issues and Planned Resolutions

| Issue | Count | Severity | Planned Resolution |
| --- | --- | --- | --- |
| Rows with numeric airport codes (no weather) | 9,723 | Medium | Drop in cleaning phase |
| Cancelled flights with `DEPARTURE_TIME` | 97 | Low | Flights that began taxiing before cancellation, document as data quirk, keep |
| Missing `TAIL_NUMBER` | 258 | Low | Drop column — not a model feature |
| Delay breakdown cols (post-departure leakage) | 5 cols | High | Drop columns in cleaning phase |