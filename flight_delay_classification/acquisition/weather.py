"""
Author: Akram Hany
Date: 2026-04-07

Description:
    Fetch hourly weather data from the Open-Meteo Historical Weather API for
    each unique origin airport in the sampled flights dataset.

Flow:
    1. Load flights_sampled.csv
    2. Load airports.csv  (IATA_CODE → LATITUDE, LONGITUDE)
    3. Extract unique ORIGIN_AIRPORT codes
    4. For each unique airport, fetch the full year (2015-01-01 → 2015-12-31)
       in a single API call with local JSON caching (one file per airport)
    5. Parse all 8,760 hourly entries into a lookup dict keyed by
       (IATA, month, day, hour) → weather values
    6. Join weather back onto flights using (ORIGIN_AIRPORT, MONTH, DAY, departure_hour)
    7. Save merged dataset to data/processed/flights_weather.csv
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import pandas as pd
import requests
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# Paths from .env
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
EXTERNAL_DIR = Path(os.getenv("EXTERNAL_DATA_DIR", "data/external"))

CACHE_DIR = RAW_DIR / "weather_cache"
FLIGHTS_FILE = RAW_DIR / "flights_sampled.csv"
AIRPORTS_FILE = EXTERNAL_DIR / "airports.csv"
OUTPUT_FILE = PROCESSED_DIR / "flights_weather.csv"

# API Config
API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARIABLES = (
    "temperature_2m,precipitation,rain,snowfall,"
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "cloud_cover,weather_code,relative_humidity_2m,pressure_msl"
)
YEAR = 2015


class RateLimiter:
    """
    Smart rate limiter that tracks API usage.
    Uses a sliding time window to count recent requests.
    """

    def __init__(self, max_requests=600, time_window=60):
        """
        Args:
            max_requests: Maximum requests allowed in the time window
            time_window: Time window in seconds (60 = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []  # List of timestamps of past requests

    def wait_if_needed(self):
        """Wait if we've hit the rate limit before making a new request."""
        now = time.time()

        # Remove old timestamps outside the sliding time window
        self.requests = [
            req_time for req_time in self.requests if now - req_time < self.time_window
        ]

        # If we've used up our quota, sleep until the oldest request expires
        if len(self.requests) >= self.max_requests:
            oldest_request = self.requests[0]
            sleep_time = self.time_window - (now - oldest_request)
            if sleep_time > 0:
                print(
                    f"⏰ Rate limit reached. Sleeping for {sleep_time:.1f} seconds..."
                )
                time.sleep(sleep_time)
            self.requests = []  # Clear after sleeping

        # Record the timestamp of this new request
        self.requests.append(now)


rate_limiter = RateLimiter()


def load_airport_coords(airports_path: Path) -> dict[str, tuple[float, float]]:
    """Return {IATA_CODE: (latitude, longitude)} for each airport with valid coordinates."""
    df = pd.read_csv(airports_path).dropna(subset=["LATITUDE", "LONGITUDE"])
    return {
        row["IATA_CODE"]: (row["LATITUDE"], row["LONGITUDE"])
        for _, row in df.iterrows()
    }


def cache_path(iata: str) -> Path:
    """Return the local cache file path for a given airport (full year)."""
    return CACHE_DIR / f"{iata}_{YEAR}.json"


def fetch_weather_year(lat: float, lon: float) -> dict | None:
    """
    Fetch full-year hourly weather from Open-Meteo for a single airport.
    Retries with exponential backoff on 429 rate limit errors.
    Returns the parsed JSON response or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{YEAR}-01-01",
        "end_date": f"{YEAR}-12-31",
        "hourly": WEATHER_VARIABLES,
        "timezone": "America/New_York",
    }
    max_retries = 5
    backoff = 2  # seconds, doubles each retry

    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params, timeout=30)
            if response.status_code == 429:
                wait = backoff * (2**attempt)
                log.warning(
                    "Rate limited (429). Retrying in %ds (attempt %d/%d)...",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log.warning("API call failed for (%s, %s): %s", lat, lon, e)
            return None

    log.error("All %d retries failed for (%s, %s)", max_retries, lat, lon)
    return None


def get_weather_cached(iata: str, lat: float, lon: float) -> dict | None:
    """Return cached full-year weather JSON if it exists, otherwise fetch and cache it."""
    path = cache_path(iata)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    rate_limiter.wait_if_needed()
    data = fetch_weather_year(lat, lon)
    if data:
        with open(path, "w") as f:
            json.dump(data, f)
    return data


def parse_full_year(data: dict) -> dict[tuple, dict]:
    """
    Parse a full-year API response into a lookup dict.
    Returns {(month, day, hour): weather_dict} for all 8,760 hours.
    """
    hourly = data.get("hourly", {})
    timestamps = hourly.get("time", [])
    lookup = {}

    for i, ts in enumerate(timestamps):
        # ts format: "2015-03-15T14:00"
        date_part, time_part = ts.split("T")
        _, month_str, day_str = date_part.split("-")
        hour = int(time_part.split(":")[0])
        month = int(month_str)
        day = int(day_str)

        lookup[(month, day, hour)] = {
            "temperature_c": hourly.get("temperature_2m", [None])[i],
            "precipitation_mm": hourly.get("precipitation", [None])[i],
            "rain_mm": hourly.get("rain", [None])[i],
            "snowfall_cm": hourly.get("snowfall", [None])[i],
            "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[i],
            "wind_direction_deg": hourly.get("wind_direction_10m", [None])[i],
            "wind_gusts_kmh": hourly.get("wind_gusts_10m", [None])[i],
            "cloud_cover_pct": hourly.get("cloud_cover", [None])[i],
            "weather_code": hourly.get("weather_code", [None])[i],
            "relative_humidity_pct": hourly.get("relative_humidity_2m", [None])[i],
            "pressure_msl_hpa": hourly.get("pressure_msl", [None])[i],
        }

    return lookup


def build_weather_lookup(
    unique_airports: list[str],
    coords: dict[str, tuple[float, float]],
) -> dict[tuple, dict]:
    """
    Fetch full-year weather for each unique airport.
    Returns a dict keyed by (IATA, month, day, hour) -> weather values.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lookup: dict[tuple, dict] = {}
    total = len(unique_airports)

    for i, iata in enumerate(unique_airports, 1):
        if iata not in coords:
            log.warning("No coordinates found for airport: %s — skipping", iata)
            continue

        lat, lon = coords[iata]
        log.info("Fetching weather for %s (%d / %d)", iata, i, total)

        data = get_weather_cached(iata, lat, lon)
        if data is None:
            continue
        # time.sleep(1)  # 1 second between requests to avoid rate limiting

        airport_lookup = parse_full_year(data)
        for (month, day, hour), weather in airport_lookup.items():
            lookup[(iata, month, day, hour)] = weather

    return lookup


def build_airport_timezones(
    unique_airports: list[str],
    coords: dict[str, tuple[float, float]],
) -> dict[str, str]:
    """Return {IATA: IANA_timezone_name} for each airport using its coordinates."""
    tf = TimezoneFinder()
    result: dict[str, str] = {}
    for iata in unique_airports:
        if iata not in coords:
            continue
        lat, lon = coords[iata]
        tz_name = tf.timezone_at(lat=lat, lng=lon)
        result[iata] = tz_name or "America/New_York"
    return result


def local_to_eastern(
    month: int, day: int, local_hour: int, airport_tz: str
) -> tuple[int, int, int]:
    """
    Convert a local (month, day, hour) to Eastern time.
    Returns (month, day, hour) in America/New_York, accounting for DST.
    """
    dt_local = datetime(YEAR, month, day, local_hour, tzinfo=ZoneInfo(airport_tz))
    dt_eastern = dt_local.astimezone(ZoneInfo("America/New_York"))
    return dt_eastern.month, dt_eastern.day, dt_eastern.hour


def arrival_day(
    departure_day: int, departure_hour: int, scheduled_time_min: float
) -> int:
    """
    Return the arrival day-of-month, accounting for overnight flights.
    If scheduled_time_min is NaN we assume same-day arrival.
    """
    if pd.isna(scheduled_time_min):
        return departure_day
    arrival_minutes = departure_hour * 60 + scheduled_time_min
    return departure_day + int(arrival_minutes // (24 * 60))


def merge_weather(
    flights: pd.DataFrame,
    lookup: dict,
    airport_timezones: dict[str, str],
) -> pd.DataFrame:
    """Join origin and destination weather onto flights.

    Origin  : keyed on (ORIGIN_AIRPORT,      MONTH, DAY,          departure_hour)
    Destination: keyed on (DESTINATION_AIRPORT, MONTH, arrival_day, scheduled_arrival_hour)

    All hours are converted from local airport time to Eastern before lookup
    because the cached weather timestamps are in America/New_York.
    """
    flights["departure_hour"] = flights["SCHEDULED_DEPARTURE"] // 100
    flights["scheduled_arrival_hour"] = flights["SCHEDULED_ARRIVAL"] // 100

    def get_origin_weather(r):
        iata = r["ORIGIN_AIRPORT"]
        tz = airport_timezones.get(iata, "America/New_York")
        e_month, e_day, e_hour = local_to_eastern(
            int(r["MONTH"]), int(r["DAY"]), int(r["departure_hour"]), tz
        )
        return lookup.get((iata, e_month, e_day, e_hour), {})

    def get_dest_weather(r):
        iata = r["DESTINATION_AIRPORT"]
        tz = airport_timezones.get(iata, "America/New_York")
        arr_day = arrival_day(
            int(r["DAY"]), int(r["departure_hour"]), r["SCHEDULED_TIME"]
        )
        try:
            e_month, e_day, e_hour = local_to_eastern(
                int(r["MONTH"]), arr_day, int(r["scheduled_arrival_hour"]), tz
            )
        except ValueError:
            e_month, e_day, e_hour = local_to_eastern(
                int(r["MONTH"]), int(r["DAY"]), int(r["scheduled_arrival_hour"]), tz
            )
        return lookup.get((iata, e_month, e_day, e_hour), {})

    log.info("Joining origin weather ...")
    origin_weather_df = pd.DataFrame(
        flights.apply(get_origin_weather, axis=1).tolist(), index=flights.index
    )

    log.info("Joining destination weather ...")
    dest_weather_df = pd.DataFrame(
        flights.apply(get_dest_weather, axis=1).tolist(), index=flights.index
    ).rename(columns=lambda c: f"dest_{c}")

    return pd.concat([flights, origin_weather_df, dest_weather_df], axis=1)


def main():
    log.info("Loading flights from %s", FLIGHTS_FILE)
    flights = pd.read_csv(FLIGHTS_FILE, low_memory=False)
    log.info("Loaded %d flights", len(flights))

    log.info("Loading airport coordinates from %s", AIRPORTS_FILE)
    coords = load_airport_coords(AIRPORTS_FILE)
    log.info("Loaded coordinates for %d airports", len(coords))

    unique_airports = list(
        set(flights["ORIGIN_AIRPORT"].dropna().tolist())
        | set(flights["DESTINATION_AIRPORT"].dropna().tolist())
    )
    log.info(
        "Unique airports to fetch (origin + destination): %d", len(unique_airports)
    )

    lookup = build_weather_lookup(unique_airports, coords)
    log.info("Weather lookup built with %d hourly entries", len(lookup))

    log.info("Resolving airport timezones ...")
    airport_timezones = build_airport_timezones(unique_airports, coords)
    log.info("Resolved timezones for %d airports", len(airport_timezones))

    log.info("Merging weather onto flights ...")
    merged = merge_weather(flights, lookup, airport_timezones)

    origin_weather_cols = [
        "temperature_c",
        "precipitation_mm",
        "rain_mm",
        "snowfall_cm",
        "wind_speed_kmh",
        "wind_direction_deg",
        "wind_gusts_kmh",
        "cloud_cover_pct",
        "weather_code",
        "relative_humidity_pct",
        "pressure_msl_hpa",
    ]
    dest_weather_cols = [f"dest_{c}" for c in origin_weather_cols]

    missing_origin = merged[origin_weather_cols].isna().any(axis=1).sum()
    missing_dest = merged[dest_weather_cols].isna().any(axis=1).sum()

    log.info("Rows before merge : %d", len(flights))
    log.info("Rows after merge  : %d", len(merged))
    log.info(
        "Flights missing origin weather : %d (%.2f%%)",
        missing_origin,
        100 * missing_origin / len(merged),
    )
    log.info(
        "Flights missing dest weather   : %d (%.2f%%)",
        missing_dest,
        100 * missing_dest / len(merged),
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved merged dataset to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
