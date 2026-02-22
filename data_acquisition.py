"""
NeuroSense AIoT — Data Acquisition Layer
-----------------------------------------
Handles retrieval of multivariate physiological time-series from ThingSpeak
via its REST API.  Supports JSON and CSV endpoints, robust retry logic with
exponential backoff, and correct timestamp parsing.
"""

import time
import logging
from typing import Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_json_url(results: int = config.THINGSPEAK_NUM_RESULTS) -> str:
    """Construct the ThingSpeak JSON feed URL."""
    return (
        f"{config.THINGSPEAK_BASE_URL}/channels/{config.THINGSPEAK_CHANNEL_ID}"
        f"/feeds.json?api_key={config.THINGSPEAK_READ_API_KEY}"
        f"&results={results}"
    )


def _build_csv_url(results: int = config.THINGSPEAK_NUM_RESULTS) -> str:
    """Construct the ThingSpeak CSV feed URL."""
    return (
        f"{config.THINGSPEAK_BASE_URL}/channels/{config.THINGSPEAK_CHANNEL_ID}"
        f"/feeds.csv?api_key={config.THINGSPEAK_READ_API_KEY}"
        f"&results={results}"
    )


def _request_with_retry(url: str) -> requests.Response:
    """
    Execute an HTTP GET with exponential-backoff retry logic.

    Raises
    ------
    ConnectionError
        If all retry attempts are exhausted.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.THINGSPEAK_MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=config.THINGSPEAK_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exception = exc
            wait = config.THINGSPEAK_RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1f s ...",
                attempt, config.THINGSPEAK_MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise ConnectionError(
        f"ThingSpeak API unreachable after {config.THINGSPEAK_MAX_RETRIES} attempts. "
        f"Last error: {last_exception}"
    )


def _rename_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename ThingSpeak field columns to human-readable signal names and
    parse the timestamp column to a proper DatetimeIndex.
    """
    rename_map = {k: v for k, v in config.FIELD_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if config.TIMESTAMP_COLUMN in df.columns:
        df[config.TIMESTAMP_COLUMN] = pd.to_datetime(
            df[config.TIMESTAMP_COLUMN], utc=True, errors="coerce"
        )
        df = df.set_index(config.TIMESTAMP_COLUMN).sort_index()

    # Coerce signal columns to numeric, replacing unparseable tokens with NaN
    for col in config.SIGNAL_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_json(results: int = config.THINGSPEAK_NUM_RESULTS) -> pd.DataFrame:
    """
    Retrieve physiological data from ThingSpeak as JSON and return a
    cleaned DataFrame with meaningful column names and a DatetimeIndex.

    Parameters
    ----------
    results : int
        Number of most-recent entries to fetch.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with columns: HR, RMSSD, SkinTemp,
        Moisture, StressScore.
    """
    url = _build_json_url(results)
    logger.info("Fetching JSON feed: %s", url)
    response = _request_with_retry(url)
    payload = response.json()
    feeds = payload.get("feeds", [])

    if not feeds:
        logger.warning("JSON feed returned zero entries.")
        return pd.DataFrame()

    df = pd.DataFrame(feeds)
    df = _rename_and_parse(df)
    logger.info("Acquired %d records via JSON.", len(df))
    return df


def fetch_csv(results: int = config.THINGSPEAK_NUM_RESULTS) -> pd.DataFrame:
    """
    Retrieve physiological data from ThingSpeak as CSV and return a
    cleaned DataFrame.

    Parameters
    ----------
    results : int
        Number of most-recent entries to fetch.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with columns: HR, RMSSD, SkinTemp,
        Moisture, StressScore.
    """
    url = _build_csv_url(results)
    logger.info("Fetching CSV feed: %s", url)
    response = _request_with_retry(url)

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    df = _rename_and_parse(df)
    logger.info("Acquired %d records via CSV.", len(df))
    return df


def fetch_data(
    mode: str = "json",
    results: int = config.THINGSPEAK_NUM_RESULTS,
) -> pd.DataFrame:
    """
    Unified entry point for data acquisition.

    Parameters
    ----------
    mode : str
        'json' or 'csv'.
    results : int
        Number of entries to retrieve.

    Returns
    -------
    pd.DataFrame
    """
    if mode == "csv":
        return fetch_csv(results)
    return fetch_json(results)
