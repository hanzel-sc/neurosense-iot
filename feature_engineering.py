"""
NeuroSense AIoT — Feature Engineering Layer
---------------------------------------------
Derives computational physiology features from cleaned time-series data:
inter-sample deltas, z-score normalisation, rolling statistics, first and
second derivatives (slope and acceleration), and signal stability metrics.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delta (inter-sample difference) features
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sample-to-sample differences for each primary signal.

    New columns added: HR_delta, RMSSD_delta, SkinTemp_delta, Moisture_delta.
    """
    df = df.copy()
    delta_map = {
        "HR": "HR_delta",
        "RMSSD": "RMSSD_delta",
        "SkinTemp": "SkinTemp_delta",
        "Moisture": "Moisture_delta",
    }
    for src, dst in delta_map.items():
        if src in df.columns:
            df[dst] = df[src].diff()
    return df


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-score normalisation to all primary signals.

    Uses a numerically stable denominator with epsilon to avoid
    division-by-zero on constant-valued signals.
    """
    df = df.copy()
    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        denominator = std if std > config.Z_SCORE_EPSILON else config.Z_SCORE_EPSILON
        df[f"{col}_zscore"] = (df[col] - mean) / denominator
    return df


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = config.ROLLING_WINDOW_SIZE,
    min_periods: int = config.MIN_PERIODS_ROLLING,
) -> pd.DataFrame:
    """
    Compute rolling mean and rolling variance for each primary signal.
    """
    df = df.copy()
    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        rolling = df[col].rolling(window=window, min_periods=min_periods)
        df[f"{col}_roll_mean"] = rolling.mean()
        df[f"{col}_roll_var"] = rolling.var()
    return df


# ---------------------------------------------------------------------------
# Signal dynamics — derivatives
# ---------------------------------------------------------------------------

def compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute first-order (slope) and second-order (acceleration) derivatives
    for each primary signal using finite differences.

    When timestamps are available as the index, derivatives are expressed
    per-second.  Otherwise, derivatives are per-sample.
    """
    df = df.copy()

    # Determine time spacing in seconds if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        dt_seconds = df.index.to_series().diff().dt.total_seconds()
    else:
        dt_seconds = pd.Series(1.0, index=df.index)

    # Avoid division by zero for identical timestamps
    dt_seconds = dt_seconds.replace(0, np.nan)

    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        # First derivative (slope)
        df[f"{col}_slope"] = df[col].diff() / dt_seconds

        # Second derivative (acceleration)
        df[f"{col}_accel"] = df[f"{col}_slope"].diff() / dt_seconds

    return df


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_stability(
    df: pd.DataFrame,
    window: int = config.ROLLING_WINDOW_SIZE,
    min_periods: int = config.MIN_PERIODS_ROLLING,
) -> pd.DataFrame:
    """
    Compute rolling coefficient of variation (CV) as a dimensionless
    stability metric for each signal.

    Lower CV indicates more stable signal behaviour.
    """
    df = df.copy()
    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        rolling = df[col].rolling(window=window, min_periods=min_periods)
        roll_mean = rolling.mean()
        roll_std = rolling.std()
        # Numerical guard against zero-mean segments
        safe_mean = roll_mean.replace(0, np.nan)
        df[f"{col}_stability_cv"] = (roll_std / safe_mean).abs()
    return df


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in sequence.

    Order
    -----
    1. Inter-sample deltas
    2. Z-score normalisation
    3. Rolling statistics (mean, variance)
    4. Signal derivatives (slope, acceleration)
    5. Stability coefficients

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned physiological DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    logger.info("Starting feature engineering on %d samples.", len(df))
    df = compute_deltas(df)
    df = compute_zscores(df)
    df = compute_rolling_stats(df)
    df = compute_derivatives(df)
    df = compute_stability(df)

    # Drop the first few rows where diff/rolling yield NaN
    initial_len = len(df)
    df = df.dropna(subset=[f"{config.SIGNAL_NAMES[0]}_delta"], how="any")
    logger.info(
        "Feature engineering complete. %d -> %d samples (dropped %d lead-in rows).",
        initial_len, len(df), initial_len - len(df),
    )
    return df
