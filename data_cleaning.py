"""
NeuroSense AIoT — Data Cleaning & Validation Layer
----------------------------------------------------
Removes physiologically implausible values, handles missing data, and
computes data-quality metrics that quantify signal integrity before
downstream analytics.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _flag_out_of_range(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Replace values outside physiological bounds with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw signal DataFrame.
    bounds : dict
        Mapping of column name to (min, max) acceptable range.

    Returns
    -------
    pd.DataFrame
        DataFrame with out-of-range entries set to NaN.
    """
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_flagged = mask.sum()
        if n_flagged > 0:
            logger.info(
                "Column '%s': %d samples outside [%.1f, %.1f] flagged as NaN.",
                col, n_flagged, lo, hi,
            )
            df.loc[mask, col] = np.nan
    return df


def _remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with duplicate timestamps, keeping the last occurrence."""
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate-timestamp rows.", removed)
    return df


# ---------------------------------------------------------------------------
# Data quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> Dict[str, object]:
    """
    Compute data-quality summary statistics comparing raw and cleaned data.

    Returns
    -------
    dict
        Keys include per-signal missing percentages, overall row retention
        rate, and signal stability (coefficient of variation).
    """
    metrics: Dict[str, object] = {}
    total_raw = len(df_raw)
    total_clean = len(df_clean)

    metrics["total_raw_samples"] = total_raw
    metrics["total_clean_samples"] = total_clean
    metrics["row_retention_pct"] = (
        round(100.0 * total_clean / total_raw, 2) if total_raw > 0 else 0.0
    )

    per_signal: Dict[str, Dict[str, float]] = {}
    for col in config.SIGNAL_NAMES:
        if col not in df_clean.columns:
            continue
        series = df_clean[col]
        n_missing = series.isna().sum()
        n_valid = series.notna().sum()
        mean_val = series.mean()
        std_val = series.std()

        # Coefficient of variation as a stability proxy
        cv = (std_val / mean_val * 100.0) if mean_val != 0 else np.nan

        per_signal[col] = {
            "missing_count": int(n_missing),
            "missing_pct": round(100.0 * n_missing / total_clean, 2) if total_clean > 0 else 0.0,
            "valid_count": int(n_valid),
            "mean": round(float(mean_val), 4) if not np.isnan(mean_val) else None,
            "std": round(float(std_val), 4) if not np.isnan(std_val) else None,
            "coefficient_of_variation_pct": round(float(cv), 2) if not np.isnan(cv) else None,
        }

    metrics["per_signal"] = per_signal
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Execute the full cleaning pipeline.

    Steps
    -----
    1. Remove duplicate timestamps.
    2. Flag physiologically impossible values as NaN.
    3. Forward-fill short NaN gaps (max 3 consecutive), then drop
       remaining rows where any signal is still NaN.
    4. Compute and return quality metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time-indexed physiological DataFrame.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame with valid physiological signals.
    quality : dict
        Data-quality summary metrics.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to clean_data. Returning as-is.")
        return df, {}

    df_raw_snapshot = df.copy()

    # Step 1 — duplicate timestamps
    df = _remove_duplicate_timestamps(df)

    # Step 2 — physiological bounds enforcement
    df = _flag_out_of_range(df, config.PHYSIOLOGICAL_BOUNDS)

    # Step 3 — interpolation for short gaps, then drop residual NaNs
    signal_cols = [c for c in config.SIGNAL_NAMES if c in df.columns]
    df[signal_cols] = df[signal_cols].ffill(limit=3)
    before_drop = len(df)
    df = df.dropna(subset=signal_cols)
    logger.info(
        "Dropped %d rows with residual NaN values after forward-fill.",
        before_drop - len(df),
    )

    # Step 4 — quality report
    quality = compute_quality_metrics(df_raw_snapshot, df)
    logger.info("Data quality metrics: %s", quality)

    return df, quality
