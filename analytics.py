"""
NeuroSense AIoT — Quantitative Analytics Layer
------------------------------------------------
Computes research-grade physiological metrics including descriptive
statistics, temporal dynamics, and cross-signal correlation structures.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal-level descriptive statistics
# ---------------------------------------------------------------------------

def compute_signal_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute descriptive statistics for each primary physiological signal.

    Metrics include mean, standard deviation, median, IQR, skewness,
    kurtosis, and range.
    """
    results: Dict[str, Any] = {}
    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        results[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "median": round(float(series.median()), 4),
            "iqr": round(float(q3 - q1), 4),
            "skewness": round(float(sp_stats.skew(series)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(series)), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "range": round(float(series.max() - series.min()), 4),
        }
    return results


# ---------------------------------------------------------------------------
# Specialised physiological statistics
# ---------------------------------------------------------------------------

def compute_hr_variability(df: pd.DataFrame) -> Dict[str, float]:
    """Heart rate variability proxies: SDNN-like std and pNN-style metric."""
    if "HR" not in df.columns:
        return {}
    hr = df["HR"].dropna()
    hr_diff = hr.diff().dropna().abs()
    return {
        "hr_mean": round(float(hr.mean()), 2),
        "hr_std": round(float(hr.std()), 2),
        "hr_diff_mean": round(float(hr_diff.mean()), 4),
        "hr_diff_std": round(float(hr_diff.std()), 4),
    }


def compute_rmssd_suppression(df: pd.DataFrame) -> Dict[str, float]:
    """Quantify RMSSD suppression magnitude relative to baseline mean."""
    if "RMSSD" not in df.columns:
        return {}
    rmssd = df["RMSSD"].dropna()
    baseline_mean = rmssd.iloc[: max(1, len(rmssd) // 2)].mean()
    activation_mean = rmssd.iloc[len(rmssd) // 2 :].mean()
    suppression = baseline_mean - activation_mean
    return {
        "rmssd_baseline_mean": round(float(baseline_mean), 4),
        "rmssd_activation_mean": round(float(activation_mean), 4),
        "rmssd_suppression_magnitude": round(float(suppression), 4),
    }


def compute_temperature_drift(df: pd.DataFrame) -> Dict[str, float]:
    """Linear temperature drift over the recording window."""
    if "SkinTemp" not in df.columns:
        return {}
    temp = df["SkinTemp"].dropna()
    if len(temp) < 2:
        return {}
    x = np.arange(len(temp), dtype=np.float64)
    slope, intercept, r_value, _, _ = sp_stats.linregress(x, temp.values)
    return {
        "temp_slope_per_sample": round(float(slope), 6),
        "temp_intercept": round(float(intercept), 4),
        "temp_r_squared": round(float(r_value ** 2), 4),
        "temp_total_drift": round(float(slope * len(temp)), 4),
    }


def compute_moisture_variability(df: pd.DataFrame) -> Dict[str, float]:
    """Moisture variability as coefficient of variation."""
    if "Moisture" not in df.columns:
        return {}
    m = df["Moisture"].dropna()
    mean_val = m.mean()
    cv = (m.std() / mean_val * 100.0) if mean_val != 0 else np.nan
    return {
        "moisture_mean": round(float(mean_val), 4),
        "moisture_std": round(float(m.std()), 4),
        "moisture_cv_pct": round(float(cv), 2) if not np.isnan(cv) else None,
    }


# ---------------------------------------------------------------------------
# Temporal dynamics
# ---------------------------------------------------------------------------

def compute_temporal_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute response velocity, signal volatility, and recovery slope for
    each signal.

    Definitions
    -----------
    Response velocity : mean absolute first derivative (rate of change).
    Signal volatility : standard deviation of the first derivative.
    Recovery slope    : linear slope of the last 25 % of the signal
                        (proxy for return-to-baseline behaviour).
    """
    dynamics: Dict[str, Any] = {}
    for col in config.SIGNAL_NAMES:
        slope_col = f"{col}_slope"
        if slope_col not in df.columns:
            continue

        slope_series = df[slope_col].dropna()
        if slope_series.empty:
            continue

        response_velocity = float(slope_series.abs().mean())
        volatility = float(slope_series.std())

        # Recovery slope — last 25 % of the primary signal
        signal = df[col].dropna()
        tail = signal.iloc[max(1, int(len(signal) * 0.75)) :]
        if len(tail) >= 2:
            x = np.arange(len(tail), dtype=np.float64)
            rec_slope, _, _, _, _ = sp_stats.linregress(x, tail.values)
        else:
            rec_slope = np.nan

        dynamics[col] = {
            "response_velocity": round(response_velocity, 6),
            "signal_volatility": round(volatility, 6),
            "recovery_slope": round(float(rec_slope), 6) if not np.isnan(rec_slope) else None,
        }
    return dynamics


# ---------------------------------------------------------------------------
# Cross-signal correlation
# ---------------------------------------------------------------------------

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across all primary signals.

    Returns a square DataFrame of pairwise correlations.
    """
    cols = [c for c in config.SIGNAL_NAMES if c in df.columns]
    corr = df[cols].corr(method="pearson")
    return corr


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute the full quantitative analytics pipeline.

    Returns
    -------
    dict
        Nested dictionary containing:
        - signal_statistics
        - hr_variability
        - rmssd_suppression
        - temperature_drift
        - moisture_variability
        - temporal_dynamics
        - correlation_matrix (as nested dict for serialisation)
    """
    logger.info("Running quantitative analytics on %d samples.", len(df))

    report: Dict[str, Any] = {
        "signal_statistics": compute_signal_statistics(df),
        "hr_variability": compute_hr_variability(df),
        "rmssd_suppression": compute_rmssd_suppression(df),
        "temperature_drift": compute_temperature_drift(df),
        "moisture_variability": compute_moisture_variability(df),
        "temporal_dynamics": compute_temporal_dynamics(df),
        "correlation_matrix": compute_correlation_matrix(df).to_dict(),
    }

    logger.info("Quantitative analytics complete.")
    return report
