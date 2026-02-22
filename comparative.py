"""
NeuroSense AIoT — Comparative / Research Layer
------------------------------------------------
Provides functions for multi-subject comparison, baseline-vs-activation
segmentation, response magnitude analysis, and temporal response comparison
across recording sessions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def split_baseline_activation(
    df: pd.DataFrame,
    baseline_frac: float = config.BASELINE_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a recording into baseline and activation segments.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    baseline_frac : float
        Fraction of total samples that constitute the baseline.

    Returns
    -------
    baseline : pd.DataFrame
    activation : pd.DataFrame
    """
    split_idx = int(len(df) * baseline_frac)
    baseline = df.iloc[:split_idx].copy()
    activation = df.iloc[split_idx:].copy()
    logger.info(
        "Split %d samples -> baseline (%d), activation (%d).",
        len(df), len(baseline), len(activation),
    )
    return baseline, activation


# ---------------------------------------------------------------------------
# Baseline vs Activation comparison
# ---------------------------------------------------------------------------

def compare_baseline_activation(
    df: pd.DataFrame,
    baseline_frac: float = config.BASELINE_FRACTION,
) -> Dict[str, Any]:
    """
    Compare signal statistics between baseline and activation segments.

    For each primary signal computes:
    - Mean difference (activation - baseline)
    - Cohen's d effect size
    - Welch's t-test p-value
    - Percent change

    Returns
    -------
    dict
        Per-signal comparison metrics.
    """
    baseline, activation = split_baseline_activation(df, baseline_frac)
    comparison: Dict[str, Any] = {}

    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        b = baseline[col].dropna()
        a = activation[col].dropna()
        if b.empty or a.empty:
            continue

        mean_b, mean_a = float(b.mean()), float(a.mean())
        std_b, std_a = float(b.std()), float(a.std())

        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(
            ((len(b) - 1) * std_b ** 2 + (len(a) - 1) * std_a ** 2)
            / max(len(b) + len(a) - 2, 1)
        )
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Welch's t-test
        t_stat, p_value = sp_stats.ttest_ind(b, a, equal_var=False)

        pct_change = (
            (mean_a - mean_b) / abs(mean_b) * 100.0
            if abs(mean_b) > config.Z_SCORE_EPSILON else 0.0
        )

        comparison[col] = {
            "baseline_mean": round(mean_b, 4),
            "activation_mean": round(mean_a, 4),
            "mean_diff": round(mean_a - mean_b, 4),
            "cohens_d": round(float(cohens_d), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "pct_change": round(float(pct_change), 2),
        }

    return comparison


# ---------------------------------------------------------------------------
# Response magnitude analysis
# ---------------------------------------------------------------------------

def compute_response_magnitude(
    df: pd.DataFrame,
    baseline_frac: float = config.BASELINE_FRACTION,
) -> Dict[str, Dict[str, float]]:
    """
    Quantify the magnitude of physiological response for each signal as
    the peak deviation from baseline mean normalised by baseline std.

    Returns
    -------
    dict
        Per-signal response magnitude metrics.
    """
    baseline, activation = split_baseline_activation(df, baseline_frac)
    magnitudes: Dict[str, Dict[str, float]] = {}

    for col in config.SIGNAL_NAMES:
        if col not in df.columns:
            continue
        b = baseline[col].dropna()
        a = activation[col].dropna()
        if b.empty or a.empty:
            continue

        baseline_mean = float(b.mean())
        baseline_std = float(b.std())
        if baseline_std < config.Z_SCORE_EPSILON:
            baseline_std = config.Z_SCORE_EPSILON

        # Peak deviation from baseline mean
        deviations = (a - baseline_mean).abs()
        peak_deviation = float(deviations.max())
        mean_deviation = float(deviations.mean())

        magnitudes[col] = {
            "peak_deviation_raw": round(peak_deviation, 4),
            "peak_deviation_zscore": round(peak_deviation / baseline_std, 4),
            "mean_deviation_raw": round(mean_deviation, 4),
            "mean_deviation_zscore": round(mean_deviation / baseline_std, 4),
        }

    return magnitudes


# ---------------------------------------------------------------------------
# Multi-subject comparison
# ---------------------------------------------------------------------------

def compare_subjects(
    subject_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Compare summary statistics across multiple subject recordings.

    Parameters
    ----------
    subject_dfs : dict
        Mapping of subject ID/label to their respective DataFrame.

    Returns
    -------
    dict
        Per-signal, per-subject summary and cross-subject ANOVA p-value.
    """
    results: Dict[str, Any] = {}

    for col in config.SIGNAL_NAMES:
        per_subject: Dict[str, Dict[str, float]] = {}
        groups: List[np.ndarray] = []

        for subj_id, df in subject_dfs.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            per_subject[subj_id] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "median": round(float(series.median()), 4),
                "n_samples": int(len(series)),
            }
            groups.append(series.values)

        # One-way ANOVA across subjects (requires at least 2 groups)
        anova_p: Optional[float] = None
        if len(groups) >= 2:
            _, p_val = sp_stats.f_oneway(*groups)
            anova_p = round(float(p_val), 6)

        results[col] = {
            "per_subject": per_subject,
            "anova_p_value": anova_p,
        }

    return results


# ---------------------------------------------------------------------------
# Temporal response comparison
# ---------------------------------------------------------------------------

def compare_temporal_responses(
    subject_dfs: Dict[str, pd.DataFrame],
    signal: str = "HR",
) -> Dict[str, Dict[str, float]]:
    """
    Compare temporal response dynamics across subjects for a given signal.

    Computes response velocity, volatility, and recovery slope per subject
    and returns a comparative summary.
    """
    results: Dict[str, Dict[str, float]] = {}

    for subj_id, df in subject_dfs.items():
        slope_col = f"{signal}_slope"
        if slope_col not in df.columns or signal not in df.columns:
            continue

        slope_series = df[slope_col].dropna()
        signal_series = df[signal].dropna()

        velocity = float(slope_series.abs().mean()) if not slope_series.empty else np.nan
        volatility = float(slope_series.std()) if not slope_series.empty else np.nan

        # Recovery slope from last 25 % of signal
        tail = signal_series.iloc[max(1, int(len(signal_series) * 0.75)):]
        if len(tail) >= 2:
            x = np.arange(len(tail), dtype=np.float64)
            rec_slope, _, _, _, _ = sp_stats.linregress(x, tail.values)
        else:
            rec_slope = np.nan

        results[subj_id] = {
            "response_velocity": round(velocity, 6) if not np.isnan(velocity) else None,
            "signal_volatility": round(volatility, 6) if not np.isnan(volatility) else None,
            "recovery_slope": round(float(rec_slope), 6) if not np.isnan(rec_slope) else None,
        }

    return results


# ---------------------------------------------------------------------------
# Formatted comparison report
# ---------------------------------------------------------------------------

def format_comparison_report(
    baseline_activation: Dict[str, Any],
    response_magnitudes: Dict[str, Dict[str, float]],
) -> str:
    """Render baseline-vs-activation comparison as a text report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("NEUROSENSE COMPARATIVE ANALYSIS REPORT")
    lines.append("=" * 72)

    lines.append("\n-- Baseline vs Activation --\n")
    for col, metrics in baseline_activation.items():
        lines.append(f"  {col}:")
        for k, v in metrics.items():
            lines.append(f"    {k:>22s} : {v}")
        lines.append("")

    lines.append("-- Response Magnitudes --\n")
    for col, metrics in response_magnitudes.items():
        lines.append(f"  {col}:")
        for k, v in metrics.items():
            lines.append(f"    {k:>25s} : {v}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
