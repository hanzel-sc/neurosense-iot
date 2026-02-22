"""
NeuroSense AIoT — Explainability Layer
----------------------------------------
Generates interpretable, quantitative explanations for anomaly detections
by decomposing the deviation score into per-signal contributions.
"""

import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample contribution decomposition
# ---------------------------------------------------------------------------

def _compute_signal_contributions(
    row: pd.Series,
    baseline_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    For a single sample, compute each signal's contribution to the overall
    deviation as the absolute z-score relative to the baseline distribution.

    Parameters
    ----------
    row : pd.Series
        A single row from the scored DataFrame.
    baseline_stats : dict
        Per-signal baseline mean and std from the training segment.

    Returns
    -------
    dict
        Signal name to absolute z-deviation mapping.
    """
    contributions: Dict[str, float] = {}
    for col in config.SIGNAL_NAMES:
        if col not in row.index or col not in baseline_stats:
            continue
        mean = baseline_stats[col]["mean"]
        std = baseline_stats[col]["std"]
        if std < config.Z_SCORE_EPSILON:
            std = config.Z_SCORE_EPSILON
        z = abs((row[col] - mean) / std)
        contributions[col] = round(float(z), 4)
    return contributions


def compute_baseline_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute per-signal mean and std from the baseline segment.

    This mirrors the training window used by the anomaly detector so that
    explainability metrics are aligned with the model's frame of reference.
    """
    baseline_end = int(len(df) * config.BASELINE_FRACTION)
    baseline = df.iloc[:baseline_end]
    stats: Dict[str, Dict[str, float]] = {}
    for col in config.SIGNAL_NAMES:
        if col not in baseline.columns:
            continue
        series = baseline[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return stats


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

def explain_anomalies(
    df: pd.DataFrame,
    top_n: int = config.CONTRIBUTOR_TOP_N,
) -> List[Dict[str, Any]]:
    """
    Generate human-readable explanations for every detected anomaly.

    Each explanation includes:
    - timestamp of the anomalous sample
    - overall deviation score
    - ranked list of contributing signals with their z-deviation magnitude
    - a natural-language summary sentence

    Parameters
    ----------
    df : pd.DataFrame
        Scored DataFrame containing ``is_anomaly`` and ``deviation_score``.
    top_n : int
        How many top-contributing signals to include per anomaly.

    Returns
    -------
    list[dict]
        One explanation dict per anomalous sample.
    """
    if "is_anomaly" not in df.columns:
        logger.warning("No anomaly labels found; skipping explainability.")
        return []

    anomalies = df[df["is_anomaly"]]
    if anomalies.empty:
        logger.info("No anomalies detected; nothing to explain.")
        return []

    baseline_stats = compute_baseline_stats(df)
    explanations: List[Dict[str, Any]] = []

    for idx, row in anomalies.iterrows():
        contributions = _compute_signal_contributions(row, baseline_stats)
        # Sort by descending contribution
        ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        top_contributors = ranked[:top_n]

        # Build summary sentence
        primary = top_contributors[0] if top_contributors else ("N/A", 0.0)
        summary = (
            f"Deviation Score = {row.get('deviation_score', 'N/A'):.4f}. "
            f"Primary contributor: {primary[0]} (z = {primary[1]:.2f}). "
        )
        if len(top_contributors) > 1:
            secondary_parts = [
                f"{name} (z = {val:.2f})" for name, val in top_contributors[1:]
            ]
            summary += "Additional contributors: " + ", ".join(secondary_parts) + "."

        explanation = {
            "timestamp": str(idx),
            "deviation_score": round(float(row.get("deviation_score", 0)), 4),
            "contributors": {name: val for name, val in top_contributors},
            "summary": summary,
        }
        explanations.append(explanation)

    logger.info("Generated explanations for %d anomalous samples.", len(explanations))
    return explanations


# ---------------------------------------------------------------------------
# Formatted report
# ---------------------------------------------------------------------------

def format_explanation_report(explanations: List[Dict[str, Any]]) -> str:
    """
    Render a multi-line, human-readable explainability report suitable
    for console output or text-file persistence.
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("NEUROSENSE EXPLAINABILITY REPORT")
    lines.append("=" * 72)
    lines.append(f"Total anomalous events: {len(explanations)}")
    lines.append("")

    for i, exp in enumerate(explanations, 1):
        lines.append(f"--- Event {i} [{exp['timestamp']}] ---")
        lines.append(f"  Deviation Score : {exp['deviation_score']:.4f}")
        lines.append("  Contributors:")
        for name, val in exp["contributors"].items():
            bar = "#" * min(int(val * 4), 40)
            lines.append(f"    {name:>12s} : z = {val:.2f}  {bar}")
        lines.append(f"  Summary: {exp['summary']}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
