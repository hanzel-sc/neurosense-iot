"""
NeuroSense AIoT — Insight Generation Engine
----------------------------------------------
Applies rule-based physiological reasoning to generate research-style
narrative interpretations of the current physiological state and temporal
trends.
"""

import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual rule evaluators
# ---------------------------------------------------------------------------

def _check_cardiovascular_activation(row: pd.Series) -> str | None:
    """Detect elevated heart rate indicative of sympathetic activation."""
    if "HR" not in row.index:
        return None
    hr = row["HR"]
    if hr > config.INSIGHT_HR_ELEVATED:
        return (
            f"Cardiovascular activation detected: HR = {hr:.1f} BPM "
            f"exceeds resting threshold ({config.INSIGHT_HR_ELEVATED} BPM). "
            "This pattern is consistent with sympathetic nervous system "
            "upregulation or physical exertion."
        )
    return None


def _check_bradycardic_signature(row: pd.Series) -> str | None:
    """Detect abnormally low heart rate."""
    if "HR" not in row.index:
        return None
    hr = row["HR"]
    if hr < config.INSIGHT_HR_LOW:
        return (
            f"Bradycardic signature observed: HR = {hr:.1f} BPM is below "
            f"the resting lower bound ({config.INSIGHT_HR_LOW} BPM). "
            "This may indicate enhanced vagal tone, deep relaxation, or "
            "a sensor contact artefact."
        )
    return None


def _check_autonomic_suppression(row: pd.Series) -> str | None:
    """Detect RMSSD suppression indicative of reduced parasympathetic tone."""
    if "RMSSD" not in row.index:
        return None
    rmssd = row["RMSSD"]
    if rmssd < config.INSIGHT_RMSSD_SUPPRESSED:
        return (
            f"Autonomic suppression signature: RMSSD = {rmssd:.1f} ms is "
            f"below {config.INSIGHT_RMSSD_SUPPRESSED} ms, suggesting "
            "reduced parasympathetic modulation. This is commonly observed "
            "during acute stress, cognitive load, or sympathetic dominance."
        )
    return None


def _check_vasoconstriction(row: pd.Series) -> str | None:
    """Detect peripheral vasoconstriction from skin temperature drop."""
    if "SkinTemp_delta" not in row.index:
        return None
    delta = row.get("SkinTemp_delta", 0)
    if delta is not None and delta < config.INSIGHT_TEMP_DROP:
        return (
            f"Peripheral vasoconstriction trend: skin temperature dropped by "
            f"{abs(delta):.2f} C. Peripheral cooling may reflect sympathetically "
            "mediated vascular redistribution or environmental cold exposure."
        )
    return None


def _check_peripheral_flush(row: pd.Series) -> str | None:
    """Detect temperature rise suggesting peripheral vasodilation."""
    if "SkinTemp_delta" not in row.index:
        return None
    delta = row.get("SkinTemp_delta", 0)
    if delta is not None and delta > config.INSIGHT_TEMP_RISE:
        return (
            f"Peripheral vasodilation detected: skin temperature rose by "
            f"{delta:.2f} C, consistent with active thermoregulatory "
            "vasodilation or post-exercise recovery."
        )
    return None


def _check_sudomotor_burst(row: pd.Series) -> str | None:
    """Detect rapid moisture increase consistent with sudomotor activation."""
    if "Moisture_delta" not in row.index:
        return None
    delta = row.get("Moisture_delta", 0)
    if delta is not None and delta > config.INSIGHT_MOISTURE_SURGE:
        return (
            f"Sudomotor burst detected: moisture surged by {delta:.1f}%, "
            "indicating rapid eccrine gland activation. This pattern is "
            "a hallmark of acute sympathetic arousal."
        )
    return None


def _check_high_stress(row: pd.Series) -> str | None:
    """Detect elevated heuristic stress score."""
    if "StressScore" not in row.index:
        return None
    stress = row["StressScore"]
    if stress > config.INSIGHT_STRESS_HIGH:
        return (
            f"Elevated composite stress index: StressScore = {stress:.1f} "
            f"exceeds the high-stress threshold ({config.INSIGHT_STRESS_HIGH}). "
            "Multi-modal physiological strain is indicated."
        )
    return None


def _check_instability(row: pd.Series) -> str | None:
    """
    Detect multi-signal instability from rolling variance columns.
    A sample is considered unstable if more than half of the available
    signals have elevated rolling variance (above the 90th percentile
    computed earlier by the feature engineering layer).
    """
    var_cols = [f"{col}_roll_var" for col in config.SIGNAL_NAMES if f"{col}_roll_var" in row.index]
    if not var_cols:
        return None

    # Heuristic: flag if any variance is non-finite or suspiciously large
    elevated_count = sum(1 for c in var_cols if row[c] is not None and not np.isnan(row[c]) and row[c] > 0)
    total = len(var_cols)

    # This rule fires contextually when called on anomalous rows
    if elevated_count >= total * 0.6:
        return (
            "Physiological instability signature: multiple signals exhibit "
            "elevated rolling variance simultaneously, suggesting a transient "
            "perturbation across cardiovascular, thermoregulatory, and "
            "electrodermal subsystems."
        )
    return None


# ---------------------------------------------------------------------------
# Aggregated rule engine
# ---------------------------------------------------------------------------

RULE_FUNCTIONS = [
    _check_cardiovascular_activation,
    _check_bradycardic_signature,
    _check_autonomic_suppression,
    _check_vasoconstriction,
    _check_peripheral_flush,
    _check_sudomotor_burst,
    _check_high_stress,
    _check_instability,
]


def generate_insights_for_row(row: pd.Series) -> List[str]:
    """
    Evaluate all physiological rules against a single sample and collect
    triggered insight strings.
    """
    insights: List[str] = []
    for rule_fn in RULE_FUNCTIONS:
        result = rule_fn(row)
        if result is not None:
            insights.append(result)
    return insights


def generate_insights(df: pd.DataFrame, anomalies_only: bool = True) -> List[Dict[str, Any]]:
    """
    Generate insights across the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Scored, feature-enriched DataFrame.
    anomalies_only : bool
        If True, only evaluate rules for samples flagged as anomalous.
        If False, evaluate all samples (may produce verbose output).

    Returns
    -------
    list[dict]
        Each dict contains 'timestamp' and a 'insights' list of strings.
    """
    if anomalies_only and "is_anomaly" in df.columns:
        subset = df[df["is_anomaly"]]
    else:
        subset = df

    all_insights: List[Dict[str, Any]] = []
    for idx, row in subset.iterrows():
        row_insights = generate_insights_for_row(row)
        if row_insights:
            all_insights.append({
                "timestamp": str(idx),
                "insights": row_insights,
            })

    logger.info("Generated insights for %d samples.", len(all_insights))
    return all_insights


def format_insight_report(insight_list: List[Dict[str, Any]]) -> str:
    """Render insights as a human-readable console/text report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("NEUROSENSE PHYSIOLOGICAL INSIGHT REPORT")
    lines.append("=" * 72)
    lines.append(f"Samples with triggered insights: {len(insight_list)}")
    lines.append("")

    for entry in insight_list:
        lines.append(f"[{entry['timestamp']}]")
        for insight in entry["insights"]:
            lines.append(f"  - {insight}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
