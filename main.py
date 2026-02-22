"""
NeuroSense AIoT — Pipeline Orchestrator
-----------------------------------------
Entry point that wires every pipeline stage together: data acquisition,
cleaning, feature engineering, quantitative analytics, ML inference,
visualisation, explainability, insight generation, and comparative
analysis.

Usage
-----
    python main.py

Ensure ThingSpeak credentials are configured in config.py before running.
"""

import json
import logging
import os
import sys
from typing import Dict, Any

import config
import data_acquisition
import data_cleaning
import feature_engineering
import analytics
import ml_inference
import visualization
import explainability
import insights
import comparative
import dashboard

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neurosense.main")


# ---------------------------------------------------------------------------
# Output persistence helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (config.OUTPUT_DIR, config.FIGURE_DIR, config.REPORT_DIR):
        os.makedirs(d, exist_ok=True)


def _save_json(data: Any, filename: str) -> None:
    """Serialise a dict/list to a JSON file in the reports directory."""
    path = os.path.join(config.REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Report saved: %s", path)


def _save_text(text: str, filename: str) -> None:
    """Write a plain-text report to the reports directory."""
    path = os.path.join(config.REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Report saved: %s", path)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_acquire() -> "pd.DataFrame":
    """Stage 1 — Data Acquisition from ThingSpeak."""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA ACQUISITION")
    logger.info("=" * 60)
    df = data_acquisition.fetch_data(mode="json")
    if df.empty:
        logger.error("No data retrieved from ThingSpeak. Aborting.")
        sys.exit(1)
    logger.info("Acquired %d raw records.", len(df))
    return df


def stage_clean(df: "pd.DataFrame") -> "tuple[pd.DataFrame, Dict[str, Any]]":
    """Stage 2 — Data Cleaning & Validation."""
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA CLEANING & VALIDATION")
    logger.info("=" * 60)
    df_clean, quality = data_cleaning.clean_data(df)
    _save_json(quality, "data_quality_metrics.json")
    logger.info("Clean dataset: %d samples.", len(df_clean))
    return df_clean, quality


def stage_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Stage 3 — Feature Engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)
    df_feat = feature_engineering.engineer_features(df)
    logger.info("Engineered feature set: %d columns.", len(df_feat.columns))
    return df_feat


def stage_analytics(df: "pd.DataFrame") -> Dict[str, Any]:
    """Stage 4 — Quantitative Analytics."""
    logger.info("=" * 60)
    logger.info("STAGE 4: QUANTITATIVE ANALYTICS")
    logger.info("=" * 60)
    report = analytics.run_analytics(df)
    _save_json(report, "quantitative_analytics.json")
    return report


def stage_inference(df: "pd.DataFrame") -> "tuple[pd.DataFrame, ml_inference.AnomalyDetector]":
    """Stage 5 — ML Anomaly Detection."""
    logger.info("=" * 60)
    logger.info("STAGE 5: ML / COMPUTATIONAL INFERENCE")
    logger.info("=" * 60)
    df_scored, detector = ml_inference.run_anomaly_detection(df)
    return df_scored, detector


def stage_visualise(df: "pd.DataFrame") -> None:
    """Stage 6 — Visualisation."""
    logger.info("=" * 60)
    logger.info("STAGE 6: VISUALIZATION")
    logger.info("=" * 60)
    visualization.generate_all_visualisations(df)


def stage_explain(df: "pd.DataFrame") -> None:
    """Stage 7 — Explainability."""
    logger.info("=" * 60)
    logger.info("STAGE 7: EXPLAINABILITY")
    logger.info("=" * 60)
    explanations = explainability.explain_anomalies(df)
    _save_json(explanations, "explainability_report.json")
    text_report = explainability.format_explanation_report(explanations)
    _save_text(text_report, "explainability_report.txt")
    print(text_report)


def stage_insights(df: "pd.DataFrame") -> None:
    """Stage 8 — Insight Generation."""
    logger.info("=" * 60)
    logger.info("STAGE 8: INSIGHT GENERATION")
    logger.info("=" * 60)
    insight_list = insights.generate_insights(df, anomalies_only=True)
    _save_json(insight_list, "physiological_insights.json")
    text_report = insights.format_insight_report(insight_list)
    _save_text(text_report, "physiological_insights.txt")
    print(text_report)


def stage_comparative(df: "pd.DataFrame") -> None:
    """Stage 9 — Comparative / Research Analysis."""
    logger.info("=" * 60)
    logger.info("STAGE 9: COMPARATIVE ANALYSIS")
    logger.info("=" * 60)

    # Baseline vs Activation
    ba_comparison = comparative.compare_baseline_activation(df)
    _save_json(ba_comparison, "baseline_vs_activation.json")

    # Response Magnitude
    magnitudes = comparative.compute_response_magnitude(df)
    _save_json(magnitudes, "response_magnitudes.json")

    # Formatted text report
    text_report = comparative.format_comparison_report(ba_comparison, magnitudes)
    _save_text(text_report, "comparative_report.txt")
    print(text_report)

    # Multi-subject comparison is available but requires multiple DataFrames.
    # Demonstration: treat baseline and activation halves as two "subjects".
    baseline, activation = comparative.split_baseline_activation(df)
    multi_subject = comparative.compare_subjects({
        "baseline_segment": baseline,
        "activation_segment": activation,
    })
    _save_json(multi_subject, "multi_segment_comparison.json")

    # Temporal response comparison across pseudo-subjects
    temporal_cmp = comparative.compare_temporal_responses(
        {"baseline_segment": baseline, "activation_segment": activation},
        signal="HR",
    )
    _save_json(temporal_cmp, "temporal_response_comparison.json")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stage_dashboard(df: "pd.DataFrame") -> None:
    """Stage 10 — Dashboard Generation."""
    logger.info("=" * 60)
    logger.info("STAGE 10: DASHBOARD GENERATION")
    logger.info("=" * 60)
    path = dashboard.save_dashboard(df)
    logger.info("Dashboard ready: %s", path)


def main() -> None:
    """
    Execute the complete NeuroSense AIoT pipeline end-to-end.

    Pipeline
    --------
    1. Data Acquisition
    2. Data Cleaning & Validation
    3. Feature Engineering
    4. Quantitative Analytics
    5. ML Anomaly Detection
    6. Visualisation
    7. Explainability
    8. Insight Generation
    9. Comparative Analysis
    10. Dashboard Generation
    """
    _ensure_dirs()

    logger.info("NeuroSense AIoT Pipeline — Starting")
    logger.info("=" * 60)

    # Stage 1 — Acquire
    df_raw = stage_acquire()

    # Stage 2 — Clean
    df_clean, quality = stage_clean(df_raw)

    # Stage 3 — Engineer features
    df_feat = stage_features(df_clean)

    # Stage 4 — Quantitative analytics
    analytics_report = stage_analytics(df_feat)

    # Stage 5 — ML inference
    df_scored, detector = stage_inference(df_feat)

    # Stage 6 — Visualise
    stage_visualise(df_scored)

    # Stage 7 — Explain
    stage_explain(df_scored)

    # Stage 8 — Insights
    stage_insights(df_scored)

    # Stage 9 — Comparative analysis
    stage_comparative(df_scored)

    # Stage 10 — Dashboard
    stage_dashboard(df_scored)

    logger.info("=" * 60)
    logger.info("NeuroSense AIoT Pipeline — Complete")
    logger.info("All reports saved to: %s", config.REPORT_DIR)
    logger.info("All figures saved to: %s", config.FIGURE_DIR)
    logger.info("Dashboard: %s", os.path.join(config.OUTPUT_DIR, "dashboard.html"))


if __name__ == "__main__":
    main()
