"""
NeuroSense AIoT — ML / Computational Inference Layer
------------------------------------------------------
Implements anomaly detection using Isolation Forest trained on a baseline
segment.  Produces per-sample deviation labels and continuous anomaly
scores suitable for downstream explainability.
"""

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature selection for ML input
# ---------------------------------------------------------------------------

def _select_ml_features(df: pd.DataFrame) -> List[str]:
    """
    Determine which columns to feed into the anomaly detection model.

    Uses primary signals plus their z-score normalised counterparts and
    rolling variance columns when available.
    """
    candidates = []
    for col in config.SIGNAL_NAMES:
        if col in df.columns:
            candidates.append(col)
        zscore_col = f"{col}_zscore"
        if zscore_col in df.columns:
            candidates.append(zscore_col)
        roll_var_col = f"{col}_roll_var"
        if roll_var_col in df.columns:
            candidates.append(roll_var_col)
    return candidates


# ---------------------------------------------------------------------------
# Isolation Forest wrapper
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Encapsulates training, prediction, and scoring using an Isolation Forest.

    The model is fitted on a configurable baseline fraction of the data
    (default: first 50 %) and then applied to the entire recording, enabling
    detection of deviations from the subject's own baseline.

    Attributes
    ----------
    model : IsolationForest
        Trained sklearn Isolation Forest instance.
    scaler : StandardScaler
        Fitted scaler (re-used at inference to ensure consistent transform).
    feature_cols : list[str]
        Column names used as model input.
    """

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=config.ISOLATION_FOREST_N_ESTIMATORS,
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=config.ISOLATION_FOREST_RANDOM_STATE,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []

    # --------------------------------------------------------------------- #
    # Training
    # --------------------------------------------------------------------- #

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on the baseline segment of the data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-enriched physiological DataFrame.

        Returns
        -------
        self
        """
        self.feature_cols = _select_ml_features(df)
        if not self.feature_cols:
            raise ValueError("No valid ML features found in the DataFrame.")

        baseline_end = int(len(df) * config.BASELINE_FRACTION)
        baseline = df.iloc[:baseline_end][self.feature_cols].dropna()

        if baseline.empty:
            raise ValueError("Baseline segment is empty after dropping NaN.")

        logger.info(
            "Training Isolation Forest on %d baseline samples with %d features.",
            len(baseline), len(self.feature_cols),
        )

        X_baseline = self.scaler.fit_transform(baseline.values)
        self.model.fit(X_baseline)
        return self

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on the full dataset.

        Returns the original DataFrame augmented with:
        - ``deviation_label``  : 1 (normal) or -1 (anomaly)
        - ``deviation_score``  : continuous anomaly score (lower = more anomalous)
        - ``is_anomaly``       : boolean convenience column

        Parameters
        ----------
        df : pd.DataFrame
            Feature-enriched physiological DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        labels = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        df["deviation_label"] = labels
        df["deviation_score"] = scores
        df["is_anomaly"] = labels == -1

        n_anomalies = int((labels == -1).sum())
        logger.info(
            "Inference complete: %d / %d samples flagged as anomalous (%.1f%%).",
            n_anomalies, len(df), 100.0 * n_anomalies / max(len(df), 1),
        )
        return df


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def run_anomaly_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, AnomalyDetector]:
    """
    Train an anomaly detector on the baseline segment and predict across
    the full recording.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched physiological DataFrame.

    Returns
    -------
    df_scored : pd.DataFrame
        DataFrame with deviation labels and scores.
    detector : AnomalyDetector
        Trained detector (retained for downstream explainability).
    """
    detector = AnomalyDetector()
    detector.fit(df)
    df_scored = detector.predict(df)
    return df_scored, detector
