"""
NeuroSense AIoT — Configuration Module
---------------------------------------
Centralizes all system constants, ThingSpeak credentials, physiological
bounds, feature engineering parameters, and visualization themes.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# ThingSpeak API Configuration
# ---------------------------------------------------------------------------
THINGSPEAK_CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID", "YOUR_CHANNEL_ID")
THINGSPEAK_READ_API_KEY = os.getenv("THINGSPEAK_READ_API_KEY", "YOUR_READ_API_KEY")
THINGSPEAK_BASE_URL = "https://api.thingspeak.com"
THINGSPEAK_NUM_RESULTS = 8000  # Maximum entries per request
THINGSPEAK_TIMEOUT_SECONDS = 30
THINGSPEAK_MAX_RETRIES = 5
THINGSPEAK_RETRY_BACKOFF_FACTOR = 1.5  # Exponential backoff multiplier

# ---------------------------------------------------------------------------
# Field Mapping — ThingSpeak field indices to readable signal names
# ---------------------------------------------------------------------------
FIELD_MAP = {
    "field1": "HR",
    "field2": "RMSSD",
    "field3": "SkinTemp",
    "field4": "Moisture",
    "field5": "StressScore",
}

SIGNAL_NAMES = list(FIELD_MAP.values())
TIMESTAMP_COLUMN = "created_at"

# ---------------------------------------------------------------------------
# Physiological Validity Bounds
# ---------------------------------------------------------------------------
# These ranges define plausible physiological values.  Samples outside these
# bounds are flagged for removal during the cleaning stage.
PHYSIOLOGICAL_BOUNDS = {
    "HR":          (30.0, 220.0),       # beats per minute
    "RMSSD":       (5.0, 300.0),        # milliseconds
    "SkinTemp":    (20.0, 42.0),        # degrees Celsius
    "Moisture":    (0.0, 100.0),        # percent
    "StressScore": (0.0, 100.0),        # heuristic score
}

# ---------------------------------------------------------------------------
# Feature Engineering Parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW_SIZE = 10               # Samples for rolling statistics
MIN_PERIODS_ROLLING = 3                # Minimum valid observations in window
Z_SCORE_EPSILON = 1e-8                 # Prevents division-by-zero in z-score

# ---------------------------------------------------------------------------
# ML / Anomaly Detection Configuration
# ---------------------------------------------------------------------------
ISOLATION_FOREST_CONTAMINATION = 0.10  # Expected proportion of anomalies
ISOLATION_FOREST_N_ESTIMATORS = 200
ISOLATION_FOREST_RANDOM_STATE = 42
BASELINE_FRACTION = 0.5               # First N% of data used as baseline

# ---------------------------------------------------------------------------
# Visualization Theme — Plotly-compatible styling
# ---------------------------------------------------------------------------
VIZ_TEMPLATE = "plotly_dark"
VIZ_COLOR_PALETTE = [
    "#00d4ff",  # cyan
    "#ff6ec7",  # pink
    "#39ff14",  # neon green
    "#ffaa00",  # amber
    "#bf5fff",  # purple
]
VIZ_ANOMALY_COLOR = "#ff3333"
VIZ_NORMAL_COLOR = "#336699"
VIZ_BACKGROUND_COLOR = "#0e1117"
VIZ_GRID_COLOR = "#1e2530"
VIZ_FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, sans-serif"
VIZ_TITLE_FONT_SIZE = 20
VIZ_AXIS_FONT_SIZE = 12
VIZ_PLOT_HEIGHT = 600
VIZ_PLOT_WIDTH = 1200

# ---------------------------------------------------------------------------
# Explainability Thresholds
# ---------------------------------------------------------------------------
DEVIATION_THRESHOLD_SIGMA = 1.5       # Z-score sigma threshold for flagging
CONTRIBUTOR_TOP_N = 5                 # Top N contributors to report

# ---------------------------------------------------------------------------
# Insight Generation — Physiological Rule Thresholds
# ---------------------------------------------------------------------------
INSIGHT_HR_ELEVATED = 100.0           # BPM above which HR is "elevated"
INSIGHT_HR_LOW = 55.0                 # BPM below which HR is "bradycardic"
INSIGHT_RMSSD_SUPPRESSED = 20.0       # ms below which autonomic suppression
INSIGHT_TEMP_DROP = -0.5              # Celsius delta for vasoconstriction
INSIGHT_TEMP_RISE = 0.5               # Celsius delta for peripheral flush
INSIGHT_MOISTURE_SURGE = 15.0         # Percent delta for sudomotor burst
INSIGHT_STRESS_HIGH = 70.0            # Heuristic stress threshold

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
FIGURE_DIR = f"{OUTPUT_DIR}/figures"
REPORT_DIR = f"{OUTPUT_DIR}/reports"
