"""
NeuroSense AIoT — Visualization Layer
---------------------------------------
Generates interactive, presentation-grade Plotly charts for multivariate
physiological time-series, anomaly overlays, correlation structures, and
signal dynamics.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    """Create the figure output directory if it does not exist."""
    os.makedirs(config.FIGURE_DIR, exist_ok=True)


def _apply_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "Time",
    yaxis_title: str = "",
    height: int = config.VIZ_PLOT_HEIGHT,
    width: int = config.VIZ_PLOT_WIDTH,
) -> go.Figure:
    """Apply a consistent, dark-themed layout to a Plotly figure."""
    fig.update_layout(
        template=config.VIZ_TEMPLATE,
        title=dict(
            text=title,
            font=dict(size=config.VIZ_TITLE_FONT_SIZE, family=config.VIZ_FONT_FAMILY),
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family=config.VIZ_FONT_FAMILY, size=config.VIZ_AXIS_FONT_SIZE),
        paper_bgcolor=config.VIZ_BACKGROUND_COLOR,
        plot_bgcolor=config.VIZ_BACKGROUND_COLOR,
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=config.VIZ_GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=config.VIZ_GRID_COLOR, zeroline=False)
    return fig


def _save_and_show(fig: go.Figure, filename: str) -> None:
    """Persist a figure as interactive HTML and display in browser."""
    _ensure_output_dir()
    path = os.path.join(config.FIGURE_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("Figure saved: %s", path)
    fig.show()


# ---------------------------------------------------------------------------
# 1. Multi-signal time-series plot
# ---------------------------------------------------------------------------

def plot_multisignal_timeseries(df: pd.DataFrame) -> go.Figure:
    """
    Overlay all primary physiological signals on a shared time-axis with
    independent y-axes for each signal group.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Cardiovascular — HR & RMSSD",
            "Thermoregulatory — Skin Temperature",
            "Electrodermal & Stress — Moisture & StressScore",
        ],
    )

    # Row 1: HR and RMSSD
    if "HR" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["HR"],
                name="HR (BPM)", mode="lines",
                line=dict(color=config.VIZ_COLOR_PALETTE[0], width=1.5),
            ),
            row=1, col=1,
        )
    if "RMSSD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["RMSSD"],
                name="RMSSD (ms)", mode="lines",
                line=dict(color=config.VIZ_COLOR_PALETTE[1], width=1.5),
            ),
            row=1, col=1,
        )

    # Row 2: Skin temperature
    if "SkinTemp" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SkinTemp"],
                name="SkinTemp (C)", mode="lines",
                line=dict(color=config.VIZ_COLOR_PALETTE[2], width=1.5),
            ),
            row=2, col=1,
        )

    # Row 3: Moisture and StressScore
    if "Moisture" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Moisture"],
                name="Moisture (%)", mode="lines",
                line=dict(color=config.VIZ_COLOR_PALETTE[3], width=1.5),
            ),
            row=3, col=1,
        )
    if "StressScore" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["StressScore"],
                name="StressScore", mode="lines",
                line=dict(color=config.VIZ_COLOR_PALETTE[4], width=1.5),
            ),
            row=3, col=1,
        )

    fig = _apply_layout(fig, "Multi-Signal Physiological Time-Series", height=900)
    _save_and_show(fig, "multisignal_timeseries.html")
    return fig


# ---------------------------------------------------------------------------
# 2. Normalised signal deviations (z-score overlay)
# ---------------------------------------------------------------------------

def plot_normalised_deviations(df: pd.DataFrame) -> go.Figure:
    """
    Plot z-score normalised versions of all signals on a single axis to
    enable cross-signal deviation comparison.
    """
    fig = go.Figure()
    for i, col in enumerate(config.SIGNAL_NAMES):
        zscore_col = f"{col}_zscore"
        if zscore_col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[zscore_col],
            name=f"{col} (z)",
            mode="lines",
            line=dict(color=config.VIZ_COLOR_PALETTE[i % len(config.VIZ_COLOR_PALETTE)], width=1.4),
        ))

    # Reference bands at +/- threshold
    sigma = config.DEVIATION_THRESHOLD_SIGMA
    fig.add_hline(y=sigma, line_dash="dash", line_color="white", opacity=0.3,
                  annotation_text=f"+{sigma} sigma")
    fig.add_hline(y=-sigma, line_dash="dash", line_color="white", opacity=0.3,
                  annotation_text=f"-{sigma} sigma")

    fig = _apply_layout(fig, "Normalised Signal Deviations (Z-Score)", yaxis_title="Z-Score")
    _save_and_show(fig, "normalised_deviations.html")
    return fig


# ---------------------------------------------------------------------------
# 3. Signal slope / dynamics
# ---------------------------------------------------------------------------

def plot_signal_dynamics(df: pd.DataFrame) -> go.Figure:
    """Plot first-derivative (slope) for all primary signals."""
    fig = go.Figure()
    for i, col in enumerate(config.SIGNAL_NAMES):
        slope_col = f"{col}_slope"
        if slope_col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[slope_col],
            name=f"{col} slope",
            mode="lines",
            line=dict(color=config.VIZ_COLOR_PALETTE[i % len(config.VIZ_COLOR_PALETTE)], width=1.2),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.25)
    fig = _apply_layout(fig, "Signal Rate of Change (First Derivative)", yaxis_title="dSignal / dt")
    _save_and_show(fig, "signal_dynamics.html")
    return fig


# ---------------------------------------------------------------------------
# 4. Rolling variance / stability
# ---------------------------------------------------------------------------

def plot_rolling_variance(df: pd.DataFrame) -> go.Figure:
    """Display rolling variance for all signals to highlight instability windows."""
    fig = go.Figure()
    for i, col in enumerate(config.SIGNAL_NAMES):
        var_col = f"{col}_roll_var"
        if var_col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[var_col],
            name=f"{col} variance",
            mode="lines",
            line=dict(color=config.VIZ_COLOR_PALETTE[i % len(config.VIZ_COLOR_PALETTE)], width=1.3),
            fill="tozeroy",
            opacity=0.6,
        ))

    fig = _apply_layout(fig, "Rolling Signal Variance (Stability Monitor)", yaxis_title="Variance")
    _save_and_show(fig, "rolling_variance.html")
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Render an interactive Pearson correlation heatmap across primary signals."""
    cols = [c for c in config.SIGNAL_NAMES if c in df.columns]
    corr = df[cols].corr(method="pearson")

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="Corr(%{x}, %{y}) = %{z:.3f}<extra></extra>",
    ))

    fig = _apply_layout(
        fig,
        "Cross-Signal Pearson Correlation Matrix",
        xaxis_title="", yaxis_title="",
        height=550, width=650,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    _save_and_show(fig, "correlation_heatmap.html")
    return fig


# ---------------------------------------------------------------------------
# 6. Anomaly / deviation visualisation
# ---------------------------------------------------------------------------

def plot_anomaly_overlay(df: pd.DataFrame, signal: str = "HR") -> go.Figure:
    """
    Overlay detected anomalies on a chosen signal's time-series.

    Normal points are rendered with a muted colour; anomalies are
    highlighted with a vivid marker.
    """
    if signal not in df.columns or "is_anomaly" not in df.columns:
        logger.warning("Required columns missing for anomaly overlay.")
        return go.Figure()

    normal = df[~df["is_anomaly"]]
    anomalous = df[df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal.index, y=normal[signal],
        mode="lines",
        name="Normal",
        line=dict(color=config.VIZ_NORMAL_COLOR, width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=anomalous.index, y=anomalous[signal],
        mode="markers",
        name="Anomaly",
        marker=dict(
            color=config.VIZ_ANOMALY_COLOR,
            size=7,
            symbol="circle",
            line=dict(width=1, color="white"),
        ),
    ))

    fig = _apply_layout(
        fig,
        f"Anomaly Detection Overlay — {signal}",
        yaxis_title=signal,
    )
    _save_and_show(fig, f"anomaly_overlay_{signal.lower()}.html")
    return fig


def plot_deviation_score_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Plot the continuous deviation score over time, with a threshold line
    separating normal from anomalous regions.
    """
    if "deviation_score" not in df.columns:
        logger.warning("deviation_score column missing.")
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["deviation_score"],
        mode="lines",
        name="Deviation Score",
        line=dict(color=config.VIZ_COLOR_PALETTE[0], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
    ))

    # Approximate decision boundary (score = 0 is the default threshold
    # for sklearn IsolationForest decision_function)
    fig.add_hline(y=0, line_dash="dash", line_color=config.VIZ_ANOMALY_COLOR,
                  annotation_text="Decision Boundary", opacity=0.6)

    fig = _apply_layout(
        fig,
        "Isolation Forest Deviation Score Timeline",
        yaxis_title="Decision Function Score",
    )
    _save_and_show(fig, "deviation_score_timeline.html")
    return fig


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def generate_all_visualisations(df: pd.DataFrame) -> None:
    """
    Run the complete visualisation suite and persist all figures to disk.
    """
    logger.info("Generating visualisations for %d samples ...", len(df))

    plot_multisignal_timeseries(df)
    plot_normalised_deviations(df)
    plot_signal_dynamics(df)
    plot_rolling_variance(df)
    plot_correlation_heatmap(df)
    plot_deviation_score_timeline(df)

    # Anomaly overlays for each primary signal
    for signal in config.SIGNAL_NAMES:
        if signal in df.columns and "is_anomaly" in df.columns:
            plot_anomaly_overlay(df, signal=signal)

    logger.info("All visualisations generated.")
