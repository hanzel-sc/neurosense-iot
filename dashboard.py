"""
NeuroSense AIoT — Dashboard Generator
---------------------------------------
Reads all pipeline JSON reports and the scored DataFrame, then produces a
single self-contained HTML dashboard with shadcn-inspired dark styling
and interactive Plotly.js charts.

Usage
-----
    Called automatically at the end of main.py, or standalone:

        python dashboard.py

    Requires that the pipeline has already run and produced output files.
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report loaders
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    """Load a JSON report from the reports directory."""
    path = os.path.join(config.REPORT_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Report file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# HTML template fragments
# ---------------------------------------------------------------------------

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-primary: #09090b;
    --bg-card: #0c0c0f;
    --bg-card-hover: #131318;
    --bg-muted: #18181b;
    --border: #27272a;
    --border-subtle: #1e1e22;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-cyan: #22d3ee;
    --accent-pink: #f472b6;
    --accent-green: #4ade80;
    --accent-amber: #fbbf24;
    --accent-purple: #a78bfa;
    --accent-red: #f87171;
    --ring: rgba(34,211,238,0.15);
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5), 0 4px 6px -2px rgba(0,0,0,0.3);
    --transition: 150ms cubic-bezier(0.4, 0, 0.2, 1);
  }

  html { scroll-behavior: smooth; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

  /* Layout */
  .dashboard { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }

  .header {
    text-align: center;
    margin-bottom: 48px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--border);
  }
  .header h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }
  .header p {
    color: var(--text-secondary);
    font-size: 15px;
    font-weight: 400;
  }
  .header .badge {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.2);
    color: var(--accent-cyan);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* Navigation */
  .nav {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 40px;
    padding: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .nav a {
    padding: 8px 16px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    transition: all var(--transition);
  }
  .nav a:hover {
    background: var(--bg-muted);
    color: var(--text-primary);
  }

  /* Section */
  .section {
    margin-bottom: 48px;
    scroll-margin-top: 24px;
  }
  .section-title {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.3px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-title .icon {
    width: 28px;
    height: 28px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .section-desc {
    color: var(--text-muted);
    font-size: 13px;
    margin-bottom: 20px;
    max-width: 700px;
  }

  /* Cards */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    transition: all var(--transition);
    box-shadow: var(--shadow);
  }
  .card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-subtle);
    box-shadow: var(--shadow-lg);
  }
  .card-title {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  .card-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .card-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  /* Grids */
  .grid-5 { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
  .grid-1 { display: grid; grid-template-columns: 1fr; gap: 16px; }

  /* Chart container */
  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .chart-card .chart-header {
    margin-bottom: 16px;
  }
  .chart-card .chart-header h3 {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.2px;
  }
  .chart-card .chart-header p {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  .chart-wrap { width: 100%; }
  .chart-wrap .js-plotly-plot { border-radius: var(--radius-sm); }

  /* Table */
  .data-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
  }
  .data-table th {
    text-align: left;
    padding: 10px 14px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-muted);
  }
  .data-table th:first-child { border-radius: var(--radius-sm) 0 0 0; }
  .data-table th:last-child { border-radius: 0 var(--radius-sm) 0 0; }
  .data-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background: var(--bg-card-hover); }
  .data-table .signal-name { color: var(--text-primary); font-weight: 500; }

  /* Insight cards */
  .insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 3px solid var(--accent-amber);
    transition: all var(--transition);
  }
  .insight-card:hover { background: var(--bg-card-hover); }
  .insight-timestamp {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .insight-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.7;
  }

  /* Anomaly event */
  .anomaly-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 3px solid var(--accent-red);
  }
  .anomaly-card:hover { background: var(--bg-card-hover); }
  .anomaly-score {
    font-size: 22px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .contrib-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
  }
  .contrib-label {
    font-size: 12px;
    color: var(--text-secondary);
    width: 90px;
    text-align: right;
    flex-shrink: 0;
  }
  .contrib-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--bg-muted);
    border-radius: 4px;
    overflow: hidden;
  }
  .contrib-bar {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
  }
  .contrib-val {
    font-size: 11px;
    color: var(--text-muted);
    width: 50px;
    font-variant-numeric: tabular-nums;
  }

  /* Comparison badges */
  .change-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
  }
  .change-badge.positive {
    background: rgba(74,222,128,0.1);
    color: var(--accent-green);
  }
  .change-badge.negative {
    background: rgba(248,113,113,0.1);
    color: var(--accent-red);
  }
  .change-badge.neutral {
    background: rgba(161,161,170,0.1);
    color: var(--text-muted);
  }

  /* Footer */
  .footer {
    text-align: center;
    padding: 32px 0;
    margin-top: 48px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .dashboard { padding: 16px 12px; }
    .header h1 { font-size: 24px; }
    .grid-5, .grid-3, .grid-2 { grid-template-columns: 1fr; }
    .nav { flex-direction: column; }
  }
</style>
"""

_PLOTLY_CONFIG = "{ responsive: true, displayModeBar: false }"
_PLOTLY_DARK_LAYOUT = """{
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, sans-serif', color: '#a1a1aa', size: 11 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { gridcolor: '#1e1e22', zerolinecolor: '#27272a' },
    yaxis: { gridcolor: '#1e1e22', zerolinecolor: '#27272a' },
    legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
    hoverlabel: { bgcolor: '#18181b', bordercolor: '#27272a', font: { family: 'Inter', size: 12 } }
}"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_kpi_cards(quality: Dict, analytics: Dict) -> str:
    """Top-row KPI summary cards."""
    stats = analytics.get("signal_statistics", {})
    hr = stats.get("HR", {})
    rmssd = stats.get("RMSSD", {})
    stress = stats.get("StressScore", {})
    temp = stats.get("SkinTemp", {})
    moisture = stats.get("Moisture", {})

    cards = [
        ("Heart Rate", f"{hr.get('mean', '--')}", "BPM avg", "var(--accent-cyan)"),
        ("HRV (RMSSD)", f"{rmssd.get('mean', '--')}", "ms avg", "var(--accent-pink)"),
        ("Skin Temp", f"{temp.get('mean', '--')}", "C avg", "var(--accent-green)"),
        ("Moisture", f"{moisture.get('mean', '--')}", "% avg", "var(--accent-amber)"),
        ("Stress Score", f"{stress.get('mean', '--')}", "heuristic", "var(--accent-purple)"),
    ]

    html = '<div class="grid-5">'
    for title, value, sub, color in cards:
        html += f'''
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value" style="color:{color}">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>'''
    html += '</div>'
    return html


def _build_quality_cards(quality: Dict) -> str:
    """Data quality summary row."""
    total = quality.get("total_raw_samples", 0)
    clean = quality.get("total_clean_samples", 0)
    retention = quality.get("row_retention_pct", 0)

    html = '<div class="grid-3">'
    html += f'''
    <div class="card">
      <div class="card-title">Raw Samples</div>
      <div class="card-value" style="color:var(--text-primary)">{total}</div>
      <div class="card-sub">Total records acquired</div>
    </div>
    <div class="card">
      <div class="card-title">Clean Samples</div>
      <div class="card-value" style="color:var(--accent-green)">{clean}</div>
      <div class="card-sub">After validation and cleaning</div>
    </div>
    <div class="card">
      <div class="card-title">Retention Rate</div>
      <div class="card-value" style="color:var(--accent-cyan)">{retention}%</div>
      <div class="card-sub">Data integrity score</div>
    </div>'''
    html += '</div>'
    return html


def _build_signal_stats_table(analytics: Dict) -> str:
    """Signal statistics table."""
    stats = analytics.get("signal_statistics", {})
    if not stats:
        return '<p style="color:var(--text-muted)">No signal statistics available.</p>'

    html = '''<div class="card" style="padding:0; overflow:hidden;">
    <table class="data-table">
      <thead><tr>
        <th>Signal</th><th>Mean</th><th>Std</th><th>Median</th>
        <th>IQR</th><th>Min</th><th>Max</th><th>Skew</th><th>Kurtosis</th>
      </tr></thead><tbody>'''

    for signal, m in stats.items():
        html += f'''<tr>
          <td class="signal-name">{signal}</td>
          <td>{m.get("mean","")}</td><td>{m.get("std","")}</td>
          <td>{m.get("median","")}</td><td>{m.get("iqr","")}</td>
          <td>{m.get("min","")}</td><td>{m.get("max","")}</td>
          <td>{m.get("skewness","")}</td><td>{m.get("kurtosis","")}</td>
        </tr>'''

    html += '</tbody></table></div>'
    return html


def _build_baseline_comparison_table(ba: Dict) -> str:
    """Baseline vs Activation comparison table."""
    if not ba:
        return '<p style="color:var(--text-muted)">No comparison data available.</p>'

    html = '''<div class="card" style="padding:0; overflow:hidden;">
    <table class="data-table">
      <thead><tr>
        <th>Signal</th><th>Baseline Mean</th><th>Activation Mean</th>
        <th>Diff</th><th>% Change</th><th>Cohen's d</th><th>p-value</th>
      </tr></thead><tbody>'''

    for signal, m in ba.items():
        pct = m.get("pct_change", 0)
        badge_cls = "positive" if pct > 0 else ("negative" if pct < 0 else "neutral")
        sign = "+" if pct > 0 else ""
        pval = m.get("p_value", 1)
        sig_marker = " *" if pval < 0.05 else ""

        html += f'''<tr>
          <td class="signal-name">{signal}</td>
          <td>{m.get("baseline_mean","")}</td>
          <td>{m.get("activation_mean","")}</td>
          <td>{m.get("mean_diff","")}</td>
          <td><span class="change-badge {badge_cls}">{sign}{pct}%</span></td>
          <td>{m.get("cohens_d","")}</td>
          <td>{pval}{sig_marker}</td>
        </tr>'''

    html += '</tbody></table></div>'
    return html


def _build_anomaly_events(explanations: List) -> str:
    """Anomaly explainability event cards with contribution bars."""
    if not explanations:
        return '<p style="color:var(--text-muted)">No anomalous events detected.</p>'

    contrib_colors = {
        "HR": "#22d3ee",
        "RMSSD": "#f472b6",
        "SkinTemp": "#4ade80",
        "Moisture": "#fbbf24",
        "StressScore": "#a78bfa",
    }

    html = ""
    for i, evt in enumerate(explanations):
        score = evt.get("deviation_score", 0)
        contributors = evt.get("contributors", {})
        max_z = max(contributors.values()) if contributors else 1

        html += f'''<div class="anomaly-card">
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px;">
            <div class="insight-timestamp">Event {i+1} &mdash; {evt.get("timestamp","")}</div>
            <div class="anomaly-score" style="color:var(--accent-red)">{score:.4f}</div>
          </div>'''

        for sig, z_val in contributors.items():
            pct = min(z_val / max(max_z, 0.01) * 100, 100)
            color = contrib_colors.get(sig, "#71717a")
            html += f'''
          <div class="contrib-bar-wrap">
            <div class="contrib-label">{sig}</div>
            <div class="contrib-bar-bg">
              <div class="contrib-bar" style="width:{pct:.0f}%; background:{color};"></div>
            </div>
            <div class="contrib-val">z={z_val:.2f}</div>
          </div>'''

        html += '</div>'
    return html


def _build_insight_cards(insights_data: List) -> str:
    """Physiological insight cards."""
    if not insights_data:
        return '<p style="color:var(--text-muted)">No insights generated.</p>'

    html = ""
    for entry in insights_data:
        ts = entry.get("timestamp", "")
        for insight in entry.get("insights", []):
            html += f'''<div class="insight-card">
              <div class="insight-timestamp">{ts}</div>
              <div class="insight-text">{insight}</div>
            </div>'''
    return html


def _build_response_magnitudes_table(magnitudes: Dict) -> str:
    """Response magnitude table."""
    if not magnitudes:
        return '<p style="color:var(--text-muted)">No response magnitude data.</p>'

    html = '''<div class="card" style="padding:0; overflow:hidden;">
    <table class="data-table">
      <thead><tr>
        <th>Signal</th><th>Peak Dev (raw)</th><th>Peak Dev (z)</th>
        <th>Mean Dev (raw)</th><th>Mean Dev (z)</th>
      </tr></thead><tbody>'''

    for signal, m in magnitudes.items():
        html += f'''<tr>
          <td class="signal-name">{signal}</td>
          <td>{m.get("peak_deviation_raw","")}</td>
          <td>{m.get("peak_deviation_zscore","")}</td>
          <td>{m.get("mean_deviation_raw","")}</td>
          <td>{m.get("mean_deviation_zscore","")}</td>
        </tr>'''

    html += '</tbody></table></div>'
    return html


# ---------------------------------------------------------------------------
# Plotly chart JS generators
# ---------------------------------------------------------------------------

def _js_timeseries_chart(df: pd.DataFrame) -> str:
    """Multi-signal time-series chart."""
    signals = [c for c in config.SIGNAL_NAMES if c in df.columns]
    colors = config.VIZ_COLOR_PALETTE

    traces = ""
    for i, sig in enumerate(signals):
        x_vals = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        y_vals = df[sig].tolist()
        color = colors[i % len(colors)]
        traces += f"""{{
            x: {json.dumps(x_vals)},
            y: {json.dumps(y_vals)},
            name: '{sig}',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '{color}', width: 2 }},
            marker: {{ size: 4 }},
        }},\n"""

    return f"""
    Plotly.newPlot('chart-timeseries', [{traces}],
      Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 400,
        xaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.xaxis, title: 'Time' }},
      }}), {_PLOTLY_CONFIG});
    """


def _js_zscore_chart(df: pd.DataFrame) -> str:
    """Normalized z-score deviations chart."""
    colors = config.VIZ_COLOR_PALETTE
    traces = ""
    for i, sig in enumerate(config.SIGNAL_NAMES):
        zcol = f"{sig}_zscore"
        if zcol not in df.columns:
            continue
        x_vals = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        y_vals = df[zcol].replace({np.nan: None}).tolist()
        color = colors[i % len(colors)]
        traces += f"""{{
            x: {json.dumps(x_vals)},
            y: {json.dumps(y_vals)},
            name: '{sig} (z)',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '{color}', width: 1.5 }},
        }},\n"""

    sigma = config.DEVIATION_THRESHOLD_SIGMA
    return f"""
    Plotly.newPlot('chart-zscore', [{traces}],
      Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 380,
        yaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.yaxis, title: 'Z-Score' }},
        shapes: [
          {{ type: 'line', y0: {sigma}, y1: {sigma}, x0: 0, x1: 1, xref: 'paper',
             line: {{ color: 'rgba(255,255,255,0.2)', dash: 'dash', width: 1 }} }},
          {{ type: 'line', y0: {-sigma}, y1: {-sigma}, x0: 0, x1: 1, xref: 'paper',
             line: {{ color: 'rgba(255,255,255,0.2)', dash: 'dash', width: 1 }} }},
        ],
        annotations: [
          {{ x: 1, xref: 'paper', y: {sigma}, text: '+{sigma}σ', showarrow: false,
             font: {{ color: '#71717a', size: 10 }} }},
          {{ x: 1, xref: 'paper', y: {-sigma}, text: '-{sigma}σ', showarrow: false,
             font: {{ color: '#71717a', size: 10 }} }},
        ]
      }}), {_PLOTLY_CONFIG});
    """


def _js_correlation_heatmap(analytics: Dict) -> str:
    """Correlation heatmap chart."""
    corr = analytics.get("correlation_matrix", {})
    if not corr:
        return ""
    signals = list(corr.keys())
    z_matrix = [[round(corr[r].get(c, 0), 3) for c in signals] for r in signals]

    return f"""
    Plotly.newPlot('chart-correlation', [{{
        z: {json.dumps(z_matrix)},
        x: {json.dumps(signals)},
        y: {json.dumps(signals)},
        type: 'heatmap',
        colorscale: [
          [0, '#3b0764'], [0.25, '#6d28d9'], [0.5, '#18181b'],
          [0.75, '#0e7490'], [1, '#22d3ee']
        ],
        zmin: -1, zmax: 1,
        text: {json.dumps(z_matrix)},
        texttemplate: '%{{text}}',
        textfont: {{ size: 12, color: '#a1a1aa' }},
        hovertemplate: 'Corr(%{{x}}, %{{y}}) = %{{z:.3f}}<extra></extra>',
        showscale: true,
        colorbar: {{ tickfont: {{ color: '#71717a' }}, thickness: 12, outlinewidth: 0 }}
    }}],
    Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 420, width: null,
        yaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.yaxis, autorange: 'reversed' }},
    }}), {_PLOTLY_CONFIG});
    """


def _js_rolling_variance_chart(df: pd.DataFrame) -> str:
    """Rolling variance chart."""
    colors = config.VIZ_COLOR_PALETTE
    traces = ""
    for i, sig in enumerate(config.SIGNAL_NAMES):
        vcol = f"{sig}_roll_var"
        if vcol not in df.columns:
            continue
        x_vals = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        y_vals = df[vcol].replace({np.nan: None}).tolist()
        color = colors[i % len(colors)]
        traces += f"""{{
            x: {json.dumps(x_vals)},
            y: {json.dumps(y_vals)},
            name: '{sig}',
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: {{ color: '{color}', width: 1.5 }},
            fillcolor: '{color}11',
        }},\n"""

    return f"""
    Plotly.newPlot('chart-variance', [{traces}],
      Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 380,
        yaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.yaxis, title: 'Variance' }},
      }}), {_PLOTLY_CONFIG});
    """


def _js_slope_chart(df: pd.DataFrame) -> str:
    """Signal slope / first derivative chart."""
    colors = config.VIZ_COLOR_PALETTE
    traces = ""
    for i, sig in enumerate(config.SIGNAL_NAMES):
        scol = f"{sig}_slope"
        if scol not in df.columns:
            continue
        x_vals = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        y_vals = df[scol].replace({np.nan: None}).tolist()
        color = colors[i % len(colors)]
        traces += f"""{{
            x: {json.dumps(x_vals)},
            y: {json.dumps(y_vals)},
            name: '{sig}',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '{color}', width: 1.5 }},
        }},\n"""

    return f"""
    Plotly.newPlot('chart-slope', [{traces}],
      Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 380,
        yaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.yaxis, title: 'dSignal / dt' }},
        shapes: [{{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper',
                    line: {{ color: 'rgba(255,255,255,0.1)', dash: 'dot', width: 1 }} }}]
      }}), {_PLOTLY_CONFIG});
    """


def _js_anomaly_chart(df: pd.DataFrame) -> str:
    """Deviation score timeline with anomaly markers."""
    if "deviation_score" not in df.columns or "is_anomaly" not in df.columns:
        return ""

    x_all = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    y_all = df["deviation_score"].tolist()

    anomalous = df[df["is_anomaly"]]
    x_anom = anomalous.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    y_anom = anomalous["deviation_score"].tolist()

    return f"""
    Plotly.newPlot('chart-anomaly', [
      {{
        x: {json.dumps(x_all)},
        y: {json.dumps(y_all)},
        name: 'Deviation Score',
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#22d3ee', width: 2 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(34,211,238,0.06)',
      }},
      {{
        x: {json.dumps(x_anom)},
        y: {json.dumps(y_anom)},
        name: 'Anomaly',
        type: 'scatter',
        mode: 'markers',
        marker: {{ color: '#f87171', size: 9, symbol: 'circle',
                  line: {{ color: 'white', width: 1 }} }},
      }}
    ],
    Object.assign({{}}, {_PLOTLY_DARK_LAYOUT}, {{
        height: 380,
        yaxis: {{ ...{_PLOTLY_DARK_LAYOUT}.yaxis, title: 'Score' }},
        shapes: [{{ type: 'line', y0: 0, y1: 0, x0: 0, x1: 1, xref: 'paper',
                    line: {{ color: '#f87171', dash: 'dash', width: 1 }} }}]
    }}), {_PLOTLY_CONFIG});
    """


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_dashboard(df_scored: pd.DataFrame) -> str:
    """
    Generate the full dashboard HTML string.

    Parameters
    ----------
    df_scored : pd.DataFrame
        The scored DataFrame from the ML inference stage.

    Returns
    -------
    str
        Complete HTML document.
    """
    # Load reports
    quality = _load_json("data_quality_metrics.json") or {}
    analytics_data = _load_json("quantitative_analytics.json") or {}
    explanations = _load_json("explainability_report.json") or []
    insights_data = _load_json("physiological_insights.json") or []
    ba_comparison = _load_json("baseline_vs_activation.json") or {}
    magnitudes = _load_json("response_magnitudes.json") or {}

    # Build HTML sections
    kpi_html = _build_kpi_cards(quality, analytics_data)
    quality_html = _build_quality_cards(quality)
    stats_table = _build_signal_stats_table(analytics_data)
    ba_table = _build_baseline_comparison_table(ba_comparison)
    anomaly_events = _build_anomaly_events(explanations)
    insight_cards = _build_insight_cards(insights_data)
    magnitudes_table = _build_response_magnitudes_table(magnitudes)

    # Build JS for charts
    js_charts = "\n".join([
        _js_timeseries_chart(df_scored),
        _js_zscore_chart(df_scored),
        _js_correlation_heatmap(analytics_data),
        _js_rolling_variance_chart(df_scored),
        _js_slope_chart(df_scored),
        _js_anomaly_chart(df_scored),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeuroSense AIoT Dashboard</title>
  <meta name="description" content="Personalized Multimodal Physiological State Inference System — Analytics Dashboard">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  {_CSS}
</head>
<body>
  <div class="dashboard">

    <!-- Header -->
    <div class="header">
      <h1>NeuroSense AIoT</h1>
      <p>Personalized Multimodal Physiological State Inference System</p>
      <div class="badge">Live Analytics Dashboard</div>
    </div>

    <!-- Navigation -->
    <div class="nav">
      <a href="#overview">Overview</a>
      <a href="#quality">Data Quality</a>
      <a href="#timeseries">Time-Series</a>
      <a href="#deviations">Deviations</a>
      <a href="#dynamics">Dynamics</a>
      <a href="#stability">Stability</a>
      <a href="#correlation">Correlation</a>
      <a href="#anomalies">Anomalies</a>
      <a href="#explainability">Explainability</a>
      <a href="#insights">Insights</a>
      <a href="#comparison">Comparison</a>
    </div>

    <!-- Section: Overview KPIs -->
    <div class="section" id="overview">
      <div class="section-title">
        <div class="icon" style="background:rgba(34,211,238,0.1); color:var(--accent-cyan);">&#9876;</div>
        Signal Overview
      </div>
      <div class="section-desc">Mean physiological signal values across the recording session.</div>
      {kpi_html}
    </div>

    <!-- Section: Data Quality -->
    <div class="section" id="quality">
      <div class="section-title">
        <div class="icon" style="background:rgba(74,222,128,0.1); color:var(--accent-green);">&#9745;</div>
        Data Quality
      </div>
      <div class="section-desc">Acquisition integrity metrics: raw samples, validated samples, and retention rate after cleaning.</div>
      {quality_html}
    </div>

    <!-- Section: Signal Statistics -->
    <div class="section" id="statistics">
      <div class="section-title">
        <div class="icon" style="background:rgba(167,139,250,0.1); color:var(--accent-purple);">&#931;</div>
        Descriptive Statistics
      </div>
      <div class="section-desc">Per-signal descriptive statistics including central tendency, dispersion, and distribution shape.</div>
      {stats_table}
    </div>

    <!-- Section: Multi-Signal Time-Series -->
    <div class="section" id="timeseries">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Multi-Signal Physiological Time-Series</h3>
          <p>All primary physiological signals plotted on a shared time axis. HR and RMSSD represent cardiovascular state; SkinTemp reflects thermoregulation; Moisture tracks electrodermal activity; StressScore is a composite heuristic.</p>
        </div>
        <div class="chart-wrap" id="chart-timeseries"></div>
      </div>
    </div>

    <!-- Section: Z-Score Deviations -->
    <div class="section" id="deviations">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Normalized Signal Deviations (Z-Score)</h3>
          <p>All signals normalized to z-scores for cross-signal comparison. Dashed lines mark the deviation threshold at +/- {config.DEVIATION_THRESHOLD_SIGMA} sigma. Values beyond these thresholds indicate statistically significant departure from the session mean.</p>
        </div>
        <div class="chart-wrap" id="chart-zscore"></div>
      </div>
    </div>

    <!-- Section: Signal Dynamics -->
    <div class="section" id="dynamics">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Signal Rate of Change (First Derivative)</h3>
          <p>Instantaneous slope of each signal computed via finite differences over time. Positive values indicate increasing signal magnitude; negative values indicate decreasing. Rapid changes highlight acute physiological events.</p>
        </div>
        <div class="chart-wrap" id="chart-slope"></div>
      </div>
    </div>

    <!-- Section: Rolling Variance -->
    <div class="section" id="stability">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Rolling Signal Variance (Stability Monitor)</h3>
          <p>Windowed variance (window = {config.ROLLING_WINDOW_SIZE} samples) for each signal. Elevated variance indicates signal instability during physiological perturbation. Low variance suggests stable homeostatic regulation.</p>
        </div>
        <div class="chart-wrap" id="chart-variance"></div>
      </div>
    </div>

    <!-- Section: Correlation Heatmap -->
    <div class="section" id="correlation">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Cross-Signal Correlation Matrix</h3>
          <p>Pearson correlation coefficients between all primary signals. Strong negative correlation (e.g., RMSSD vs StressScore) suggests antagonistic regulatory mechanisms. Values near zero indicate statistical independence.</p>
        </div>
        <div class="chart-wrap" id="chart-correlation"></div>
      </div>
    </div>

    <!-- Section: Anomaly Detection -->
    <div class="section" id="anomalies">
      <div class="chart-card">
        <div class="chart-header">
          <h3>Anomaly Detection — Deviation Score Timeline</h3>
          <p>Continuous output of the Isolation Forest decision function. Scores below the decision boundary (red dashed line at 0) are classified as anomalous. Lower scores indicate stronger deviation from the learned baseline distribution.</p>
        </div>
        <div class="chart-wrap" id="chart-anomaly"></div>
      </div>
    </div>

    <!-- Section: Explainability -->
    <div class="section" id="explainability">
      <div class="section-title">
        <div class="icon" style="background:rgba(248,113,113,0.1); color:var(--accent-red);">&#9888;</div>
        Anomaly Explainability
      </div>
      <div class="section-desc">Per-event decomposition of deviation scores into individual signal contributions. Each bar represents the z-score deviation of that signal from its baseline distribution. Longer bars indicate greater responsibility for the anomaly flag.</div>
      {anomaly_events}
    </div>

    <!-- Section: Physiological Insights -->
    <div class="section" id="insights">
      <div class="section-title">
        <div class="icon" style="background:rgba(251,191,36,0.1); color:var(--accent-amber);">&#9733;</div>
        Physiological Insights
      </div>
      <div class="section-desc">Rule-based interpretations generated by evaluating physiological heuristics against detected anomalies. These observations describe plausible biological mechanisms underlying the signal patterns.</div>
      {insight_cards}
    </div>

    <!-- Section: Baseline vs Activation Comparison -->
    <div class="section" id="comparison">
      <div class="section-title">
        <div class="icon" style="background:rgba(244,114,182,0.1); color:var(--accent-pink);">&#8644;</div>
        Baseline vs Activation Comparison
      </div>
      <div class="section-desc">Statistical comparison between the first {int(config.BASELINE_FRACTION*100)}% of the recording (baseline) and the remainder (activation). Cohen's d quantifies effect size; p-values are from Welch's t-test. An asterisk (*) marks p &lt; 0.05.</div>
      {ba_table}
    </div>

    <!-- Section: Response Magnitudes -->
    <div class="section" style="margin-top: 24px;">
      <div class="section-title">
        <div class="icon" style="background:rgba(167,139,250,0.1); color:var(--accent-purple);">&#8679;</div>
        Response Magnitudes
      </div>
      <div class="section-desc">Peak and mean deviation of each signal in the activation segment relative to the baseline mean, expressed in both raw units and baseline-normalized z-scores.</div>
      {magnitudes_table}
    </div>

    <!-- Footer -->
    <div class="footer">
      NeuroSense AIoT &mdash; Personalized Multimodal Physiological State Inference System<br>
      Generated by the NeuroSense analytics pipeline
    </div>

  </div>

  <script>
    document.addEventListener('DOMContentLoaded', function() {{
      {js_charts}
    }});
  </script>
</body>
</html>"""

    return html


def save_dashboard(df_scored: pd.DataFrame) -> str:
    """
    Generate and save the dashboard HTML to the output directory.

    Returns
    -------
    str
        Absolute path to the saved dashboard file.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    html = generate_dashboard(df_scored)
    path = os.path.join(config.OUTPUT_DIR, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data_acquisition
    import data_cleaning
    import feature_engineering
    import ml_inference

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    logger.info("Standalone dashboard generation mode.")
    logger.info("Fetching data from ThingSpeak ...")
    df = data_acquisition.fetch_data()
    df, _ = data_cleaning.clean_data(df)
    df = feature_engineering.engineer_features(df)
    df, _ = ml_inference.run_anomaly_detection(df)
    path = save_dashboard(df)
    logger.info("Dashboard ready at: %s", path)
