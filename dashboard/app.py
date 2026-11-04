from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
DATA = PROJECT_ROOT / "data"

PATHS = {
    "study_config": OUT / "data" / "study_config.json",
    "machine_population": OUT / "data" / "machine_population.csv",
    "spc": OUT / "spc" / "spc_results.csv",
    "nelson": OUT / "spc" / "nelson_summary.csv",
    "capability": OUT / "spc" / "capability_results.csv",
    "break_modes": OUT / "spc" / "break_mode_summary.csv",
    "latest": OUT / "deployment" / "latest_machine_status.csv",
    "arima_forecast": OUT / "models" / "arima_next_5_forecast.csv",
    "arima_metrics": OUT / "evaluation" / "arima_metrics.csv",
    "mlp_latest": OUT / "models" / "mlp_latest_predictions.csv",
    "mlp_metrics": OUT / "evaluation" / "mlp_metrics.csv",
    "mlp_validation": OUT / "evaluation" / "mlp_validation.csv",
    "mlp_cm": OUT / "evaluation" / "mlp_confusion_matrix.csv",
    "mlp_class": OUT / "evaluation" / "mlp_class_report.csv",
    "comparison": OUT / "evaluation" / "model_comparison.csv",
    "split": OUT / "evaluation" / "split_summary.csv",
    "mapping": DATA / "FAILURE_MODE_MAPPING.csv",
}

STATE_ORDER = {"NORMAL": 0, "WARNING": 1, "ABNORMAL": 2, "INSUFFICIENT": -1}
STATE_COLOR = {
    "NORMAL": "#2E8B57",
    "WARNING": "#E6A700",
    "ABNORMAL": "#C62828",
    "INSUFFICIENT": "#808080",
    "REVIEW": "#6A5ACD",
}
CTQ_NAME = {"WP": "Wire Pull", "BS": "Ball Shear", "SP": "Stitch Pull"}
D4_N5 = 2.114

st.set_page_config(
    page_title="Wire Bond SPC-AI Engineering Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2.0rem;}
[data-testid="stMetric"] {border:1px solid #e6e6e6; border-radius:10px; padding:9px 12px; background:#ffffff;}
.small-note {font-size:0.82rem; color:#5f6368;}
.section-title {font-size:1.18rem; font-weight:700; margin-top:0.4rem; margin-bottom:0.2rem;}
.health-normal {padding:14px; border-radius:10px; border-left:6px solid #2E8B57; background:#f3fbf6;}
.health-warning {padding:14px; border-radius:10px; border-left:6px solid #E6A700; background:#fff9e8;}
.health-abnormal {padding:14px; border-radius:10px; border-left:6px solid #C62828; background:#fff3f3;}
.flow-card {padding:10px 8px; border:1px solid #dedede; border-radius:10px; text-align:center; min-height:82px; background:white;}
.flow-arrow {font-size:1.5rem; text-align:center; padding-top:20px; color:#777;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.upper().isin(["TRUE", "1", "YES", "Y"])


def clean_state(x: object) -> str:
    s = str(x).upper().strip()
    return s if s in STATE_ORDER else "INSUFFICIENT"


def state_from_z(z: float) -> str:
    if not np.isfinite(z):
        return "INSUFFICIENT"
    if abs(z) > 3:
        return "ABNORMAL"
    if abs(z) > 2:
        return "WARNING"
    return "NORMAL"


def more_severe(*states: str) -> str:
    valid = [clean_state(s) for s in states]
    valid = [s for s in valid if s != "INSUFFICIENT"]
    if not valid:
        return "INSUFFICIENT"
    return max(valid, key=lambda s: STATE_ORDER[s])


def capability_judgement(value: float) -> str:
    if not np.isfinite(value):
        return "NOT AVAILABLE"
    if value >= 1.33:
        return "CAPABLE"
    if value >= 1.00:
        return "MONITOR"
    return "REVIEW"


def action_logic(
    current: str,
    hybrid: str,
    p_abnormal: float,
    capability_value: float,
    recent_nelson_events: int,
    latest_nelson: str,
) -> tuple[str, str]:
    """Conservative engineering DSS rule. This is decision support, not automatic PM control."""
    current = clean_state(current)
    hybrid = clean_state(hybrid)
    cap_bad = np.isfinite(capability_value) and capability_value < 1.0
    cap_marginal = np.isfinite(capability_value) and 1.0 <= capability_value < 1.33
    strong_signal = "N1" in str(latest_nelson).upper()

    if current == "ABNORMAL" or strong_signal:
        return "ENGINEERING REVIEW", "ABNORMAL"
    if hybrid == "ABNORMAL" and (p_abnormal >= 0.50 or cap_bad or recent_nelson_events >= 2):
        return "EARLY PM CANDIDATE", "ABNORMAL"
    if hybrid in {"WARNING", "ABNORMAL"} or recent_nelson_events > 0 or cap_marginal:
        return "INCREASED MONITORING", "WARNING"
    return "ROUTINE MONITORING", "NORMAL"


def engineering_focus(ctq: str, action: str) -> list[str]:
    focus = {
        "WP": [
            "Review capillary condition, wire feed and clamp operation.",
            "Check loop / neck behaviour and recent bonding parameter changes.",
            "Review recent wire-pull break-mode distribution before adjustment.",
        ],
        "BS": [
            "Review first-bond capillary condition and bond interface evidence.",
            "Check ultrasonic power, bond force and bond-head / stage temperature trends.",
            "Review ball-shear break-mode distribution and pad / surface condition.",
        ],
        "SP": [
            "Review second-bond capillary condition, clamp timing and tail control.",
            "Check stitch force / ultrasonic trend and lead or substrate surface condition.",
            "Review stitch-pull break-mode distribution before process adjustment.",
        ],
    }
    items = focus.get(ctq, ["Review recent SPC evidence and machine condition."])
    if action == "ROUTINE MONITORING":
        return ["Continue scheduled PM and routine SPC monitoring.", "No immediate machine intervention is indicated by the current DSS evidence."]
    if action == "INCREASED MONITORING":
        return ["Increase monitoring frequency before considering PM.", *items[:2]]
    if action == "EARLY PM CANDIDATE":
        return ["Consider an early condition-based PM review; confirm with equipment evidence first.", *items]
    return ["Perform engineering review before further process adjustment or PM decision.", *items]


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def add_zone_shading(fig: go.Figure, row: int, col: int) -> None:
    # SPC visual zones on standardized Xbar scale.
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(46,139,87,0.06)", line_width=0, row=row, col=col)
    fig.add_hrect(y0=1, y1=2, fillcolor="rgba(230,167,0,0.06)", line_width=0, row=row, col=col)
    fig.add_hrect(y0=-2, y1=-1, fillcolor="rgba(230,167,0,0.06)", line_width=0, row=row, col=col)
    fig.add_hrect(y0=2, y1=3, fillcolor="rgba(198,40,40,0.05)", line_width=0, row=row, col=col)
    fig.add_hrect(y0=-3, y1=-2, fillcolor="rgba(198,40,40,0.05)", line_width=0, row=row, col=col)


def spc_forecast_figure(
    spc_stream: pd.DataFrame,
    forecast: pd.DataFrame,
    recent_events: int,
) -> go.Figure:
    d = spc_stream.copy().sort_values("DATE_TIME").tail(recent_events).reset_index(drop=True)
    d["EVENT_SEQ"] = np.arange(1, len(d) + 1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.14,
        row_heights=[0.67, 0.33],
        subplot_titles=("Standardized X̄ Chart + Nelson Rules + ARIMA Next 5", "Normalized R Chart"),
    )
    add_zone_shading(fig, 1, 1)

    fig.add_trace(
        go.Scatter(
            x=d["EVENT_SEQ"],
            y=d["STANDARDIZED_DEVIATION"],
            mode="lines+markers",
            name="Actual X̄",
            line=dict(color="#355C7D", width=2),
            marker=dict(size=6),
            customdata=np.stack([
                d["PROCESS_STATE"].astype(str),
                d["NELSON_RULES_TRIGGERED"].fillna("").astype(str),
            ], axis=-1),
            hovertemplate="Event %{x}<br>X̄ deviation %{y:.2f}σ<br>State %{customdata[0]}<br>Nelson %{customdata[1]}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Nelson signals as visible stars on the actual Xbar chart.
    if "NELSON_ANY" in d.columns:
        sig = bool_series(d["NELSON_ANY"])
        if sig.any():
            fig.add_trace(
                go.Scatter(
                    x=d.loc[sig, "EVENT_SEQ"],
                    y=d.loc[sig, "STANDARDIZED_DEVIATION"],
                    mode="markers+text",
                    name="Nelson signal",
                    marker=dict(symbol="star", size=15, color="#C62828", line=dict(width=1, color="white")),
                    text=d.loc[sig, "NELSON_RULES_TRIGGERED"].fillna(""),
                    textposition="top center",
                    textfont=dict(size=10, color="#C62828"),
                    hovertemplate="Event %{x}<br>%{text}<br>X̄ %{y:.2f}σ<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # SPC references.
    for y, name, dash, color in [
        (3, "+3σ UCL", "dash", "#C62828"),
        (2, "+2σ", "dot", "#E6A700"),
        (1, "+1σ", "dot", "#999999"),
        (0, "CL", "solid", "#444444"),
        (-1, "-1σ", "dot", "#999999"),
        (-2, "-2σ", "dot", "#E6A700"),
        (-3, "-3σ LCL", "dash", "#C62828"),
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color, line_width=1, row=1, col=1, annotation_text=name, annotation_position="right")

    # ARIMA forecast is plotted on the same Xbar scale immediately after the recent actual events.
    if not forecast.empty:
        f = forecast.sort_values("HORIZON_EVENT").copy()
        f["X"] = len(d) + f["HORIZON_EVENT"].astype(int)
        fig.add_vline(x=len(d) + 0.5, line_dash="dash", line_color="#666666", row=1, col=1)
        fig.add_annotation(x=len(d) + 0.7, y=3.7, text="FORECAST →", showarrow=False, row=1, col=1)
        if {"LOWER_95", "UPPER_95"}.issubset(f.columns):
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([f["X"], f["X"].iloc[::-1]]),
                    y=pd.concat([f["UPPER_95"], f["LOWER_95"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(106,90,205,0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="ARIMA 95% interval",
                ),
                row=1,
                col=1,
            )
        f["STATE"] = f["FORECAST_Z"].apply(state_from_z)
        fig.add_trace(
            go.Scatter(
                x=f["X"],
                y=f["FORECAST_Z"],
                mode="lines+markers+text",
                name="ARIMA forecast",
                line=dict(color="#6A5ACD", width=3, dash="dash"),
                marker=dict(size=10, color=[STATE_COLOR.get(s, "#777") for s in f["STATE"]]),
                text=[f"+{int(h)}" for h in f["HORIZON_EVENT"]],
                textposition="top center",
                customdata=np.stack([f["HORIZON_EVENT"], f["STATE"]], axis=-1),
                hovertemplate="Future event +%{customdata[0]}<br>Forecast %{y:.2f}σ<br>Zone state %{customdata[1]}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # R chart normalized to Rbar.
    fig.add_trace(
        go.Scatter(
            x=d["EVENT_SEQ"],
            y=d["RANGE_RATIO"],
            mode="lines+markers",
            name="R / R̄",
            line=dict(color="#00838F", width=2),
            marker=dict(size=5),
            hovertemplate="Event %{x}<br>R/R̄ %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=D4_N5, line_dash="dash", line_color="#C62828", annotation_text="UCL(R)=2.114 R̄", annotation_position="right", row=2, col=1)
    fig.add_hline(y=1.0, line_color="#444444", annotation_text="R̄", annotation_position="right", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="#999999", row=2, col=1)

    fig.update_yaxes(title_text="X̄ deviation (σ)", row=1, col=1, range=[-4.2, 4.2])
    fig.update_yaxes(title_text="R / R̄", row=2, col=1)
    fig.update_xaxes(title_text="Recent production event sequence + forecast horizon", row=1, col=1)
    fig.update_xaxes(title_text="Recent production event sequence", row=2, col=1)
    fig.update_layout(height=730, margin=dict(l=30, r=50, t=80, b=30), legend_orientation="h", legend_y=-0.12)
    return fig


def nelson_bar(nelson_stream: pd.DataFrame) -> go.Figure:
    d = nelson_stream.copy()
    if d.empty:
        return go.Figure()
    d["RULE_NO"] = d["RULE"].str.extract(r"(\d+)").astype(int)
    d = d.sort_values("RULE_NO")
    fig = go.Figure()
    fig.add_bar(x=d["RULE"], y=d["TRIGGER_COUNT"], name="Triggers", marker_color="#5470C6")
    fig.add_bar(x=d["RULE"], y=d["EPISODE_COUNT"], name="Episodes", marker_color="#91CC75")
    fig.update_layout(barmode="group", height=300, margin=dict(l=30, r=10, t=45, b=30), title="Nelson Rule Profile")
    fig.update_yaxes(title="Count")
    return fig


def capability_gauge(cap_type: str, value: float) -> go.Figure:
    vmax = max(2.0, float(value) * 1.15) if np.isfinite(value) else 2.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) if np.isfinite(value) else 0.0,
        number={"valueformat": ".2f"},
        title={"text": f"{cap_type or 'Capability'}"},
        gauge={
            "axis": {"range": [0, vmax]},
            "bar": {"color": "#355C7D"},
            "steps": [
                {"range": [0, 1.0], "color": "#FADBD8"},
                {"range": [1.0, 1.33], "color": "#FCF3CF"},
                {"range": [1.33, vmax], "color": "#D5F5E3"},
            ],
            "threshold": {"line": {"color": "#C62828", "width": 3}, "value": 1.33},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=25, r=25, t=55, b=10))
    return fig


def break_mode_figure(d: pd.DataFrame) -> go.Figure:
    if d.empty:
        return go.Figure()
    plot = d.groupby("MODE", as_index=False)["COUNT"].sum().sort_values("COUNT", ascending=False)
    fig = px.bar(plot, x="MODE", y="COUNT", text="COUNT", title="Break-Mode Distribution")
    fig.update_traces(marker_color="#5B8FF9", textposition="outside")
    fig.update_layout(height=280, margin=dict(l=30, r=10, t=55, b=30), xaxis_title="Generic break mode", yaxis_title="Count")
    return fig


def mlp_probability_figure(row: pd.Series) -> go.Figure:
    d = pd.DataFrame({
        "State": ["NORMAL", "WARNING", "ABNORMAL"],
        "Probability": [float(row.get("P_NORMAL", 0)), float(row.get("P_WARNING", 0)), float(row.get("P_ABNORMAL", 0))],
    })
    fig = go.Figure(go.Bar(
        x=d["Probability"],
        y=d["State"],
        orientation="h",
        marker_color=[STATE_COLOR[s] for s in d["State"]],
        text=[f"{x:.1%}" for x in d["Probability"]],
        textposition="outside",
    ))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Probability")
    fig.update_layout(height=260, margin=dict(l=20, r=40, t=45, b=30), title="MLP Event +5 Probability", showlegend=False)
    return fig


def decision_evidence_figure(current: str, arima: str, mlp: str, hybrid: str) -> go.Figure:
    labels = ["Current SPC", "ARIMA +5", "MLP +5", "Hybrid +5"]
    states = [clean_state(current), clean_state(arima), clean_state(mlp), clean_state(hybrid)]
    z = [[max(0, STATE_ORDER.get(s, 0)) for s in states]]
    colorscale = [[0.0, "#D5F5E3"], [0.49, "#D5F5E3"], [0.5, "#FCF3CF"], [0.74, "#FCF3CF"], [0.75, "#FADBD8"], [1.0, "#FADBD8"]]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=["Machine Health"], text=[states], texttemplate="%{text}", colorscale=colorscale, zmin=0, zmax=2, showscale=False))
    fig.update_layout(height=170, margin=dict(l=20, r=20, t=35, b=25), title="Integrated Health Check")
    return fig


# -----------------------------------------------------------------------------
# Load required data
# -----------------------------------------------------------------------------
required = ["study_config", "spc", "latest", "arima_forecast", "mlp_latest"]
missing = [k for k in required if not PATHS[k].exists()]
if missing:
    st.error("Required analysis outputs are missing. Run `python run_all.py` from the project folder first.")
    st.code("python run_all.py\npython -m streamlit run dashboard/app.py")
    st.stop()

study_config = read_json(str(PATHS["study_config"]))
spc = read_csv(str(PATHS["spc"]))
nelson = read_csv(str(PATHS["nelson"]))
capability = read_csv(str(PATHS["capability"]))
break_modes = read_csv(str(PATHS["break_modes"]))
latest = read_csv(str(PATHS["latest"]))
arima_fc = read_csv(str(PATHS["arima_forecast"]))
arima_metrics = read_csv(str(PATHS["arima_metrics"]))
mlp_latest = read_csv(str(PATHS["mlp_latest"]))
mlp_metrics = read_csv(str(PATHS["mlp_metrics"]))
mlp_validation = read_csv(str(PATHS["mlp_validation"]))
mlp_cm = read_csv(str(PATHS["mlp_cm"]))
mlp_class = read_csv(str(PATHS["mlp_class"]))
comparison = read_csv(str(PATHS["comparison"]))
split = read_csv(str(PATHS["split"]))
machine_population = read_csv(str(PATHS["machine_population"]))
mapping = read_csv(str(PATHS["mapping"]))

for frame in [spc, latest, arima_fc, mlp_latest, nelson, capability]:
    for c in ["DATE_TIME", "LATEST_TIME", "LAST_OBSERVED_TIME", "LAST_TRIGGER_TIME", "REFERENCE_END"]:
        if c in frame.columns:
            frame[c] = pd.to_datetime(frame[c], errors="coerce")

machines = sorted(spc["MACH_NO"].dropna().astype(str).unique().tolist())
ctqs = [c for c in ["WP", "BS", "SP"] if c in set(spc["CTQ"].astype(str))]

# -----------------------------------------------------------------------------
# Sidebar: only two pages
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Wire Bond SPC-AI")
    page = st.radio("View", ["Production View", "Research / Admin View"], index=0)
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("ANONYMIZED STUDY SCOPE")
    st.write(f"**Package:** {study_config.get('selected_package', 'Package A')}")
    st.write(f"**Wire:** {study_config.get('selected_wire_type', 'Wire Type A')}")
    st.write(f"**Diameter:** {study_config.get('selected_wire_diameter', 'Diameter Group A')}")
    st.write(f"**Machines:** {study_config.get('machine_count', len(machines))}")
    st.write("**CTQs:** WP · BS · SP")


# =============================================================================
# PAGE 1 — PRODUCTION VIEW
# =============================================================================
if page == "Production View":
    st.title("Wire Bond Process Health Monitor")
    st.caption("SPC + Nelson Rules → ARIMA / MLP Fork → AI-assisted engineering decision support")

    c1, c2, c3 = st.columns([1.0, 1.0, 1.1])
    with c1:
        machine = st.selectbox("Machine", machines, index=0)
    with c2:
        available_ctqs = [c for c in ctqs if ((spc["MACH_NO"] == machine) & (spc["CTQ"] == c)).any()]
        ctq = st.selectbox("CTQ", available_ctqs, format_func=lambda x: f"{x} — {CTQ_NAME.get(x, x)}")
    with c3:
        recent_events = st.select_slider("Recent events", options=[20, 30, 40, 50, 75, 100], value=40)

    stream = spc[(spc["MACH_NO"].astype(str) == machine) & (spc["CTQ"].astype(str) == ctq)].copy()
    stream = stream.sort_values("DATE_TIME")
    fc = arima_fc[(arima_fc["MACH_NO"].astype(str) == machine) & (arima_fc["CTQ"].astype(str) == ctq)].copy()
    lrow_df = latest[(latest["MACH_NO"].astype(str) == machine) & (latest["CTQ"].astype(str) == ctq)]
    mrow_df = mlp_latest[(mlp_latest["MACH_NO"].astype(str) == machine) & (mlp_latest["CTQ"].astype(str) == ctq)]

    if stream.empty or lrow_df.empty:
        st.warning("No eligible machine/CTQ output is available for this selection.")
        st.stop()

    lrow = lrow_df.iloc[-1]
    mrow = mrow_df.iloc[-1] if not mrow_df.empty else pd.Series(dtype=object)
    current = clean_state(lrow.get("CURRENT_STATE", stream.iloc[-1].get("PROCESS_STATE", "INSUFFICIENT")))
    arima_state = clean_state(lrow.get("ARIMA_PRED_STATE", "INSUFFICIENT"))
    mlp_state = clean_state(lrow.get("MLP_PRED_STATE", "INSUFFICIENT"))
    hybrid = clean_state(lrow.get("HYBRID_DECISION", more_severe(arima_state, mlp_state)))
    cap_type = str(lrow.get("CAPABILITY_TYPE", "Capability"))
    cap_value = pd.to_numeric(pd.Series([lrow.get("CAPABILITY_VALUE", np.nan)]), errors="coerce").iloc[0]
    p_abnormal = float(pd.to_numeric(pd.Series([lrow.get("P_ABNORMAL", mrow.get("P_ABNORMAL", 0.0))]), errors="coerce").fillna(0).iloc[0])

    recent = stream.tail(recent_events)
    recent_nelson_events = int(bool_series(recent.get("NELSON_ANY", pd.Series(False, index=recent.index))).sum())
    latest_rules = str(stream.iloc[-1].get("NELSON_RULES_TRIGGERED", "") or "")
    action, action_state = action_logic(current, hybrid, p_abnormal, cap_value, recent_nelson_events, latest_rules)

    # Top health cards: answer first, evidence below.
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Current SPC", current)
    k2.metric(cap_type if cap_type and cap_type != "nan" else "Capability", f"{cap_value:.2f}" if np.isfinite(cap_value) else "N/A")
    k3.metric("Recent Nelson", f"{recent_nelson_events} signal events")
    k4.metric("ARIMA +5", arima_state)
    k5.metric("MLP +5", mlp_state)
    k6.metric("DSS Action", action)

    st.markdown('<div class="section-title">1. SPC + Nelson Rules + Next-5 Forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(spc_forecast_figure(stream, fc, recent_events), use_container_width=True, key=f"spc_{machine}_{ctq}")
    st.caption("Nelson Rules are marked directly on the X̄ chart. ARIMA is shown on the same standardized SPC scale for the next five production events. The R chart checks within-subgroup variation.")

    st.markdown('<div class="section-title">2. Supporting Process Evidence</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns([1.0, 1.15, 1.15])
    with e1:
        st.plotly_chart(capability_gauge(cap_type, float(cap_value) if np.isfinite(cap_value) else np.nan), use_container_width=True, key="cap_gauge")
        st.caption(f"Capability judgement: **{capability_judgement(cap_value)}**. The index shown may be Cpk, Cpl or Cpu depending on available specification limits.")
    with e2:
        nstream = nelson[(nelson["MACH_NO"].astype(str) == machine) & (nelson["CTQ"].astype(str) == ctq)] if not nelson.empty else pd.DataFrame()
        st.plotly_chart(nelson_bar(nstream), use_container_width=True, key="nelson_bar")
    with e3:
        bstream = break_modes[(break_modes["MACH_NO"].astype(str) == machine) & (break_modes["CTQ"].astype(str) == ctq)] if not break_modes.empty else pd.DataFrame()
        st.plotly_chart(break_mode_figure(bstream), use_container_width=True, key="break_mode")
        if bstream.empty:
            st.caption("Run the updated pipeline to create break-mode summary output.")
        else:
            mapped = bstream["PASS_FAIL"].fillna("").astype(str).str.strip() if "PASS_FAIL" in bstream.columns else pd.Series([], dtype=str)
            if len(mapped) and mapped.ne("").any():
                fail = bstream[mapped.str.upper().eq("FAIL")]["COUNT"].sum()
                total = bstream["COUNT"].sum()
                if fail == 0:
                    st.success("No mapped FAIL break mode was found for this selected scope.")
                else:
                    st.warning(f"Mapped FAIL break-mode count: {int(fail)} of {int(total)}.")
            else:
                st.info("Break modes A–E are shown descriptively. Physical PASS/FAIL judgement is not made until the approved mapping file is completed.")

    st.markdown('<div class="section-title">3. MLP Future-State Prediction and Integrated Health Check</div>', unsafe_allow_html=True)
    p1, p2 = st.columns([1.0, 1.35])
    with p1:
        st.plotly_chart(mlp_probability_figure(lrow), use_container_width=True, key="mlp_prob")
        st.caption(f"MLP predicted state at event +5: **{mlp_state}**")
    with p2:
        st.plotly_chart(decision_evidence_figure(current, arima_state, mlp_state, hybrid), use_container_width=True, key="decision_evidence")
        arima_h5 = float(pd.to_numeric(pd.Series([lrow.get("FORECAST_H5_Z", np.nan)]), errors="coerce").iloc[0])
        if np.isfinite(arima_h5):
            st.caption(f"ARIMA fifth-event standardized forecast: **{arima_h5:.2f}σ**. Hybrid uses the more severe of the two predictive branches.")

    st.markdown('<div class="section-title">4. AI-Assisted Engineering Decision</div>', unsafe_allow_html=True)
    css_class = {"NORMAL": "health-normal", "WARNING": "health-warning", "ABNORMAL": "health-abnormal"}.get(action_state, "health-warning")
    focus = engineering_focus(ctq, action)
    st.markdown(
        f"""
<div class="{css_class}">
<b>{machine} · {ctq} — {action}</b><br>
Current SPC: <b>{current}</b> &nbsp; | &nbsp; ARIMA +5: <b>{arima_state}</b> &nbsp; | &nbsp; MLP +5: <b>{mlp_state}</b> &nbsp; | &nbsp; Hybrid: <b>{hybrid}</b><br>
Capability: <b>{cap_type} {cap_value:.2f}</b> if available &nbsp; | &nbsp; Recent Nelson signal events: <b>{recent_nelson_events}</b>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("**Suggested engineering focus**")
    for item in focus:
        st.write(f"• {item}")
    st.caption("Decision support only. Nelson Rules and model outputs indicate where engineering should review; they do not prove a physical root cause and do not automatically stop equipment or trigger PM.")


# =============================================================================
# PAGE 2 — RESEARCH / ADMIN VIEW
# =============================================================================
else:
    st.title("Research / Admin Validation View")
    st.caption("One-page evidence trail for VIVA: RQ1 → RQ2 → RQ3")

    # Simple research-design alignment; no extra tabs or pages.
    q1, a1, q2, a2, q3 = st.columns([1.0, 0.18, 1.0, 0.18, 1.0])
    with q1:
        st.markdown('<div class="flow-card"><b>RQ1 / RO1</b><br>Data + X̄–R SPC<br>+ Nelson Rules<br><b>Current condition</b></div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="flow-card"><b>RQ2 / RO2</b><br>ARIMA + MLP<br>parallel Fork<br><b>Future condition</b></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="flow-card"><b>RQ3 / RO3</b><br>Model evaluation<br>+ Hybrid DSS<br><b>Engineering support</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Dataset and SPC Evidence</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        if machine_population.empty:
            st.info("Machine population output is not available.")
        else:
            totals = pd.DataFrame({
                "CTQ": [c for c in ["WP", "BS", "SP"] if c in machine_population.columns],
                "Events": [machine_population[c].sum() for c in ["WP", "BS", "SP"] if c in machine_population.columns],
            })
            fig = px.bar(totals, x="CTQ", y="Events", text="Events", title="Selected Study Population by CTQ")
            fig.update_traces(marker_color="#5470C6", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=30, r=10, t=55, b=30))
            st.plotly_chart(fig, use_container_width=True, key="admin_ctq")
    with d2:
        states = spc[spc["PROCESS_STATE"].isin(["NORMAL", "WARNING", "ABNORMAL"])]["PROCESS_STATE"].value_counts().reindex(["NORMAL", "WARNING", "ABNORMAL"], fill_value=0).reset_index()
        states.columns = ["State", "Events"]
        fig = px.pie(states, names="State", values="Events", hole=0.5, title="SPC Process-State Distribution", color="State", color_discrete_map=STATE_COLOR)
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True, key="admin_states")

    n1, n2 = st.columns(2)
    with n1:
        if not nelson.empty:
            agg = nelson.groupby("RULE", as_index=False).agg(TRIGGERS=("TRIGGER_COUNT", "sum"), EPISODES=("EPISODE_COUNT", "sum"))
            agg["RULE_NO"] = agg["RULE"].str.extract(r"(\d+)").astype(int)
            agg = agg.sort_values("RULE_NO")
            fig = go.Figure()
            fig.add_bar(x=agg["RULE"], y=agg["TRIGGERS"], name="Triggers", marker_color="#5470C6")
            fig.add_bar(x=agg["RULE"], y=agg["EPISODES"], name="Episodes", marker_color="#91CC75")
            fig.update_layout(barmode="group", height=340, title="Overall Nelson Rule Results", margin=dict(l=30, r=10, t=55, b=30))
            st.plotly_chart(fig, use_container_width=True, key="admin_nelson")
    with n2:
        if not split.empty:
            val_col = "ROWS" if "ROWS" in split.columns else split.select_dtypes(include="number").columns[0]
            fig = px.bar(split, x="SPLIT", y=val_col, text=val_col, title="Training, Validation and Test Dataset")
            fig.update_traces(marker_color="#73C0DE", textposition="outside")
            fig.update_layout(height=340, margin=dict(l=30, r=10, t=55, b=30))
            st.plotly_chart(fig, use_container_width=True, key="admin_split")

    st.markdown('<div class="section-title">2. ARIMA and MLP Validation</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        if not arima_metrics.empty:
            d = arima_metrics.copy()
            d["Stream"] = d["MACH_NO"].astype(str) + "-" + d["CTQ"].astype(str)
            d = d.sort_values("H5_RMSE")
            fig = go.Figure()
            fig.add_bar(x=d["Stream"], y=d["H5_RMSE"], name="ARIMA RMSE", marker_color="#5470C6")
            fig.add_bar(x=d["Stream"], y=d["NAIVE_H5_RMSE"], name="Naïve RMSE", marker_color="#FAC858")
            fig.update_layout(barmode="group", height=390, title="ARIMA Event +5 RMSE vs Naïve", margin=dict(l=30, r=10, t=55, b=80), xaxis_tickangle=-55)
            st.plotly_chart(fig, use_container_width=True, key="admin_arima")
    with m2:
        if not mlp_cm.empty:
            pivot = mlp_cm.pivot(index="ACTUAL_STATE", columns="PREDICTED_STATE", values="COUNT").reindex(index=["NORMAL", "WARNING", "ABNORMAL"], columns=["NORMAL", "WARNING", "ABNORMAL"]).fillna(0)
            fig = px.imshow(pivot, text_auto=True, aspect="auto", title="MLP Confusion Matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
            fig.update_layout(height=390, margin=dict(l=30, r=30, t=55, b=30))
            st.plotly_chart(fig, use_container_width=True, key="admin_mlp_cm")

    v1, v2 = st.columns(2)
    with v1:
        if not mlp_validation.empty:
            d = mlp_validation.copy()
            d["Model"] = d["HIDDEN_LAYERS"].astype(str) + " / α=" + d["ALPHA"].astype(str)
            fig = px.bar(d, x="Model", y="VALIDATION_MACRO_F1", text="VALIDATION_MACRO_F1", title="MLP Model Selection — Validation Macro F1")
            fig.update_traces(marker_color="#EE6666", texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=330, margin=dict(l=30, r=10, t=55, b=60))
            st.plotly_chart(fig, use_container_width=True, key="admin_mlp_val")
    with v2:
        if not mlp_class.empty:
            d = mlp_class.melt(id_vars="STATE", value_vars=["PRECISION", "RECALL", "F1"], var_name="Metric", value_name="Score")
            fig = px.bar(d, x="STATE", y="Score", color="Metric", barmode="group", title="MLP Class-Level Performance")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(height=330, margin=dict(l=30, r=10, t=55, b=30))
            st.plotly_chart(fig, use_container_width=True, key="admin_mlp_class")

    st.markdown('<div class="section-title">3. Fork / Hybrid Model Comparison</div>', unsafe_allow_html=True)
    if not comparison.empty:
        metrics_to_show = [c for c in ["ACCURACY", "BALANCED_ACCURACY", "MACRO_F1"] if c in comparison.columns]
        d = comparison.melt(id_vars="MODEL", value_vars=metrics_to_show, var_name="Metric", value_name="Score")
        fig = px.bar(d, x="MODEL", y="Score", color="Metric", barmode="group", text_auto=".3f", title="Common-Test Comparison: SPC Persistence vs ARIMA vs MLP vs Hybrid")
        fig.update_yaxes(range=[0, max(0.7, float(d["Score"].max()) * 1.18)])
        fig.update_layout(height=430, margin=dict(l=30, r=10, t=60, b=30))
        st.plotly_chart(fig, use_container_width=True, key="admin_compare")
        st.info("VIVA interpretation: no single method is best for every metric. ARIMA provides the time-series forecast, MLP provides nonlinear state classification, and the Fork combines complementary evidence for engineering decision support rather than assuming automatic accuracy improvement.")

    st.markdown('<div class="section-title">4. Dashboard Role in the Research Design</div>', unsafe_allow_html=True)
    st.write("**SPC + Nelson Rules** explain the current process condition. **ARIMA** forecasts the standardized process trend for the next five events. **MLP** predicts the process state at the fifth event. The **Hybrid DSS** combines both predictive branches and presents an engineering review recommendation.")
    st.caption("This prototype supports engineering judgement. It does not automatically change bonding parameters, stop equipment, disposition product or prove root cause.")
