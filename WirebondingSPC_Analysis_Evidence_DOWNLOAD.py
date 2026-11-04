# %% [markdown]
# # Wirebonding SPC-AI — Analysis Evidence Workbook
#
# **Purpose**
#
# This single VS Code interactive Python file follows the research design used in the
# Master Project and provides one runnable section for each major analysis stage.
#
# Research flow:
#
# **Anonymous historical SPC data**
# → **Data cleaning and study population selection**
# → **X̄–R SPC + Nelson Rules + capability + break-mode review**
# → **Feature engineering + train/validation/test preparation**
# → **Fork: ARIMA forecasting || MLP future-state classification**
# → **Common-test evaluation + Hybrid integration**
# → **Engineering DSS / machine-health output**
#
# The file is designed for VS Code. Every `# %%` block is an executable cell and every
# `# %% [markdown]` block is an explanation. Run the cells from top to bottom once.
# After that, any analysis section can be re-run individually for screenshots or thesis evidence.
#
# **Important research rule**
#
# ARIMA and MLP are parallel branches. The MLP does **not** use the ARIMA prediction as an
# input. ARIMA forecasts the standardized SPC trend, while MLP predicts the state of the
# fifth next event from engineered SPC features. They are combined only at decision level.

# %%
# SECTION 0 — COMMON SETUP
# Run this cell first whenever a new VS Code Interactive session is started.

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from config import DATA_FILE, PROJECT_ROOT
from src.data_cleaning import (
    load_and_clean_data,
    select_study_population,
    build_eda_tables,
)
from src.anonymization import anonymize_study, public_study_config
from src.features import (
    MODEL_FEATURES,
    build_features,
    derive_time_split_meta,
    chronological_split,
)
from src.spc import apply_spc
from src.models import train_arima, train_mlp, evaluate_fork, build_latest_status
from src.thesis_outputs import main as generate_thesis_outputs

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

EVIDENCE_DIR = PROJECT_ROOT / "outputs" / "evidence_by_section"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["NORMAL", "WARNING", "ABNORMAL"]
D4_N5 = 2.114


def save_show(fig, filename: str):
    """Save a report-ready PNG and show it inside the VS Code Interactive window."""
    path = EVIDENCE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.show()
    print(f"Saved evidence: {path}")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.upper().isin(["TRUE", "1", "YES", "Y"])


def section_heading(title: str, chapter_use: str):
    display(Markdown(f"## {title}\n**Suggested thesis use:** {chapter_use}"))


print("Project root:", PROJECT_ROOT)
print("Anonymous source:", DATA_FILE)
print("Evidence folder:", EVIDENCE_DIR)

# %% [markdown]
# ---
# # SECTION 1 — Data Cleaning and Data Understanding
#
# **Research alignment:** RQ1 / RO1 — prepare historical SPC data for rule-based analysis.
#
# This section checks the anonymized event-level dataset, standardizes the input structure,
# verifies subgroup measurements, recalculates the event statistics used by SPC, and confirms
# that the data are arranged in production sequence.
#
# **Evidence to capture**
# - Cleaning audit table
# - Raw-to-clean event summary
# - Event-level dataset preview
#
# **Suggested thesis placement**
# - Chapter 4: Data Understanding and Preparation
# - Chapter 5: Data preparation result

# %%
section_heading("1. Data Cleaning and Data Understanding", "Chapter 4 §4.2 / Chapter 5 data-preparation evidence")

cleaned, cleaning_audit, normalization_audit = load_and_clean_data(DATA_FILE)

print("Cleaned event-level dataset shape:", cleaned.shape)
display(cleaning_audit)

display(cleaned[[c for c in [
    "CTQ", "MACH_NO", "DATE_TIME", "PACKAGE", "WIRE_TYPE", "WIRE_DIA",
    "VALUE1", "VALUE2", "VALUE3", "VALUE4", "VALUE5",
    "AVG", "SIGMA", "RANGE", "LSL", "USL", "BREAK_MODE_PATTERN"
] if c in cleaned.columns]].head(10))

# Coding evidence: source event rows versus clean event-level records by CTQ.
# The anonymous project starts from event-level data, so SOURCE_EVENTS is the correct
# pre-cleaning reference. If the private raw-workbook route is used, RAW_ROWS may exist instead.
plot_df = cleaning_audit.copy()
source_col = "RAW_ROWS" if "RAW_ROWS" in plot_df.columns else "SOURCE_EVENTS"
if {"CTQ", source_col, "CLEAN_EVENTS"}.issubset(plot_df.columns):
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, plot_df[source_col], 0.4, label="Source records")
    ax.bar(x + 0.2, plot_df["CLEAN_EVENTS"], 0.4, label="Clean event-level records")
    ax.set_xticks(x, plot_df["CTQ"])
    ax.set_xlabel("CTQ")
    ax.set_ylabel("Rows / events")
    ax.set_title("Data Cleaning and Event-Level Validation")
    ax.legend()
    save_show(fig, "01_data_cleaning_summary.png")

cleaning_audit.to_csv(EVIDENCE_DIR / "01_data_cleaning_audit.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “The original records are converted into consistent event-level observations before any
# modelling is performed. The same cleaned records are then used by SPC, ARIMA and MLP,
# which keeps the analytical flow consistent.”

# %% [markdown]
# ---
# # SECTION 2 — Study Population Selection
#
# The research does not select one machine first. The study population is selected using the
# **Package + Wire Type + Wire Diameter** combination with the highest usable record count.
# All wire-bond machines associated with that selected population are retained.
#
# In the thesis-facing dataset the identifiers are already anonymized, therefore the result is
# expressed as Package A / Wire Type A / Diameter Group A and WBxx machine codes.
#
# **Evidence to capture**
# - Selected population summary
# - CTQ event distribution
# - Machine event distribution

# %%
section_heading("2. Study Population Selection", "Chapter 4 §4.2 / Chapter 5 §5.2")

study_raw, study_config_raw, population_audit, machine_population_raw = select_study_population(cleaned)
study, machine_map = anonymize_study(study_raw)
study_config = public_study_config(study_config_raw, machine_map, study)

print(json.dumps(study_config, indent=2))

population_summary = pd.DataFrame({
    "CTQ": ["WP", "BS", "SP"],
    "EVENTS": [
        int((study["CTQ"] == "WP").sum()),
        int((study["CTQ"] == "BS").sum()),
        int((study["CTQ"] == "SP").sum()),
    ],
})
display(population_summary)

fig, ax = plt.subplots(figsize=(6.8, 4.5))
ax.bar(population_summary["CTQ"], population_summary["EVENTS"])
ax.set_title("Selected Study Population by CTQ")
ax.set_xlabel("Critical to Quality (CTQ)")
ax.set_ylabel("Events")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
save_show(fig, "02_selected_population_by_ctq.png")

machine_population = study.groupby(["MACH_NO", "CTQ"]).size().unstack(fill_value=0).reset_index()
for ctq in ["WP", "BS", "SP"]:
    if ctq not in machine_population.columns:
        machine_population[ctq] = 0
machine_population["TOTAL_EVENTS"] = machine_population[["WP", "BS", "SP"]].sum(axis=1)
machine_population = machine_population.sort_values("TOTAL_EVENTS", ascending=False)
display(machine_population)

ax = machine_population.set_index("MACH_NO")[["WP", "BS", "SP"]].plot(
    kind="bar", stacked=True, figsize=(11, 4.8)
)
ax.set_title("SPC Event Distribution Across Anonymized Wire-Bond Machines")
ax.set_xlabel("Anonymized Machine")
ax.set_ylabel("Events")
plt.xticks(rotation=45, ha="right")
save_show(ax.get_figure(), "02_machine_event_distribution.png")

population_summary.to_csv(EVIDENCE_DIR / "02_selected_population_by_ctq.csv", index=False)
machine_population.to_csv(EVIDENCE_DIR / "02_machine_event_distribution.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “Population selection is based on a common product/wire configuration rather than on one
# preferred machine. This reduces product-mix effects while retaining the machine-to-machine
# variation required by the research.”

# %% [markdown]
# ---
# # SECTION 3 — X̄–R SPC, Nelson Rules, Capability and Break-Mode Analysis
#
# This is the statistical foundation of the framework.
#
# - **X̄ chart:** monitors the subgroup mean / process centering.
# - **R chart:** monitors within-subgroup variation.
# - **Nelson N1–N8:** detects non-random process patterns.
# - **Capability:** provides supporting information relative to the applicable specification.
# - **Break mode:** provides supporting wire-bond engineering evidence.
#
# Process-state logic used by the research:
#
# - **Abnormal** = N1 OR control-limit violation OR specification violation
# - **Warning** = N2–N8 only, without an Abnormal condition
# - **Normal** = none of the above
#
# **Evidence to capture**
# - Representative X̄ chart with Nelson Rule labels on the actual points
# - Corresponding normalized R chart
# - Nelson trigger/episode summary
# - Process-state distribution
# - Capability plot
# - Break-mode distribution

# %%
section_heading("3. SPC + Nelson Rules + Engineering Evidence", "Chapter 4 §4.3 / Chapter 5 §5.2")

split_meta = derive_time_split_meta(study)
spc, limits, nelson_summary, machine_spc_summary, capability = apply_spc(study, split_meta)

spc_available = spc[bool_series(spc["SPC_AVAILABLE"])].copy()
print(f"SPC-available events: {len(spc_available)} / {len(spc)}")

# 3A. Process states
state_counts = (
    spc_available["PROCESS_STATE"].value_counts()
    .reindex(STATE_ORDER, fill_value=0)
    .rename_axis("STATE")
    .reset_index(name="EVENTS")
)
state_counts["PERCENT"] = 100 * state_counts["EVENTS"] / state_counts["EVENTS"].sum()
display(state_counts)

fig, ax = plt.subplots(figsize=(6.8, 4.5))
ax.bar(state_counts["STATE"], state_counts["EVENTS"])
ax.set_title("SPC Process-State Distribution")
ax.set_ylabel("Events")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
save_show(fig, "03_process_state_distribution.png")

# 3B. Nelson summary
nelson_total = (
    nelson_summary.groupby("RULE", as_index=False)[["TRIGGER_COUNT", "EPISODE_COUNT"]]
    .sum()
)
nelson_total["RULE_NUM"] = pd.to_numeric(nelson_total["RULE"].str.extract(r"(\d+)")[0], errors="coerce")
nelson_total = nelson_total.sort_values("RULE_NUM").drop(columns="RULE_NUM")
display(nelson_total)

x = np.arange(len(nelson_total))
fig, ax = plt.subplots(figsize=(8.5, 4.7))
ax.bar(x - 0.2, nelson_total["TRIGGER_COUNT"], 0.4, label="Triggers")
ax.bar(x + 0.2, nelson_total["EPISODE_COUNT"], 0.4, label="Episodes")
ax.set_xticks(x, nelson_total["RULE"])
ax.set_title("Nelson Rule Triggers and Episodes")
ax.set_xlabel("Nelson Rule")
ax.set_ylabel("Count")
ax.legend()
save_show(fig, "03_nelson_rule_summary.png")

# 3C. Select a representative Machine–CTQ stream with strong Nelson evidence.
spc_available["_NELSON_SIGNAL"] = bool_series(spc_available["NELSON_ANY"]).astype(int)
stream_rank = (
    spc_available.groupby(["MACH_NO", "CTQ"])
    .agg(NELSON_SIGNALS=("_NELSON_SIGNAL", "sum"), EVENTS=("_NELSON_SIGNAL", "size"))
    .sort_values(["NELSON_SIGNALS", "EVENTS"], ascending=False)
)
rep_machine, rep_ctq = stream_rank.index[0]
rep = spc_available[(spc_available["MACH_NO"] == rep_machine) & (spc_available["CTQ"] == rep_ctq)].copy()
rep["DATE_TIME"] = pd.to_datetime(rep["DATE_TIME"], errors="coerce")
rep = rep.sort_values("DATE_TIME").tail(60).reset_index(drop=True)
print("Representative stream:", rep_machine, rep_ctq)

# 3D. Standardized Xbar chart with Nelson labels.
x = np.arange(1, len(rep) + 1)
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.plot(x, rep["STANDARDIZED_DEVIATION"], marker="o", markersize=3, linewidth=1.1, label="Actual standardized X̄")
for y, label, linestyle in [
    (3, "+3σ UCL", "--"), (2, "+2σ", ":"), (1, "+1σ", ":"),
    (0, "CL", "-"), (-1, "-1σ", ":"), (-2, "-2σ", ":"), (-3, "-3σ LCL", "--")
]:
    ax.axhline(y, linestyle=linestyle, linewidth=0.9, label=label if y in [3, 0, -3] else None)

sig = bool_series(rep["NELSON_ANY"])
if sig.any():
    ax.scatter(x[sig.to_numpy()], rep.loc[sig, "STANDARDIZED_DEVIATION"], marker="*", s=90, label="Nelson signal")
    if "NELSON_RULES_TRIGGERED" in rep.columns:
        for xi, yi, label in zip(
            x[sig.to_numpy()],
            rep.loc[sig, "STANDARDIZED_DEVIATION"],
            rep.loc[sig, "NELSON_RULES_TRIGGERED"].fillna("")
        ):
            if str(label).strip():
                ax.annotate(str(label), (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7)

ax.set_title(f"Standardized X̄ Chart with Nelson Rules — {rep_machine} / {rep_ctq}")
ax.set_xlabel("Production event sequence")
ax.set_ylabel("Standardized X̄ deviation (σ)")
ax.legend(ncol=3, fontsize=8)
save_show(fig, "03_representative_xbar_nelson.png")

# 3E. Corresponding normalized R chart.
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(x, rep["RANGE_RATIO"], marker="o", markersize=3, linewidth=1.1, label="R / R̄")
ax.axhline(D4_N5, linestyle="--", label=f"UCL(R) = {D4_N5:.3f} × R̄")
ax.axhline(1.0, linewidth=1.0, label="R̄")
ax.axhline(0.0, linestyle=":", label="LCL = 0")
ax.set_title(f"Normalized R Chart — {rep_machine} / {rep_ctq}")
ax.set_xlabel("Production event sequence")
ax.set_ylabel("Range ratio (R / R̄)")
ax.legend()
save_show(fig, "03_representative_r_chart.png")

# 3F. Capability analysis.
cap_valid = capability.dropna(subset=["CAPABILITY_VALUE"]).copy()
cap_summary = pd.DataFrame()
if not cap_valid.empty:
    cap_valid["STREAM"] = cap_valid["MACH_NO"].astype(str) + "-" + cap_valid["CTQ"].astype(str)
    cap_summary = (
        cap_valid.groupby("STREAM", as_index=False)["CAPABILITY_VALUE"]
        .min()
        .sort_values("CAPABILITY_VALUE")
    )
    display(cap_summary)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(cap_summary["STREAM"], cap_summary["CAPABILITY_VALUE"])
    ax.axhline(1.33, linestyle="--", label="Reference 1.33")
    ax.axhline(1.00, linestyle=":", label="Reference 1.00")
    ax.set_title("Minimum Valid Capability Index by Machine–CTQ Stream")
    ax.set_xlabel("Anonymized Machine–CTQ")
    ax.set_ylabel("Capability index")
    ax.legend()
    plt.xticks(rotation=55, ha="right", fontsize=7)
    save_show(fig, "03_capability_by_stream.png")

# 3G. Break-mode analysis directly from generic A–E counts in SPC output.
mode_cols = [c for c in ["MODE_A_COUNT", "MODE_B_COUNT", "MODE_C_COUNT", "MODE_D_COUNT", "MODE_E_COUNT"] if c in spc.columns]
break_mode_summary = pd.DataFrame()
if mode_cols:
    rows = []
    for col in mode_cols:
        rows.append({"MODE": col.replace("MODE_", "").replace("_COUNT", ""), "COUNT": int(spc[col].fillna(0).sum())})
    break_mode_summary = pd.DataFrame(rows)
    display(break_mode_summary)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(break_mode_summary["MODE"], break_mode_summary["COUNT"])
    ax.set_title("Generic Break-Mode Distribution")
    ax.set_xlabel("Break Mode")
    ax.set_ylabel("Count")
    save_show(fig, "03_break_mode_distribution.png")

# Save tables used in this section.
state_counts.to_csv(EVIDENCE_DIR / "03_process_state_distribution.csv", index=False)
nelson_total.to_csv(EVIDENCE_DIR / "03_nelson_rule_summary.csv", index=False)
cap_summary.to_csv(EVIDENCE_DIR / "03_capability_by_stream.csv", index=False)
break_mode_summary.to_csv(EVIDENCE_DIR / "03_break_mode_distribution.csv", index=False)

# %% [markdown]
# **Wire-bond engineering interpretation**
#
# - X̄ movement shows whether the bond-strength process is shifting relative to its reference.
# - R shows whether the five measurements in a subgroup are becoming less consistent.
# - Nelson labels explain *why* a point or sequence is treated as unusual.
# - Capability is supporting evidence; it should not replace control-chart interpretation.
# - Break mode is supporting mechanical evidence. It should not be used alone to claim a physical root cause.

# %% [markdown]
# ---
# # SECTION 4 — Feature Engineering and Model Dataset Preparation
#
# The MLP uses current SPC evidence and recent process history. The target is the state of the
# **fifth subsequent event in the same Machine–CTQ stream**.
#
# The data are separated by production sequence rather than by random sampling. Observations
# close to a dataset boundary are removed if their +5 target would cross into the next dataset.
# This prevents future information from leaking into model development.
#
# **Evidence to capture**
# - Feature-group table
# - Training / validation / test distribution
# - Future-state class distribution by dataset

# %%
section_heading("4. Feature Engineering and Model Dataset", "Chapter 4 §4.3–4.4 / Chapter 5 §5.3")

features = build_features(spc)
labelled, split_meta, split_summary, split_class_summary = chronological_split(features, split_meta)

feature_groups = pd.DataFrame([
    ["Current statistics", "AVG, SIGMA, RANGE, CV"],
    ["SPC position", "STANDARDIZED_DEVIATION, RANGE_RATIO"],
    ["Capability", "CAPABILITY_VALUE"],
    ["Nelson evidence", "NELSON_RULE_COUNT, N1–N8"],
    ["Break-mode evidence", "MODE_A_COUNT–MODE_E_COUNT"],
    ["Lag / rolling history", "Lagged Z, rolling mean/std, moving average"],
    ["Previous state history", "Previous Warning/Abnormal/Nelson rates"],
    ["Categorical", "Anonymized Machine ID, CTQ"],
], columns=["FEATURE_GROUP", "EXAMPLES"])

display(feature_groups)
print("\nActual model-feature list:")
for item in MODEL_FEATURES:
    print(" -", item)

print("\nDataset split:")
display(split_summary)
print("\nFuture-state class distribution by split:")
display(split_class_summary)

fig, ax = plt.subplots(figsize=(7, 4.5))
y_col = "PERCENT_OF_LABELLED" if "PERCENT_OF_LABELLED" in split_summary.columns else "ROWS"
ax.bar(split_summary["SPLIT"], split_summary[y_col])
ax.set_title("Training, Validation, Test and Boundary-Removed Observations")
ax.set_ylabel("Percent of labelled observations" if y_col == "PERCENT_OF_LABELLED" else "Rows")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}" if y_col == "PERCENT_OF_LABELLED" else f"{int(p.get_height())}",
                (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points")
save_show(fig, "04_dataset_split.png")

split_pivot = split_class_summary.pivot(index="SPLIT", columns="STATE", values="ROWS").fillna(0)
split_pivot = split_pivot.reindex(columns=STATE_ORDER, fill_value=0)
ax = split_pivot.plot(kind="bar", figsize=(8, 4.6))
ax.set_title("Future-State Class Distribution by Dataset")
ax.set_xlabel("Dataset")
ax.set_ylabel("Rows")
plt.xticks(rotation=0)
save_show(ax.get_figure(), "04_split_class_distribution.png")

feature_groups.to_csv(EVIDENCE_DIR / "04_feature_groups.csv", index=False)
split_summary.to_csv(EVIDENCE_DIR / "04_dataset_split.csv", index=False)
split_class_summary.to_csv(EVIDENCE_DIR / "04_split_class_distribution.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “The MLP is not trained from raw force values only. It receives SPC position, rule evidence,
# capability and recent historical behaviour. The +5 target is created within the same machine
# and CTQ stream, and the time-ordered split prevents future leakage.”

# %% [markdown]
# ---
# # SECTION 5 — ARIMA Time-Series Forecasting Branch
#
# ARIMA is fitted independently for each eligible Machine–CTQ stream. Candidate ARIMA orders
# are selected using validation event+5 RMSE. Final test performance is compared against a
# simple naïve fifth-event forecast.
#
# **Evidence to capture**
# - Candidate-order validation table
# - ARIMA vs naïve RMSE chart
# - Next-five-event forecast displayed on the standardized SPC scale

# %%
section_heading("5. ARIMA Forecasting", "Chapter 4 §4.4.1 / Chapter 5 §5.3")

arima = train_arima(features, split_meta)

print("ARIMA validation candidate results:")
display(arima.selection)
print("\nARIMA final test metrics:")
display(arima.metrics)

arima_summary = pd.DataFrame({
    "ITEM": ["Evaluated streams", "ARIMA better than naïve", "Adequate evidence streams"],
    "RESULT": [
        len(arima.metrics),
        int((arima.metrics["H5_RMSE"] < arima.metrics["NAIVE_H5_RMSE"]).sum()) if not arima.metrics.empty else 0,
        int(arima.metrics["EVIDENCE_LEVEL"].eq("ADEQUATE").sum()) if "EVIDENCE_LEVEL" in arima.metrics.columns else np.nan,
    ],
})
display(arima_summary)

if not arima.metrics.empty:
    d = arima.metrics.copy()
    d["STREAM"] = d["MACH_NO"].astype(str) + "-" + d["CTQ"].astype(str)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - 0.2, d["H5_RMSE"], 0.4, label="ARIMA H5 RMSE")
    ax.bar(x + 0.2, d["NAIVE_H5_RMSE"], 0.4, label="Naïve H5 RMSE")
    ax.set_xticks(x, d["STREAM"], rotation=55, ha="right", fontsize=7)
    ax.set_title("ARIMA Fifth-Event RMSE Compared with Naïve Forecast")
    ax.set_ylabel("RMSE")
    ax.legend()
    save_show(fig, "05_arima_rmse_vs_naive.png")

# Choose an adequate stream, preferably one where ARIMA beats naïve.
if not arima.metrics.empty and not arima.forecast.empty:
    candidates = arima.metrics.copy()
    if "EVIDENCE_LEVEL" in candidates.columns:
        adequate = candidates[candidates["EVIDENCE_LEVEL"] == "ADEQUATE"]
        if not adequate.empty:
            candidates = adequate
    improved = candidates[candidates["H5_RMSE"] < candidates["NAIVE_H5_RMSE"]]
    chosen = improved.iloc[0] if not improved.empty else candidates.iloc[0]
    arima_machine, arima_ctq = chosen["MACH_NO"], chosen["CTQ"]

    hist = spc_available[(spc_available["MACH_NO"] == arima_machine) & (spc_available["CTQ"] == arima_ctq)].copy()
    hist["DATE_TIME"] = pd.to_datetime(hist["DATE_TIME"], errors="coerce")
    hist = hist.sort_values("DATE_TIME").tail(40).reset_index(drop=True)
    future = arima.forecast[(arima.forecast["MACH_NO"] == arima_machine) & (arima.forecast["CTQ"] == arima_ctq)].copy()
    future = future.sort_values("HORIZON_EVENT")

    hx = np.arange(1, len(hist) + 1)
    fx = np.arange(len(hist) + 1, len(hist) + 1 + len(future))
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.plot(hx, hist["STANDARDIZED_DEVIATION"], marker="o", markersize=3, linewidth=1.1, label="Actual standardized X̄")
    ax.plot(fx, future["FORECAST_Z"], marker="o", linestyle="--", linewidth=1.3, label="ARIMA next 5")
    if {"LOWER_95", "UPPER_95"}.issubset(future.columns):
        ax.fill_between(fx, future["LOWER_95"], future["UPPER_95"], alpha=0.18, label="95% prediction interval")
    sig = bool_series(hist["NELSON_ANY"]) if "NELSON_ANY" in hist.columns else pd.Series(False, index=hist.index)
    if sig.any():
        ax.scatter(hx[sig.to_numpy()], hist.loc[sig, "STANDARDIZED_DEVIATION"], marker="*", s=75, label="Nelson signal")
    for y, ls in [(3, "--"), (2, ":"), (1, ":"), (0, "-"), (-1, ":"), (-2, ":"), (-3, "--")]:
        ax.axhline(y, linestyle=ls, linewidth=0.8)
    ax.axvline(len(hist) + 0.5, linestyle="--", linewidth=1.0)
    ax.text(len(hist) + 0.8, ax.get_ylim()[1] * 0.88, "Forecast →", fontsize=9)
    ax.set_title(f"Recent SPC Trend + ARIMA Next-Five Forecast — {arima_machine} / {arima_ctq}")
    ax.set_xlabel("Production event sequence")
    ax.set_ylabel("Standardized X̄ deviation (σ)")
    ax.legend(ncol=3, fontsize=8)
    save_show(fig, "05_arima_next5_on_spc.png")

arima.selection.to_csv(EVIDENCE_DIR / "05_arima_validation_candidates.csv", index=False)
arima.metrics.to_csv(EVIDENCE_DIR / "05_arima_test_metrics.csv", index=False)
arima_summary.to_csv(EVIDENCE_DIR / "05_arima_summary.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “ARIMA does not predict the class directly. It forecasts the standardized future SPC movement.
# The fifth event is used as the common horizon so that its result can be compared with the
# MLP fifth-event state prediction.”

# %% [markdown]
# ---
# # SECTION 6 — MLP Future-State Classification Branch
#
# MLP is independent of ARIMA. It receives the engineered SPC features and predicts the
# process state of the fifth next event: **Normal / Warning / Abnormal**.
#
# Candidate networks are selected using **validation Macro F1**. Oversampling is used only
# for development data; the test set remains untouched until final evaluation.
#
# **Evidence to capture**
# - Validation Macro F1 by candidate network
# - Final test metrics
# - Confusion matrix
# - Class-level Precision / Recall / F1

# %%
section_heading("6. MLP Future-State Classification", "Chapter 4 §4.4.2 / Chapter 5 §5.3")

mlp = train_mlp(features, labelled, PROJECT_ROOT / "outputs" / "models")

print("MLP validation results:")
display(mlp.validation)
print("\nMLP test metrics:")
display(mlp.metrics)
print("\nMLP class-level performance:")
display(mlp.class_report)
print("\nMLP confusion matrix (long format):")
display(mlp.confusion)

# Validation chart.
val = mlp.validation.copy()
labels = [f"{r.HIDDEN_LAYERS}\nα={r.ALPHA}" for r in val.itertuples()]
fig, ax = plt.subplots(figsize=(7.8, 4.5))
ax.bar(labels, val["VALIDATION_MACRO_F1"])
ax.set_title("MLP Model Selection by Validation Macro F1")
ax.set_ylabel("Macro F1")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
save_show(fig, "06_mlp_validation.png")

# Test-metric chart.
metric_cols = [c for c in ["ACCURACY", "BALANCED_ACCURACY", "MACRO_PRECISION", "MACRO_RECALL", "MACRO_F1", "ROC_AUC_OVR_MACRO"] if c in mlp.metrics.columns]
metric_values = mlp.metrics.iloc[0][metric_cols].astype(float)
fig, ax = plt.subplots(figsize=(9, 4.7))
ax.bar(metric_cols, metric_values)
ax.set_ylim(0, 1)
ax.set_title("MLP Final Test Performance")
ax.set_ylabel("Score")
plt.xticks(rotation=30, ha="right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=8)
save_show(fig, "06_mlp_test_metrics.png")

# Confusion matrix.
cm = mlp.confusion.pivot(index="ACTUAL_STATE", columns="PREDICTED_STATE", values="COUNT").reindex(index=STATE_ORDER, columns=STATE_ORDER).fillna(0)
fig, ax = plt.subplots(figsize=(6, 5.2))
im = ax.imshow(cm.values)
ax.set_xticks(range(3), cm.columns)
ax.set_yticks(range(3), cm.index)
ax.set_xlabel("Predicted state")
ax.set_ylabel("Actual state")
ax.set_title("MLP Test Confusion Matrix")
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax)
save_show(fig, "06_mlp_confusion_matrix.png")

# Class-level performance.
class_cols = [c for c in ["PRECISION", "RECALL", "F1"] if c in mlp.class_report.columns]
ax = mlp.class_report.set_index("STATE")[class_cols].plot(kind="bar", figsize=(8.2, 4.7))
ax.set_ylim(0, 1)
ax.set_title("MLP Class-Level Performance")
ax.set_ylabel("Score")
plt.xticks(rotation=0)
save_show(ax.get_figure(), "06_mlp_class_performance.png")

mlp.validation.to_csv(EVIDENCE_DIR / "06_mlp_validation.csv", index=False)
mlp.metrics.to_csv(EVIDENCE_DIR / "06_mlp_test_metrics.csv", index=False)
mlp.class_report.to_csv(EVIDENCE_DIR / "06_mlp_class_performance.csv", index=False)
mlp.confusion.to_csv(EVIDENCE_DIR / "06_mlp_confusion_matrix.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “Because the three states are not equally represented, I do not rely on Accuracy alone.
# Macro F1, Balanced Accuracy, class Recall and the confusion matrix are used to judge whether
# Warning and Abnormal states are actually being detected.”

# %% [markdown]
# ---
# # SECTION 7 — Fork Integration and Common-Test Evaluation
#
# ARIMA and MLP are evaluated on the same eligible test observations before comparison.
# Four methods are compared:
#
# 1. SPC Persistence
# 2. ARIMA
# 3. MLP
# 4. Hybrid Fork decision
#
# The Hybrid decision uses the more severe state from the ARIMA and MLP branches.
# It is **not assumed to be the most accurate method**. The research question is answered by
# the actual common-test metrics.
#
# **Evidence to capture**
# - Common-test model comparison table
# - Accuracy / Balanced Accuracy / Macro F1 chart
# - Abnormal Recall comparison

# %%
section_heading("7. Fork Integration and Model Evaluation", "Chapter 4 §4.5 / Chapter 5 §5.4–5.5")

comparison, hybrid_predictions = evaluate_fork(mlp.predictions, arima.predictions)
display(comparison)

plot_cols = [c for c in ["ACCURACY", "BALANCED_ACCURACY", "MACRO_F1"] if c in comparison.columns]
ax = comparison.set_index("MODEL")[plot_cols].plot(kind="bar", figsize=(9, 4.8))
ax.set_title("Common-Test Comparison: SPC Persistence, ARIMA, MLP and Hybrid")
ax.set_xlabel("Method")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.xticks(rotation=0)
save_show(ax.get_figure(), "07_model_comparison.png")

if "RECALL_ABNORMAL" in comparison.columns:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(comparison["MODEL"], comparison["RECALL_ABNORMAL"])
    ax.set_ylim(0, 1)
    ax.set_title("Abnormal-Class Recall on the Common Test Set")
    ax.set_ylabel("Recall — Abnormal")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    save_show(fig, "07_abnormal_recall_comparison.png")

comparison.to_csv(EVIDENCE_DIR / "07_model_comparison.csv", index=False)
hybrid_predictions.to_csv(EVIDENCE_DIR / "07_hybrid_test_predictions.csv", index=False)

# %% [markdown]
# **Research interpretation**
#
# Do not write that the Hybrid is “high accuracy” unless the common-test results support it.
# A valid conclusion can be that the Fork combines complementary information even when it
# does not have the highest overall Accuracy. Abnormal Recall should also be reviewed because
# missing an abnormal future state is important from an engineering-risk perspective.

# %% [markdown]
# ---
# # SECTION 8 — Engineering DSS and Machine-Health Evidence
#
# This section converts the current SPC state and the two future predictive branches into the
# latest Machine–CTQ health view used by the dashboard.
#
# The DSS is an **engineering support tool**. It does not automatically stop equipment,
# change bonding parameters or make product disposition.
#
# **Evidence to capture**
# - Latest Machine–CTQ status table
# - Action / decision distribution
# - One selected machine health example showing Current SPC → ARIMA +5 → MLP +5 → Hybrid

# %%
section_heading("8. Engineering Decision-Support Evidence", "Chapter 4 §4.5 dashboard design / Chapter 5 §5.4 DSS result")

latest_status = build_latest_status(features, mlp.latest, arima.latest)
display(latest_status.head(30))

# Determine which available column provides the final DSS state/action.
action_col = None
for candidate in ["OVERALL_DECISION", "DSS_ACTION", "ENGINEERING_ACTION", "HYBRID_DECISION"]:
    if candidate in latest_status.columns:
        action_col = candidate
        break

if action_col:
    action_summary = latest_status[action_col].fillna("INSUFFICIENT").value_counts().rename_axis("ACTION").reset_index(name="STREAMS")
    display(action_summary)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(action_summary["ACTION"], action_summary["STREAMS"])
    ax.set_title("Latest Engineering DSS Action Distribution")
    ax.set_ylabel("Machine–CTQ streams")
    plt.xticks(rotation=25, ha="right")
    save_show(fig, "08_dss_action_distribution.png")
else:
    action_summary = pd.DataFrame()

# Select one row with the strongest available condition for a clear VIVA example.
priority_rank = {"NORMAL": 0, "WARNING": 1, "ABNORMAL": 2}
example = latest_status.copy()
for col in ["CURRENT_STATE", "ARIMA_PRED_STATE", "MLP_PRED_STATE", "HYBRID_DECISION", "OVERALL_DECISION"]:
    if col in example.columns:
        example[f"_{col}_RANK"] = example[col].astype(str).str.upper().map(priority_rank).fillna(-1)
rank_cols = [c for c in example.columns if c.startswith("_") and c.endswith("_RANK")]
if rank_cols:
    example["_MAX_RANK"] = example[rank_cols].max(axis=1)
    example = example.sort_values("_MAX_RANK", ascending=False)
selected_example = example.iloc[0] if len(example) else None

if selected_example is not None:
    evidence_cols = [c for c in [
        "MACH_NO", "CTQ", "CURRENT_STATE", "ARIMA_PRED_STATE", "MLP_PRED_STATE",
        "HYBRID_DECISION", "OVERALL_DECISION", "DECISION_PRIORITY", "ENGINEERING_GUIDANCE"
    ] if c in latest_status.columns]
    display(pd.DataFrame([selected_example[evidence_cols].to_dict()]))

latest_status.to_csv(EVIDENCE_DIR / "08_latest_machine_status.csv", index=False)
action_summary.to_csv(EVIDENCE_DIR / "08_dss_action_distribution.csv", index=False)

# %% [markdown]
# **How to explain this in VIVA**
#
# “The dashboard does not claim a machine failure. It combines current SPC evidence with the
# two +5 predictions and gives an engineering priority. The engineer still confirms the actual
# equipment condition and physical root cause.”

# %% [markdown]
# ---
# # SECTION 9 — Generate the Final Thesis Output Set
#
# After the individual analysis sections have been reviewed, this cell produces the complete
# standardized output set under `outputs/thesis/`. These are the preferred files to insert in
# the final report because the names remain stable between runs.
#
# Use `outputs/evidence_by_section/` when you want evidence directly from this VS Code file.
# Use `outputs/thesis/` for the final Chapter 4 / Chapter 5 figures and tables.

# %%
section_heading("9. Final Report Output Generation", "Final Chapter 4 and Chapter 5 evidence")

generate_thesis_outputs()

thesis_dir = PROJECT_ROOT / "outputs" / "thesis"
output_files = sorted([p.name for p in thesis_dir.glob("*") if p.is_file()])
output_index = pd.DataFrame({"FILE": output_files})
display(output_index)

print("\nFinal thesis output folder:", thesis_dir)
print("Section-by-section evidence folder:", EVIDENCE_DIR)

# %% [markdown]
# ## Recommended evidence map
#
# ### Chapter 4 — Analysis and Design
# Use evidence that explains **how the system is designed**:
#
# - Section 1: data-cleaning flow / event-level structure
# - Section 2: population-selection logic
# - Section 3: X̄–R + Nelson Rule implementation example
# - Section 4: feature groups + train/validation/test design
# - Section 5–6: ARIMA and MLP branch design evidence
# - Section 7: Fork comparison structure
# - Dashboard screenshot: two-page DSS design
#
# ### Chapter 5 — Results and Discussion
# Use the **final numerical outputs**:
#
# - selected population counts
# - Nelson trigger and episode counts
# - Normal / Warning / Abnormal distribution
# - capability / break-mode result
# - ARIMA vs naïve RMSE
# - ARIMA next-five-event SPC trend
# - MLP validation result, test metrics and confusion matrix
# - common-test model comparison
# - latest DSS engineering example
#
# **Best practice:** use clean PNG/CSV output files in the thesis and use screenshots of this
# VS Code interactive file mainly as implementation/coding evidence for the appendix or VIVA.
