from __future__ import annotations

"""Generate thesis-safe tables and figures from pipeline outputs.

The module does not re-run modelling.  It reads the frozen CSV outputs created by
``run_all.py`` and creates clean evidence for Chapter 4 (analysis/design evidence)
and Chapter 5 (results/discussion evidence).

All plots use anonymized machine IDs and standardized/normalized SPC values.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
THESIS = OUT / "thesis"
THESIS.mkdir(parents=True, exist_ok=True)

D4_N5 = 2.114
STATE_ORDER = ["NORMAL", "WARNING", "ABNORMAL"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(THESIS / f"{name}.csv", index=False)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(THESIS / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.upper().isin(["TRUE", "1", "YES", "Y"])


def _label_bars(ax: plt.Axes, decimals: int = 0) -> None:
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(
                f"{h:.{decimals}f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points",
            )


def _data_preparation_outputs() -> None:
    cleaning = read_csv(OUT / "data" / "data_cleaning_audit.csv")
    if not cleaning.empty:
        cols = [c for c in [
            "CTQ", "RAW_ROWS", "ITEM_VAL_ROWS", "FAILURE_CODE_ROWS", "PAIRED_EVENTS",
            "INVALID_EVENTS_REMOVED", "CLEAN_EVENTS"
        ] if c in cleaning.columns]
        t = cleaning[cols].copy()
        save_table(t, "table_data_cleaning_audit")

        if {"CTQ", "RAW_ROWS", "CLEAN_EVENTS"}.issubset(t.columns):
            x = np.arange(len(t))
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            ax.bar(x - 0.2, t["RAW_ROWS"], 0.4, label="Raw rows")
            ax.bar(x + 0.2, t["CLEAN_EVENTS"], 0.4, label="Clean event-level rows")
            ax.set_xticks(x, t["CTQ"])
            ax.set_ylabel("Rows / events")
            ax.set_title("Data Cleaning: Raw Records to Event-Level Records")
            ax.legend()
            save_figure(fig, "figure_data_cleaning_summary")

    cfg_path = OUT / "data" / "study_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        events = cfg.get("events_by_ctq", {})
        pop = pd.DataFrame([
            ["Package", cfg.get("selected_package", "Package A")],
            ["Wire Type", cfg.get("selected_wire_type", "Wire Type A")],
            ["Wire Diameter", cfg.get("selected_wire_diameter", "Diameter Group A")],
            ["Machines", cfg.get("machine_count", "")],
            ["Wire Pull Events", events.get("WP", "")],
            ["Ball Shear Events", events.get("BS", "")],
            ["Stitch Pull Events", events.get("SP", "")],
            ["Total Events", cfg.get("total_events", "")],
            ["Study Period", cfg.get("study_period", "Anonymized historical production period")],
        ], columns=["Item", "Selected Population"])
        save_table(pop, "table_selected_study_population")

        vals = [events.get("WP", 0), events.get("BS", 0), events.get("SP", 0)]
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        ax.bar(["WP", "BS", "SP"], vals)
        ax.set_title("Selected Study Population by CTQ")
        ax.set_xlabel("Critical to Quality (CTQ)")
        ax.set_ylabel("Events")
        _label_bars(ax)
        save_figure(fig, "figure_selected_population_by_ctq")

    machine = read_csv(OUT / "data" / "machine_population.csv")
    if not machine.empty:
        machine = machine.sort_values("TOTAL_EVENTS", ascending=False)
        save_table(machine, "table_machine_event_distribution")
        plot_cols = [c for c in ["WP", "BS", "SP"] if c in machine.columns]
        if plot_cols:
            ax = machine.set_index("MACH_NO")[plot_cols].plot(kind="bar", stacked=True, figsize=(11, 4.8))
            ax.set_title("Event Distribution Across Anonymized Machines")
            ax.set_xlabel("Machine")
            ax.set_ylabel("Events")
            plt.xticks(rotation=45, ha="right")
            save_figure(ax.get_figure(), "figure_machine_event_distribution")


def _spc_outputs() -> None:
    spc = read_csv(OUT / "spc" / "spc_results.csv")
    limits = read_csv(OUT / "spc" / "spc_limits.csv")
    nelson = read_csv(OUT / "spc" / "nelson_summary.csv")
    cap = read_csv(OUT / "spc" / "capability_results.csv")
    breakm = read_csv(OUT / "spc" / "break_mode_summary.csv")
    machine_spc = read_csv(OUT / "spc" / "machine_spc_summary.csv")
    labelled = read_csv(OUT / "spc" / "labelled_features.csv")

    if not machine_spc.empty and "BASELINE_STATUS" in machine_spc.columns:
        base = machine_spc["BASELINE_STATUS"].value_counts().rename_axis("REFERENCE_STATUS").reset_index(name="STREAMS")
        save_table(base, "table_spc_reference_status")

    if not nelson.empty:
        agg = nelson.groupby("RULE", as_index=False).agg(
            TRIGGER_COUNT=("TRIGGER_COUNT", "sum"), EPISODE_COUNT=("EPISODE_COUNT", "sum")
        )
        agg["ORDER"] = pd.to_numeric(agg["RULE"].str.extract(r"(\d+)")[0], errors="coerce")
        agg = agg.sort_values("ORDER").drop(columns="ORDER")
        save_table(agg, "table_nelson_rule_summary")
        x = np.arange(len(agg))
        fig, ax = plt.subplots(figsize=(8.3, 4.6))
        ax.bar(x - 0.19, agg["TRIGGER_COUNT"], 0.38, label="Triggers")
        ax.bar(x + 0.19, agg["EPISODE_COUNT"], 0.38, label="Episodes")
        ax.set_xticks(x, agg["RULE"])
        ax.set_title("Nelson Rule Triggers and Episodes")
        ax.set_xlabel("Rule")
        ax.set_ylabel("Count")
        ax.legend()
        save_figure(fig, "figure_nelson_rules")

    state_source = labelled if not labelled.empty and "PROCESS_STATE" in labelled.columns else spc
    if not state_source.empty and "PROCESS_STATE" in state_source.columns:
        states = state_source["PROCESS_STATE"].value_counts().reindex(STATE_ORDER, fill_value=0).rename_axis("STATE").reset_index(name="EVENTS")
        states["PERCENT"] = 100 * states["EVENTS"] / max(1, states["EVENTS"].sum())
        save_table(states, "table_process_state_distribution")
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.bar(states["STATE"], states["EVENTS"])
        ax.set_title("SPC Process-State Distribution")
        ax.set_ylabel("Events")
        _label_bars(ax)
        save_figure(fig, "figure_process_state_distribution")

    if not cap.empty:
        cols = [c for c in ["MACH_NO", "CTQ", "SPEC_KEY", "CAPABILITY_TYPE", "CAPABILITY_VALUE", "REFERENCE_N"] if c in cap.columns]
        save_table(cap[cols], "table_capability_results")
        c = cap.dropna(subset=["CAPABILITY_VALUE"]).copy()
        if not c.empty:
            c["STREAM"] = c["MACH_NO"].astype(str) + "-" + c["CTQ"].astype(str)
            best = c.groupby("STREAM", as_index=False)["CAPABILITY_VALUE"].min().sort_values("CAPABILITY_VALUE")
            fig, ax = plt.subplots(figsize=(11, 4.7))
            ax.bar(best["STREAM"], best["CAPABILITY_VALUE"])
            ax.axhline(1.33, linestyle="--", linewidth=1.2, label="Reference 1.33")
            ax.axhline(1.00, linestyle=":", linewidth=1.2, label="Reference 1.00")
            ax.set_title("Minimum Valid Capability Index by Machine–CTQ Stream")
            ax.set_xlabel("Anonymized Machine–CTQ")
            ax.set_ylabel("Capability index")
            ax.legend()
            plt.xticks(rotation=55, ha="right", fontsize=7)
            save_figure(fig, "figure_capability_by_stream")

    if not breakm.empty:
        save_table(breakm, "table_break_mode_summary")
        b = breakm.groupby(["CTQ", "MODE"], as_index=False)["COUNT"].sum()
        if not b.empty:
            pivot = b.pivot(index="MODE", columns="CTQ", values="COUNT").fillna(0)
            ax = pivot.plot(kind="bar", figsize=(8.2, 4.7))
            ax.set_title("Break-Mode Distribution by CTQ")
            ax.set_xlabel("Generic Break Mode")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            save_figure(ax.get_figure(), "figure_break_mode_distribution")

        if "PASS_FAIL" in breakm.columns:
            pf = breakm.copy()
            pf["PASS_FAIL"] = pf["PASS_FAIL"].fillna("").astype(str).str.upper().str.strip()
            mapped = pf[pf["PASS_FAIL"].isin(["PASS", "FAIL"])]
            if not mapped.empty:
                summary = mapped.groupby("PASS_FAIL", as_index=False)["COUNT"].sum()
                save_table(summary, "table_break_mode_pass_fail_summary")
                fig, ax = plt.subplots(figsize=(5.8, 4.2))
                ax.bar(summary["PASS_FAIL"], summary["COUNT"])
                ax.set_title("Mapped Break-Mode PASS / FAIL Summary")
                ax.set_ylabel("Count")
                _label_bars(ax)
                save_figure(fig, "figure_break_mode_pass_fail")

    # Representative stream for Xbar/R analysis: use stream with highest number of valid SPC events.
    required = {"MACH_NO", "CTQ", "DATE_TIME", "STANDARDIZED_DEVIATION", "RANGE_RATIO"}
    if not spc.empty and required.issubset(spc.columns):
        d0 = spc.copy()
        if "SPC_AVAILABLE" in d0.columns:
            d0 = d0[_as_bool(d0["SPC_AVAILABLE"])]
        # Prefer a stream with many Nelson signals so the representative figure visibly
        # demonstrates how Nelson Rules are overlaid on the Xbar chart.
        if "NELSON_ANY" in d0.columns:
            d0["_NELSON_BOOL"] = _as_bool(d0["NELSON_ANY"]).astype(int)
            ranking = d0.groupby(["MACH_NO", "CTQ"]).agg(
                NELSON_SIGNALS=("_NELSON_BOOL", "sum"), EVENTS=("_NELSON_BOOL", "size")
            ).sort_values(["NELSON_SIGNALS", "EVENTS"], ascending=False)
        else:
            ranking = d0.groupby(["MACH_NO", "CTQ"]).size().to_frame("EVENTS").sort_values("EVENTS", ascending=False)
        if len(ranking):
            mach, ctq = ranking.index[0]
            d = d0[(d0["MACH_NO"] == mach) & (d0["CTQ"] == ctq)].copy()
            d["DATE_TIME"] = pd.to_datetime(d["DATE_TIME"], errors="coerce")
            d = d.sort_values("DATE_TIME").tail(60).reset_index(drop=True)
            x = np.arange(1, len(d) + 1)

            fig, ax = plt.subplots(figsize=(11, 5.0))
            ax.plot(x, d["STANDARDIZED_DEVIATION"], marker="o", markersize=3, linewidth=1.0, label="Standardized X̄")
            for y, lab, ls in [(3, "+3σ UCL", "--"), (2, "+2σ", ":"), (1, "+1σ", ":"), (0, "CL", "-"), (-1, "-1σ", ":"), (-2, "-2σ", ":"), (-3, "-3σ LCL", "--")]:
                ax.axhline(y, linestyle=ls, linewidth=0.9, label=lab if y in [3, 0, -3] else None)
            if "NELSON_ANY" in d.columns:
                sig = _as_bool(d["NELSON_ANY"])
                if sig.any():
                    ax.scatter(x[sig.to_numpy()], d.loc[sig, "STANDARDIZED_DEVIATION"], marker="*", s=85, label="Nelson signal")
                    if "NELSON_RULES_TRIGGERED" in d.columns:
                        for xi, yi, txt in zip(x[sig.to_numpy()], d.loc[sig, "STANDARDIZED_DEVIATION"], d.loc[sig, "NELSON_RULES_TRIGGERED"].fillna("")):
                            if str(txt).strip():
                                ax.annotate(str(txt), (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
            ax.set_title(f"Representative Standardized X̄ Chart with Nelson Rules ({mach} · {ctq})")
            ax.set_xlabel("Recent production event sequence")
            ax.set_ylabel("Standardized X̄ deviation (σ)")
            ax.legend(ncol=3, fontsize=8)
            save_figure(fig, "figure_representative_xbar_nelson")

            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(x, d["RANGE_RATIO"], marker="o", markersize=3, linewidth=1.0, label="R / R̄")
            ax.axhline(D4_N5, linestyle="--", label=f"UCL(R) = {D4_N5:.3f} × R̄")
            ax.axhline(1.0, linewidth=1.0, label="R̄")
            ax.axhline(0.0, linestyle=":", label="LCL = 0")
            ax.set_title(f"Representative Normalized R Chart ({mach} · {ctq})")
            ax.set_xlabel("Recent production event sequence")
            ax.set_ylabel("Range ratio (R / R̄)")
            ax.legend()
            save_figure(fig, "figure_representative_r_chart")


def _feature_and_split_outputs() -> None:
    split = read_csv(OUT / "evaluation" / "split_summary.csv")
    classes = read_csv(OUT / "evaluation" / "split_class_summary.csv")
    if not split.empty:
        save_table(split, "table_training_validation_test_split")
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        y = "PERCENT_OF_LABELLED" if "PERCENT_OF_LABELLED" in split.columns else "ROWS"
        ax.bar(split["SPLIT"], split[y])
        ax.set_title("Training, Validation, Test and Boundary-Removed Observations")
        ax.set_ylabel("Percent of labelled observations" if y == "PERCENT_OF_LABELLED" else "Rows")
        _label_bars(ax, 1 if y == "PERCENT_OF_LABELLED" else 0)
        save_figure(fig, "figure_dataset_split")
    if not classes.empty:
        save_table(classes, "table_split_class_distribution")
        p = classes.pivot(index="SPLIT", columns="STATE", values="ROWS").fillna(0)
        p = p.reindex(columns=STATE_ORDER, fill_value=0)
        ax = p.plot(kind="bar", figsize=(7.8, 4.6))
        ax.set_title("Future-State Class Distribution by Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Rows")
        plt.xticks(rotation=0)
        save_figure(ax.get_figure(), "figure_split_class_distribution")

    feature_groups = pd.DataFrame([
        ["Current statistics", "AVG, SIGMA, RANGE, CV"],
        ["SPC position", "STANDARDIZED_DEVIATION, RANGE_RATIO"],
        ["Capability", "CAPABILITY_VALUE"],
        ["Nelson Rules", "NELSON_RULE_COUNT, N1–N8"],
        ["Break mode", "MODE_A_COUNT–MODE_E_COUNT"],
        ["Lag / rolling history", "Lagged Z, rolling mean/std, moving average"],
        ["Previous state history", "Previous Warning/Abnormal/Nelson rates"],
        ["Categorical", "Anonymized Machine ID, CTQ"],
    ], columns=["FEATURE_GROUP", "EXAMPLES"])
    save_table(feature_groups, "table_feature_groups")


def _arima_outputs() -> None:
    selection = read_csv(OUT / "evaluation" / "arima_validation_orders.csv")
    metrics = read_csv(OUT / "evaluation" / "arima_metrics.csv")
    fc = read_csv(OUT / "models" / "arima_next_5_forecast.csv")
    spc = read_csv(OUT / "spc" / "spc_results.csv")

    if not selection.empty:
        save_table(selection, "table_arima_validation_orders")
    if metrics.empty:
        return

    save_table(metrics, "table_arima_test_performance")
    summary = pd.DataFrame({
        "ITEM": ["Evaluated streams", "ARIMA better than naïve", "Adequate evidence streams"],
        "RESULT": [
            len(metrics),
            int((metrics["H5_RMSE"] < metrics["NAIVE_H5_RMSE"]).sum()) if {"H5_RMSE", "NAIVE_H5_RMSE"}.issubset(metrics.columns) else np.nan,
            int(metrics["EVIDENCE_LEVEL"].eq("ADEQUATE").sum()) if "EVIDENCE_LEVEL" in metrics.columns else np.nan,
        ],
    })
    save_table(summary, "table_arima_summary")

    if {"H5_RMSE", "NAIVE_H5_RMSE", "MACH_NO", "CTQ"}.issubset(metrics.columns):
        d = metrics.copy()
        d["STREAM"] = d["MACH_NO"].astype(str) + "-" + d["CTQ"].astype(str)
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        ax.bar(x - 0.2, d["H5_RMSE"], 0.4, label="ARIMA H5 RMSE")
        ax.bar(x + 0.2, d["NAIVE_H5_RMSE"], 0.4, label="Naïve H5 RMSE")
        ax.set_xticks(x, d["STREAM"], rotation=55, ha="right", fontsize=7)
        ax.set_title("ARIMA Fifth-Event RMSE Compared with Naïve Forecast")
        ax.set_ylabel("RMSE")
        ax.legend()
        save_figure(fig, "figure_arima_rmse_vs_naive")

    # Combined historical SPC + five-event ARIMA forecast for one adequate stream.
    if not fc.empty and not spc.empty:
        candidate = metrics[metrics["EVIDENCE_LEVEL"].eq("ADEQUATE")] if "EVIDENCE_LEVEL" in metrics.columns else metrics
        if candidate.empty:
            candidate = metrics
        # Prefer a stream where ARIMA beats naive, otherwise first eligible stream.
        if {"H5_RMSE", "NAIVE_H5_RMSE"}.issubset(candidate.columns):
            improved = candidate[candidate["H5_RMSE"] < candidate["NAIVE_H5_RMSE"]]
            row = improved.iloc[0] if not improved.empty else candidate.iloc[0]
        else:
            row = candidate.iloc[0]
        mach, ctq = row["MACH_NO"], row["CTQ"]
        hist = spc[(spc["MACH_NO"] == mach) & (spc["CTQ"] == ctq)].copy()
        if "SPC_AVAILABLE" in hist.columns:
            hist = hist[_as_bool(hist["SPC_AVAILABLE"])]
        hist["DATE_TIME"] = pd.to_datetime(hist["DATE_TIME"], errors="coerce")
        hist = hist.sort_values("DATE_TIME").tail(40).reset_index(drop=True)
        future = fc[(fc["MACH_NO"] == mach) & (fc["CTQ"] == ctq)].sort_values("HORIZON_EVENT")
        if not hist.empty and not future.empty:
            hx = np.arange(1, len(hist) + 1)
            fx = np.arange(len(hist) + 1, len(hist) + 1 + len(future))
            fig, ax = plt.subplots(figsize=(11.5, 5.1))
            ax.plot(hx, hist["STANDARDIZED_DEVIATION"], marker="o", markersize=3, linewidth=1.1, label="Actual standardized X̄")
            ax.plot(fx, future["FORECAST_Z"], marker="o", linestyle="--", linewidth=1.3, label="ARIMA forecast")
            if {"LOWER_95", "UPPER_95"}.issubset(future.columns):
                ax.fill_between(fx, future["LOWER_95"], future["UPPER_95"], alpha=0.18, label="95% forecast interval")
            if "NELSON_ANY" in hist.columns:
                sig = _as_bool(hist["NELSON_ANY"])
                if sig.any():
                    ax.scatter(hx[sig.to_numpy()], hist.loc[sig, "STANDARDIZED_DEVIATION"], marker="*", s=70, label="Nelson signal")
            for y, ls in [(3, "--"), (2, ":"), (1, ":"), (0, "-"), (-1, ":"), (-2, ":"), (-3, "--")]:
                ax.axhline(y, linestyle=ls, linewidth=0.8)
            ax.axvline(len(hist) + 0.5, linestyle="--", linewidth=1.0)
            ax.text(len(hist) + 0.7, ax.get_ylim()[1] * 0.9, "Forecast →", fontsize=9)
            ax.set_title(f"Recent SPC Trend with ARIMA Next-Five-Event Forecast ({mach} · {ctq})")
            ax.set_xlabel("Production event sequence")
            ax.set_ylabel("Standardized X̄ deviation (σ)")
            ax.legend(ncol=3, fontsize=8)
            save_figure(fig, "figure_arima_next5_on_spc")


def _mlp_outputs() -> None:
    val = read_csv(OUT / "evaluation" / "mlp_validation.csv")
    metrics = read_csv(OUT / "evaluation" / "mlp_metrics.csv")
    report = read_csv(OUT / "evaluation" / "mlp_class_report.csv")
    cm = read_csv(OUT / "evaluation" / "mlp_confusion_matrix.csv")

    if not val.empty:
        save_table(val, "table_mlp_validation")
        labels = [f"{r.HIDDEN_LAYERS}\nα={r.ALPHA}" for r in val.itertuples()]
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        ax.bar(labels, val["VALIDATION_MACRO_F1"])
        ax.set_title("MLP Candidate Selection by Validation Macro F1")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, max(0.5, float(val["VALIDATION_MACRO_F1"].max()) * 1.25))
        _label_bars(ax, 3)
        save_figure(fig, "figure_mlp_validation")

    if not metrics.empty:
        save_table(metrics, "table_mlp_test_metrics")
        metric_cols = [c for c in ["ACCURACY", "BALANCED_ACCURACY", "MACRO_PRECISION", "MACRO_RECALL", "MACRO_F1", "ROC_AUC_OVR_MACRO"] if c in metrics.columns]
        if metric_cols:
            values = metrics.iloc[0][metric_cols].astype(float)
            fig, ax = plt.subplots(figsize=(8.5, 4.4))
            ax.bar(metric_cols, values)
            ax.set_ylim(0, 1)
            ax.set_title("MLP Test Performance")
            ax.set_ylabel("Score")
            plt.xticks(rotation=30, ha="right")
            _label_bars(ax, 3)
            save_figure(fig, "figure_mlp_test_metrics")

    if not report.empty:
        save_table(report, "table_mlp_class_performance")
        cols = [c for c in ["PRECISION", "RECALL", "F1"] if c in report.columns]
        if cols:
            ax = report.set_index("STATE")[cols].plot(kind="bar", figsize=(8.2, 4.6))
            ax.set_title("MLP Class-Level Performance")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=0)
            save_figure(ax.get_figure(), "figure_mlp_class_performance")

    if not cm.empty:
        save_table(cm, "table_mlp_confusion_matrix_long")
        p = cm.pivot(index="ACTUAL_STATE", columns="PREDICTED_STATE", values="COUNT").reindex(index=STATE_ORDER, columns=STATE_ORDER).fillna(0)
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        im = ax.imshow(p.values)
        ax.set_xticks(range(3), p.columns)
        ax.set_yticks(range(3), p.index)
        ax.set_xlabel("Predicted state")
        ax.set_ylabel("Actual state")
        ax.set_title("MLP Test Confusion Matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(p.iloc[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        save_figure(fig, "figure_mlp_confusion_matrix")


def _hybrid_outputs() -> None:
    comp = read_csv(OUT / "evaluation" / "model_comparison.csv")
    latest = read_csv(OUT / "deployment" / "latest_machine_status.csv")
    if not comp.empty:
        cols = [c for c in ["MODEL", "TEST_N", "ACCURACY", "BALANCED_ACCURACY", "MACRO_PRECISION", "MACRO_RECALL", "MACRO_F1", "RECALL_ABNORMAL"] if c in comp.columns]
        save_table(comp[cols], "table_model_comparison")
        plot_cols = [c for c in ["ACCURACY", "BALANCED_ACCURACY", "MACRO_F1"] if c in comp.columns]
        if plot_cols:
            ax = comp.set_index("MODEL")[plot_cols].plot(kind="bar", figsize=(8.5, 4.7))
            ax.set_title("Common-Test-Set Comparison")
            ax.set_xlabel("Method")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=0)
            save_figure(ax.get_figure(), "figure_model_comparison")

    if not latest.empty:
        save_table(latest, "table_latest_machine_health")
        if "OVERALL_DECISION" in latest.columns:
            actions = latest["OVERALL_DECISION"].fillna("INSUFFICIENT").value_counts().rename_axis("STATE").reset_index(name="STREAMS")
            save_table(actions, "table_latest_health_summary")


def _output_index() -> None:
    files = []
    for p in sorted(THESIS.glob("*")):
        if p.is_file():
            files.append({"FILE": p.name, "TYPE": p.suffix.lstrip(".").upper()})
    if files:
        pd.DataFrame(files).to_csv(THESIS / "OUTPUT_INDEX.csv", index=False)


def main() -> None:
    _data_preparation_outputs()
    _spc_outputs()
    _feature_and_split_outputs()
    _arima_outputs()
    _mlp_outputs()
    _hybrid_outputs()
    _output_index()
    print(f"Thesis-safe tables and figures written to: {THESIS}")


if __name__ == "__main__":
    main()
