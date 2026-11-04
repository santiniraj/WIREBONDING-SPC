from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from config import DATA_FILE, PROJECT_ROOT
from src.data_cleaning import (
    build_eda_tables,
    ensure_output_dirs,
    load_and_clean_data,
    save_study_config,
    select_study_population,
)
from src.features import build_features, chronological_split, derive_time_split_meta
from src.anonymization import (
    anonymize_study,
    public_study_config,
    sanitize_for_public_export,
    sanitize_population_audit,
)
from src.models import build_latest_status, evaluate_fork, train_arima, train_mlp
from src.spc import apply_spc, nelson_rule_reference
from src.thesis_outputs import main as generate_thesis_outputs
from src.report_summary import main as generate_run_summary


def build_break_mode_summary(spc_private: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Create a thesis/dashboard-safe generic A-E break-mode summary.

    The physical meaning and PASS/FAIL interpretation are only added when the user has
    completed data/FAILURE_MODE_MAPPING.csv from an approved work instruction.
    """
    mode_cols = [c for c in ["MODE_A_COUNT", "MODE_B_COUNT", "MODE_C_COUNT", "MODE_D_COUNT", "MODE_E_COUNT"] if c in spc_private.columns]
    if not mode_cols:
        return pd.DataFrame(columns=["MACH_NO", "CTQ", "MODE", "COUNT", "PERCENT", "PHYSICAL_MEANING", "PASS_FAIL", "NOTES"])

    parts = []
    for col in mode_cols:
        mode = col.replace("MODE_", "").replace("_COUNT", "")
        d = (
            spc_private.groupby(["MACH_NO", "CTQ"], as_index=False)[col]
            .sum()
            .rename(columns={col: "COUNT"})
        )
        d["MODE"] = mode
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    totals = out.groupby(["MACH_NO", "CTQ"])["COUNT"].transform("sum")
    out["PERCENT"] = 100.0 * out["COUNT"] / totals.replace(0, pd.NA)

    mapping_path = project_root / "data" / "FAILURE_MODE_MAPPING.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        keep = [c for c in ["MODE", "PHYSICAL_MEANING", "PASS_FAIL", "NOTES"] if c in mapping.columns]
        if "MODE" in keep:
            out = out.merge(mapping[keep].drop_duplicates("MODE"), on="MODE", how="left")
    for c in ["PHYSICAL_MEANING", "PASS_FAIL", "NOTES"]:
        if c not in out.columns:
            out[c] = ""
    return out.sort_values(["MACH_NO", "CTQ", "MODE"]).reset_index(drop=True)


def main() -> None:
    paths = ensure_output_dirs(PROJECT_ROOT)

    print("=" * 96)
    print("WIREBOND SPC-AI RESEARCH PIPELINE")
    print("RQ1: Cleaning + SPC/Nelson | RQ2: ARIMA + MLP Fork | RQ3: Hybrid Engineering DSS")
    print("=" * 96)

    print("\n[1/8] Data preparation and cleaning")
    cleaned, cleaning_audit, normalization_audit = load_and_clean_data(DATA_FILE)
    cleaned.to_csv(paths["data"] / "cleaned_all_events_anonymous.csv", index=False)
    normalization_audit.to_csv(paths["data"] / "normalization_audit.csv", index=False)
    cleaning_audit.to_csv(paths["data"] / "data_cleaning_audit.csv", index=False)
    print(cleaning_audit.to_string(index=False))

    print("\n[2/8] EDA and study population selection")
    eda = build_eda_tables(cleaned)
    for name, table in eda.items():
        table.to_csv(paths["eda"] / f"eda_{name}.csv", index=False)

    study_raw, study_config_raw, population_audit, machine_population_raw = select_study_population(cleaned)
    study, machine_map = anonymize_study(study_raw)
    study_config = public_study_config(study_config_raw, machine_map, study)

    sanitize_for_public_export(study).to_csv(paths["data"] / "selected_population.csv", index=False)
    sanitize_population_audit(population_audit).to_csv(paths["data"] / "population_selection.csv", index=False)
    machine_population = study.groupby(["MACH_NO", "CTQ"]).size().unstack(fill_value=0).reset_index()
    for ctq in ["WP", "BS", "SP"]:
        if ctq not in machine_population:
            machine_population[ctq] = 0
    machine_population["TOTAL_EVENTS"] = machine_population[["WP", "BS", "SP"]].sum(axis=1)
    machine_population.to_csv(paths["data"] / "machine_population.csv", index=False)
    save_study_config(study_config, paths["data"] / "study_config.json")
    print(f"Selected anonymized study population | {study_config['machine_count']} machines")
    print("Events by CTQ:", study_config["events_by_ctq"])

    split_meta = derive_time_split_meta(study)
    (paths["evaluation"] / "split_meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    print("\n[3/8] SPC, capability, Nelson Rules and break-mode summary")
    spc, limits, nelson_summary, machine_spc_summary, capability = apply_spc(study, split_meta)
    sanitize_for_public_export(spc).to_csv(paths["spc"] / "spc_results.csv", index=False)
    sanitize_for_public_export(limits).to_csv(paths["spc"] / "spc_limits.csv", index=False)
    sanitize_for_public_export(capability).to_csv(paths["spc"] / "capability_results.csv", index=False)
    nelson_summary.to_csv(paths["spc"] / "nelson_summary.csv", index=False)
    nelson_rule_reference().to_csv(paths["spc"] / "nelson_rule_reference.csv", index=False)
    machine_spc_summary.to_csv(paths["spc"] / "machine_spc_summary.csv", index=False)
    build_break_mode_summary(spc, PROJECT_ROOT).to_csv(paths["spc"] / "break_mode_summary.csv", index=False)
    print(f"SPC-available events: {int(spc['SPC_AVAILABLE'].sum())} / {len(spc)}")

    print("\n[4/8] Feature engineering and training / validation / test dataset")
    features = build_features(spc)
    labelled, split_meta, split_summary, split_class_summary = chronological_split(features, split_meta)
    sanitize_for_public_export(features).to_csv(paths["spc"] / "features.csv", index=False)
    sanitize_for_public_export(labelled).to_csv(paths["spc"] / "labelled_features.csv", index=False)
    split_summary.to_csv(paths["evaluation"] / "split_summary.csv", index=False)
    split_class_summary.to_csv(paths["evaluation"] / "split_class_summary.csv", index=False)
    (paths["evaluation"] / "split_meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    print(split_summary.to_string(index=False))

    print("\n[5/8] Fork modelling: MLP and ARIMA")
    mlp = train_mlp(features, labelled, paths["models"])
    mlp.validation.to_csv(paths["evaluation"] / "mlp_validation.csv", index=False)
    mlp.metrics.to_csv(paths["evaluation"] / "mlp_metrics.csv", index=False)
    sanitize_for_public_export(mlp.predictions).to_csv(paths["evaluation"] / "mlp_test_predictions.csv", index=False)
    mlp.confusion.to_csv(paths["evaluation"] / "mlp_confusion_matrix.csv", index=False)
    mlp.class_report.to_csv(paths["evaluation"] / "mlp_class_report.csv", index=False)
    mlp.latest.to_csv(paths["models"] / "mlp_latest_predictions.csv", index=False)
    print("MLP test metrics:")
    print(mlp.metrics.to_string(index=False))

    arima = train_arima(features, split_meta)
    arima.selection.to_csv(paths["evaluation"] / "arima_validation_orders.csv", index=False)
    arima.metrics.to_csv(paths["evaluation"] / "arima_metrics.csv", index=False)
    arima.predictions.to_csv(paths["evaluation"] / "arima_test_predictions.csv", index=False)
    arima.forecast.to_csv(paths["models"] / "arima_next_5_forecast.csv", index=False)
    arima.latest.to_csv(paths["models"] / "arima_latest_predictions.csv", index=False)

    print("\n[6/8] Model evaluation and Fork integration")
    comparison, hybrid_predictions = evaluate_fork(mlp.predictions, arima.predictions)
    comparison.to_csv(paths["evaluation"] / "model_comparison.csv", index=False)
    sanitize_for_public_export(hybrid_predictions).to_csv(paths["evaluation"] / "hybrid_test_predictions.csv", index=False)
    if not comparison.empty:
        print(comparison.to_string(index=False))

    print("\n[7/8] Deployment / engineering DSS output")
    latest_status = build_latest_status(features, mlp.latest, arima.latest)
    latest_status.to_csv(paths["deployment"] / "latest_machine_status.csv", index=False)
    print(f"Latest machine/CTQ status rows: {len(latest_status)}")

    print("\n[8/8] Thesis-safe tables and figures")
    generate_thesis_outputs()
    summary_path = generate_run_summary()
    print(f"Run summary: {summary_path}")

    print("\n" + "=" * 96)
    print("PIPELINE COMPLETE")
    print("Start dashboard with: python -m streamlit run dashboard/app.py")
    print("=" * 96)


if __name__ == "__main__":
    main()
