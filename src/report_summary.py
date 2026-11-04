from __future__ import annotations

"""Create a compact Markdown run summary from frozen pipeline outputs."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(v: object, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def main() -> Path:
    lines: list[str] = [
        "# WirebondingSPC — Latest Pipeline Run Summary",
        "",
        "This file is generated automatically after `python run_all.py`.",
        "",
    ]

    cfg = OUT / "data" / "study_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        lines += [
            "## 1. Selected study population",
            "",
            f"- Machines: **{c.get('machine_count', '')}**",
            f"- WP events: **{c.get('events_by_ctq', {}).get('WP', '')}**",
            f"- BS events: **{c.get('events_by_ctq', {}).get('BS', '')}**",
            f"- SP events: **{c.get('events_by_ctq', {}).get('SP', '')}**",
            f"- Total events: **{c.get('total_events', '')}**",
            "",
        ]

    split = _csv(OUT / "evaluation" / "split_summary.csv")
    if not split.empty:
        lines += ["## 2. Model dataset", "", split.to_markdown(index=False), ""]

    nelson = _csv(OUT / "spc" / "nelson_summary.csv")
    if not nelson.empty:
        agg = nelson.groupby("RULE", as_index=False)[["TRIGGER_COUNT", "EPISODE_COUNT"]].sum()
        lines += ["## 3. Nelson Rule results", "", agg.to_markdown(index=False), ""]

    arima = _csv(OUT / "evaluation" / "arima_metrics.csv")
    if not arima.empty:
        better = int((arima["H5_RMSE"] < arima["NAIVE_H5_RMSE"]).sum()) if {"H5_RMSE", "NAIVE_H5_RMSE"}.issubset(arima.columns) else ""
        adequate = int(arima["EVIDENCE_LEVEL"].eq("ADEQUATE").sum()) if "EVIDENCE_LEVEL" in arima.columns else ""
        lines += [
            "## 4. ARIMA",
            "",
            f"- Evaluated streams: **{len(arima)}**",
            f"- ARIMA RMSE lower than naïve: **{better}**",
            f"- Adequate-evidence streams: **{adequate}**",
            "",
        ]

    mlp = _csv(OUT / "evaluation" / "mlp_metrics.csv")
    if not mlp.empty:
        r = mlp.iloc[0]
        lines += [
            "## 5. MLP",
            "",
            f"- Selected hidden layers: **{r.get('SELECTED_HIDDEN_LAYERS', '')}**",
            f"- Alpha: **{r.get('SELECTED_ALPHA', '')}**",
            f"- Accuracy: **{_fmt(r.get('ACCURACY', ''))}**",
            f"- Balanced accuracy: **{_fmt(r.get('BALANCED_ACCURACY', ''))}**",
            f"- Macro F1: **{_fmt(r.get('MACRO_F1', ''))}**",
            f"- Abnormal recall: **{_fmt(r.get('RECALL_ABNORMAL', ''))}**",
            "",
        ]

    comp = _csv(OUT / "evaluation" / "model_comparison.csv")
    if not comp.empty:
        cols = [c for c in ["MODEL", "TEST_N", "ACCURACY", "BALANCED_ACCURACY", "MACRO_F1", "RECALL_ABNORMAL"] if c in comp.columns]
        lines += ["## 6. Common-test comparison", "", comp[cols].to_markdown(index=False), ""]

    lines += [
        "## 7. Report-ready evidence",
        "",
        "See `outputs/thesis/OUTPUT_INDEX.csv` and `docs/CHAPTER_4_5_OUTPUT_MAP.md`.",
        "",
        "## Interpretation rule",
        "",
        "ARIMA and MLP are parallel branches. MLP does not use ARIMA prediction as input. Hybrid integration occurs only at decision level.",
        "",
    ]

    path = OUT / "RUN_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
