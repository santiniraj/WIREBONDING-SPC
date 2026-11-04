from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    CTQS,
    MACHINE_CORRECTIONS,
    MIN_EVENTS_PER_CTQ_FOR_POPULATION,
    PACKAGE_CORRECTIONS,
    STUDY_START_DATE,
    SUBGROUP_SIZE,
)

VALUE_PATTERN = re.compile(r"VALUE(\d+)\(PSN_NO\)", re.IGNORECASE)


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    output = root / "outputs"
    paths = {
        "root": output,
        "data": output / "data",
        "eda": output / "eda",
        "spc": output / "spc",
        "models": output / "models",
        "evaluation": output / "evaluation",
        "deployment": output / "deployment",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def normalize_package(value: object) -> str:
    """Normalize manual-entry package variants using optional local correction mappings."""
    text = normalize_text(value)
    text = re.sub(r"[\s\-_]+", "", text)
    return PACKAGE_CORRECTIONS.get(text, text)


def normalize_machine(value: object) -> str:
    """Normalize a source machine identifier without exposing plant-specific rules.

    Generic whitespace, hyphen and underscore cleanup is applied here. The distributed
    anonymous dataset does not require any company-specific correction mapping.
    """
    text = normalize_text(value)
    text = re.sub(r"[\s\-_]+", "", text)
    return MACHINE_CORRECTIONS.get(text, text)


def machine_family(value: object) -> str:
    """Return a generic source prefix for private grouping only.

    The prefix is removed from all public/thesis outputs by the anonymization layer.
    """
    machine = normalize_machine(value)
    match = re.match(r"([A-Z]+)", machine)
    return match.group(1) if match else "OTHER"


def normalize_wire_type(value: object) -> str:
    return re.sub(r"[\s\-_]+", "_", normalize_text(value)).strip("_")


def parse_wire_diameter(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace("MIL", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else np.nan


def parse_spec(value: object) -> tuple[float, float]:
    """Parse specification strings such as '3.500/' or '15.000/50.000'."""
    if pd.isna(value):
        return np.nan, np.nan
    parts = str(value).strip().split("/")

    def first_number(text: str) -> float:
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group()) if match else np.nan

    lower = first_number(parts[0]) if parts else np.nan
    upper = first_number(parts[1]) if len(parts) > 1 else np.nan
    return lower, upper


def _value_columns(columns: Iterable[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for col in columns:
        match = VALUE_PATTERN.fullmatch(str(col).strip())
        if match:
            found.append((int(match.group(1)), str(col).strip()))
    return [col for _, col in sorted(found)]


def validate_workbook(path_or_buffer: object) -> pd.DataFrame:
    required = {
        "TID", "LOT_ID", "DEVICE", "PACKAGE", "MACH_NO", "WIRE_DIAMETER",
        "WIRE_TYPE", "DATE_TIME", "SPEC", "VAL_TYPE",
    }
    results: list[dict[str, object]] = []
    try:
        book = pd.ExcelFile(path_or_buffer)
    except Exception as exc:
        return pd.DataFrame([{"CHECK": "Workbook", "STATUS": "FAIL", "DETAIL": str(exc)}])

    for ctq in CTQS:
        if ctq not in book.sheet_names:
            results.append({"CHECK": f"Sheet {ctq}", "STATUS": "FAIL", "DETAIL": "Missing sheet"})
            continue
        sample = pd.read_excel(path_or_buffer, sheet_name=ctq, nrows=5)
        sample.columns = [str(c).strip() for c in sample.columns]
        missing = sorted(required - set(sample.columns))
        values = _value_columns(sample.columns)
        status = "PASS" if not missing and len(values) >= SUBGROUP_SIZE else "FAIL"
        detail = (
            f"Required columns present; {len(values)} measurement columns found"
            if status == "PASS"
            else f"Missing columns={missing}; measurement columns={len(values)}"
        )
        results.append({"CHECK": f"Sheet {ctq}", "STATUS": status, "DETAIL": detail})
    return pd.DataFrame(results)


def _clean_break_mode(value: object) -> object:
    """Normalize A-E break-mode codes while preserving raw code in separate columns."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    match = re.match(r"([A-E])", text)
    return match.group(1) if match else pd.NA


def _normalization_rows(raw: pd.DataFrame, ctq: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field, func in [
        ("PACKAGE", normalize_package),
        ("MACH_NO", normalize_machine),
        ("WIRE_TYPE", normalize_wire_type),
    ]:
        if field not in raw.columns:
            continue
        temp = raw[[field]].dropna().copy()
        temp["STANDARDIZED"] = temp[field].map(func)
        counts = (
            temp.groupby([field, "STANDARDIZED"], dropna=False)
            .size().rename("ROWS").reset_index()
        )
        counts = counts[counts[field].astype(str).str.strip().str.upper() != counts["STANDARDIZED"]]
        for _, r in counts.iterrows():
            rows.append({
                "CTQ": ctq,
                "FIELD": field,
                "RAW_VALUE": str(r[field]),
                "STANDARDIZED_VALUE": str(r["STANDARDIZED"]),
                "ROWS": int(r["ROWS"]),
            })
    return rows




def _load_anonymous_event_csv(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the distributed anonymized event-level dataset and re-run cleaning checks.

    The shared project starts from an already anonymized event-level source so no company
    identifiers are required to reproduce the thesis analysis. The code still validates
    subgroup completeness, recomputes event statistics, standardizes categorical fields,
    removes duplicate/invalid events and preserves chronological order.

    The raw two-row ITEM_VAL/FAILURE_CODE workbook ingestion remains implemented in
    ``load_and_clean_data`` for controlled local use if a private workbook is supplied.
    """
    raw = pd.read_csv(path)
    required = {
        "EVENT_ID", "CTQ", "TID", "LOT_ID", "DEVICE", "PACKAGE",
        "MACH_NO", "WIRE_TYPE", "WIRE_DIAMETER", "DATE_TIME", "SPEC",
        *{f"VALUE_{i}" for i in range(1, SUBGROUP_SIZE + 1)},
        *{f"BREAK_MODE_{i}" for i in range(1, SUBGROUP_SIZE + 1)},
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Anonymous event dataset is missing required columns: {missing}")

    raw["DATE_TIME"] = pd.to_datetime(raw["DATE_TIME"], errors="coerce")
    raw["PACKAGE"] = raw["PACKAGE"].map(normalize_package)
    raw["MACH_NO"] = raw["MACH_NO"].map(normalize_machine)
    raw["WIRE_TYPE"] = raw["WIRE_TYPE"].map(normalize_wire_type)
    raw["WIRE_DIAMETER"] = raw["WIRE_DIAMETER"].map(parse_wire_diameter)

    value_names = [f"VALUE_{i}" for i in range(1, SUBGROUP_SIZE + 1)]
    mode_names = [f"BREAK_MODE_{i}" for i in range(1, SUBGROUP_SIZE + 1)]
    raw_mode_cols = [f"BREAK_MODE_RAW_{i}" for i in range(1, SUBGROUP_SIZE + 1)]
    for c in value_names:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in mode_names:
        raw[c] = raw[c].map(_clean_break_mode)
    for i, c in enumerate(raw_mode_cols, start=1):
        if c not in raw.columns:
            raw[c] = raw[f"BREAK_MODE_{i}"]

    numeric = raw[value_names]
    raw["MEAS_COUNT"] = numeric.notna().sum(axis=1)
    raw["AVG"] = numeric.mean(axis=1)
    raw["SIGMA"] = numeric.std(axis=1, ddof=1)
    raw["RANGE"] = numeric.max(axis=1) - numeric.min(axis=1)
    raw["MIN_VALUE"] = numeric.min(axis=1)
    raw["MAX_VALUE"] = numeric.max(axis=1)
    raw["CV"] = np.where(raw["AVG"].abs() > 1e-12, raw["SIGMA"] / raw["AVG"].abs(), np.nan)

    specs = raw["SPEC"].apply(parse_spec)
    raw["LSL"] = [x[0] for x in specs]
    raw["USL"] = [x[1] for x in specs]
    raw["SPEC_KEY"] = raw.apply(
        lambda r: f"{r['LSL'] if pd.notna(r['LSL']) else ''}/{r['USL'] if pd.notna(r['USL']) else ''}", axis=1
    )

    for mode in "ABCDE":
        raw[f"MODE_{mode}_COUNT"] = raw[mode_names].eq(mode).sum(axis=1)
    raw["BREAK_MODE_PATTERN"] = raw[mode_names].apply(
        lambda row: "/".join([str(x) for x in row if pd.notna(x)]), axis=1
    )

    if "IS_WIRE_BOND_MACHINE" not in raw.columns:
        raw["IS_WIRE_BOND_MACHINE"] = True
    else:
        raw["IS_WIRE_BOND_MACHINE"] = raw["IS_WIRE_BOND_MACHINE"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    raw["MACHINE_FAMILY"] = "WB"

    duplicate_mask = raw.duplicated("EVENT_ID", keep="last")
    valid = (
        raw["DATE_TIME"].notna()
        & raw["AVG"].notna()
        & raw["RANGE"].notna()
        & raw["MEAS_COUNT"].eq(SUBGROUP_SIZE)
        & raw["PACKAGE"].ne("")
        & raw["WIRE_TYPE"].ne("")
        & raw["WIRE_DIAMETER"].notna()
    )

    audit_rows = []
    for ctq in CTQS:
        m = raw["CTQ"].eq(ctq)
        audit_rows.append({
            "CTQ": ctq,
            "SOURCE_EVENTS": int(m.sum()),
            "DUPLICATE_EVENTS_REMOVED": int((m & duplicate_mask).sum()),
            "INVALID_EVENTS_REMOVED": int((m & ~valid).sum()),
            "CLEAN_EVENTS": int((m & valid & ~duplicate_mask).sum()),
            "SOURCE_FORMAT": "ANONYMIZED_EVENT_LEVEL",
        })

    cleaned = raw.loc[valid & ~duplicate_mask].copy()
    cleaned = cleaned.sort_values(["DATE_TIME", "CTQ", "MACH_NO", "EVENT_ID"]).reset_index(drop=True)
    normalization = pd.DataFrame(columns=["CTQ", "FIELD", "RAW_VALUE", "STANDARDIZED_VALUE", "ROWS"])
    return cleaned, pd.DataFrame(audit_rows), normalization


def load_and_clean_data(
    workbook: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean the configured source dataset.

    The distributed project uses an anonymized event-level CSV. If a controlled private
    Excel workbook is supplied instead, the same function performs the original two-row
    ITEM_VAL/FAILURE_CODE reconstruction described in the research methodology.
    """
    source = Path(workbook)
    if source.suffix.lower() == ".csv":
        return _load_anonymous_event_csv(source)

    """
    Convert the two-row raw test structure into one event-level row.

    ITEM_VAL and FAILURE_CODE are paired using TID plus lot/date/machine/device context.
    Each VALUE_i remains paired with BREAK_MODE_i. LOT_ID alone is not used because a lot
    can contain multiple tests.
    """
    validation = validate_workbook(workbook)
    if validation["STATUS"].eq("FAIL").any():
        raise ValueError("Workbook validation failed:\n" + validation.to_string(index=False))

    cleaned_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    normalization_rows: list[dict[str, object]] = []

    for ctq in CTQS:
        raw = pd.read_excel(workbook, sheet_name=ctq)
        raw.columns = [str(c).strip() for c in raw.columns]
        normalization_rows.extend(_normalization_rows(raw, ctq))

        raw["VAL_TYPE_N"] = raw["VAL_TYPE"].astype(str).str.strip().str.upper()
        raw["DATE_TIME"] = pd.to_datetime(raw["DATE_TIME"], errors="coerce")

        value_cols = _value_columns(raw.columns)[:SUBGROUP_SIZE]
        if len(value_cols) < SUBGROUP_SIZE:
            raise ValueError(f"{ctq}: fewer than {SUBGROUP_SIZE} VALUE columns were found.")

        item = raw[raw["VAL_TYPE_N"].eq("ITEM_VAL")].copy()
        failure = raw[raw["VAL_TYPE_N"].eq("FAILURE_CODE")].copy()

        for frame in (item, failure):
            frame["LOT_ID"] = frame["LOT_ID"].astype(str).str.strip()
            frame["DEVICE"] = frame["DEVICE"].map(normalize_text)
            frame["PACKAGE_RAW"] = frame["PACKAGE"]
            frame["MACH_NO_RAW"] = frame["MACH_NO"]
            frame["WIRE_TYPE_RAW"] = frame["WIRE_TYPE"]
            frame["PACKAGE"] = frame["PACKAGE"].map(normalize_package)
            frame["MACH_NO"] = frame["MACH_NO"].map(normalize_machine)
            frame["WIRE_TYPE"] = frame["WIRE_TYPE"].map(normalize_wire_type)
            frame["WIRE_DIAMETER"] = frame["WIRE_DIAMETER"].map(parse_wire_diameter)

        merge_keys = ["TID", "LOT_ID", "DATE_TIME", "MACH_NO", "DEVICE"]
        failure_keep = failure[merge_keys + value_cols].copy()
        duplicate_failure = int(failure_keep.duplicated(merge_keys, keep="last").sum())
        failure_keep = failure_keep.drop_duplicates(merge_keys, keep="last")
        failure_keep = failure_keep.rename(
            columns={value_cols[i]: f"BREAK_MODE_RAW_{i+1}" for i in range(SUBGROUP_SIZE)}
        )

        item_rows_before = len(item)
        duplicate_item = int(item.duplicated(merge_keys, keep="last").sum())
        item = item.drop_duplicates(merge_keys, keep="last")
        item = item.merge(failure_keep, on=merge_keys, how="left", validate="one_to_one")

        for i, source_col in enumerate(value_cols, start=1):
            item[f"VALUE_{i}"] = pd.to_numeric(item[source_col], errors="coerce")
            item[f"BREAK_MODE_{i}"] = item[f"BREAK_MODE_RAW_{i}"].map(_clean_break_mode)

        value_names = [f"VALUE_{i}" for i in range(1, SUBGROUP_SIZE + 1)]
        mode_names = [f"BREAK_MODE_{i}" for i in range(1, SUBGROUP_SIZE + 1)]
        raw_mode_cols = [f"BREAK_MODE_RAW_{i}" for i in range(1, SUBGROUP_SIZE + 1)]

        numeric = item[value_names]
        item["MEAS_COUNT"] = numeric.notna().sum(axis=1)
        item["AVG"] = numeric.mean(axis=1)
        item["SIGMA"] = numeric.std(axis=1, ddof=1)
        item["RANGE"] = numeric.max(axis=1) - numeric.min(axis=1)
        item["MIN_VALUE"] = numeric.min(axis=1)
        item["MAX_VALUE"] = numeric.max(axis=1)
        item["CV"] = np.where(item["AVG"].abs() > 1e-12, item["SIGMA"] / item["AVG"].abs(), np.nan)

        specs = item["SPEC"].apply(parse_spec)
        item["LSL"] = [x[0] for x in specs]
        item["USL"] = [x[1] for x in specs]
        item["SPEC_KEY"] = item.apply(
            lambda r: f"{r['LSL'] if pd.notna(r['LSL']) else ''}/{r['USL'] if pd.notna(r['USL']) else ''}", axis=1
        )

        for mode in "ABCDE":
            item[f"MODE_{mode}_COUNT"] = item[mode_names].eq(mode).sum(axis=1)
        item["BREAK_MODE_PATTERN"] = item[mode_names].apply(
            lambda row: "/".join([str(x) for x in row if pd.notna(x)]), axis=1
        )

        item["CTQ"] = ctq
        item["EVENT_ID"] = (
            ctq + "|" + item["TID"].astype(str) + "|" + item["LOT_ID"].astype(str) + "|"
            + item["DATE_TIME"].dt.strftime("%Y%m%d%H%M%S").fillna("") + "|" + item["MACH_NO"]
        )
        item["IS_WIRE_BOND_MACHINE"] = item["MACH_NO"].str.match(r"^WB[OIR]\d{3}$", na=False)
        item["MACHINE_FAMILY"] = item["MACH_NO"].map(machine_family)

        valid = (
            item["DATE_TIME"].notna()
            & item["AVG"].notna()
            & item["RANGE"].notna()
            & item["MEAS_COUNT"].eq(SUBGROUP_SIZE)
            & item["LOT_ID"].ne("")
            & item["PACKAGE"].ne("")
            & item["WIRE_TYPE"].ne("")
            & item["WIRE_DIAMETER"].notna()
        )

        keep_cols = [
            "EVENT_ID", "CTQ", "TID", "LOT_ID", "DEVICE", "PACKAGE", "MACH_NO",
            "WIRE_TYPE", "WIRE_DIAMETER", "DATE_TIME", "SHIFT", "SPEC", "SPEC_KEY",
            "LSL", "USL", "MEAS_COUNT", "AVG", "SIGMA", "RANGE", "CV", "MIN_VALUE",
            "MAX_VALUE", "IS_WIRE_BOND_MACHINE", "MACHINE_FAMILY", *value_names, *mode_names, *raw_mode_cols,
            "MODE_A_COUNT", "MODE_B_COUNT", "MODE_C_COUNT", "MODE_D_COUNT", "MODE_E_COUNT",
            "BREAK_MODE_PATTERN",
        ]
        for col in keep_cols:
            if col not in item.columns:
                item[col] = pd.NA

        paired = item["BREAK_MODE_PATTERN"].ne("")
        audit_rows.append({
            "CTQ": ctq,
            "RAW_ROWS": int(len(raw)),
            "ITEM_VAL_ROWS": int(item_rows_before),
            "FAILURE_CODE_ROWS": int(len(failure)),
            "DUPLICATE_ITEM_ROWS_REMOVED": duplicate_item,
            "DUPLICATE_FAILURE_ROWS_REMOVED": duplicate_failure,
            "PAIRED_EVENTS": int(paired.sum()),
            "UNPAIRED_EVENTS": int((~paired).sum()),
            "INVALID_EVENTS_REMOVED": int((~valid).sum()),
            "CLEAN_EVENTS": int(valid.sum()),
        })
        cleaned_frames.append(item.loc[valid, keep_cols].copy())

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    cleaned = cleaned.drop_duplicates("EVENT_ID", keep="last")
    cleaned = cleaned.sort_values(["DATE_TIME", "CTQ", "MACH_NO", "EVENT_ID"]).reset_index(drop=True)
    normalization = pd.DataFrame(normalization_rows)
    if not normalization.empty:
        normalization = normalization.sort_values(["FIELD", "ROWS"], ascending=[True, False]).reset_index(drop=True)
    return cleaned, pd.DataFrame(audit_rows), normalization


def build_eda_tables(cleaned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Simple beginner-friendly EDA summaries used by the notebook/admin dashboard."""
    tables: dict[str, pd.DataFrame] = {}
    tables["package_counts"] = cleaned.groupby("PACKAGE").size().rename("EVENTS").reset_index().sort_values("EVENTS", ascending=False)
    tables["wire_type_counts"] = cleaned.groupby("WIRE_TYPE").size().rename("EVENTS").reset_index().sort_values("EVENTS", ascending=False)
    tables["diameter_counts"] = cleaned.groupby("WIRE_DIAMETER").size().rename("EVENTS").reset_index().sort_values("EVENTS", ascending=False)
    tables["machine_counts"] = cleaned[cleaned["IS_WIRE_BOND_MACHINE"]].groupby("MACH_NO").size().rename("EVENTS").reset_index().sort_values("EVENTS", ascending=False)
    tables["ctq_counts"] = cleaned.groupby("CTQ").size().rename("EVENTS").reset_index()
    missing = cleaned[["DEVICE", "PACKAGE", "MACH_NO", "WIRE_TYPE", "WIRE_DIAMETER", "DATE_TIME", "LSL", "USL"]].isna().sum()
    tables["missing_summary"] = missing.rename("MISSING_ROWS").rename_axis("FIELD").reset_index()

    mode_rows = []
    total_positions = max(1, len(cleaned) * SUBGROUP_SIZE)
    for mode in "ABCDE":
        occurrences = int(cleaned[f"MODE_{mode}_COUNT"].sum())
        mode_rows.append({"MODE": mode, "OCCURRENCES": occurrences, "PERCENT_CODE_POSITIONS": 100 * occurrences / total_positions})
    tables["break_mode_counts"] = pd.DataFrame(mode_rows)
    return tables


def select_study_population(
    cleaned: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """
    Select the largest usable PACKAGE + WIRE_TYPE + WIRE_DIAMETER population.

    Machine is deliberately not used to choose the population. All valid wire-bond machines
    inside the selected population are retained and compared in later analysis. Product identity is kept privately
    for traceability but is not used as the primary grouping because the project assumption is
    that performance is comparable within the selected package / wire / diameter population.
    """
    eligible = cleaned[
        cleaned["DATE_TIME"].ge(pd.Timestamp(STUDY_START_DATE))
        & cleaned["IS_WIRE_BOND_MACHINE"]
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible wire-bond events remain after cleaning.")

    audit = (
        eligible.groupby(["PACKAGE", "WIRE_TYPE", "WIRE_DIAMETER", "CTQ"], dropna=False)
        .size().unstack(fill_value=0).reindex(columns=list(CTQS), fill_value=0).reset_index()
    )
    audit["MIN_CTQ_EVENTS"] = audit[list(CTQS)].min(axis=1)
    audit["TOTAL_EVENTS"] = audit[list(CTQS)].sum(axis=1)
    audit["ELIGIBLE"] = audit["MIN_CTQ_EVENTS"].ge(MIN_EVENTS_PER_CTQ_FOR_POPULATION)
    audit = audit.sort_values(["ELIGIBLE", "TOTAL_EVENTS", "MIN_CTQ_EVENTS"], ascending=[False, False, False])

    candidates = audit[audit["ELIGIBLE"]].copy()
    if candidates.empty:
        raise ValueError("No package / wire type / wire diameter combination has enough WP, BS and SP data.")
    chosen = candidates.iloc[0]

    study = eligible[
        eligible["PACKAGE"].eq(chosen["PACKAGE"])
        & eligible["WIRE_TYPE"].eq(chosen["WIRE_TYPE"])
        & eligible["WIRE_DIAMETER"].eq(chosen["WIRE_DIAMETER"])
    ].copy()

    machine_audit = (
        study.groupby(["MACHINE_FAMILY", "MACH_NO", "CTQ"]).size().unstack(fill_value=0)
        .reindex(columns=list(CTQS), fill_value=0).reset_index()
    )
    machine_audit["TOTAL_EVENTS"] = machine_audit[list(CTQS)].sum(axis=1)
    machine_audit["DEVICES"] = machine_audit["MACH_NO"].map(study.groupby("MACH_NO")["DEVICE"].nunique())
    machine_audit = machine_audit.sort_values("TOTAL_EVENTS", ascending=False).reset_index(drop=True)

    counts = study["CTQ"].value_counts().reindex(CTQS, fill_value=0)
    config = {
        "selection_rule": "Largest usable PACKAGE + WIRE_TYPE + WIRE_DIAMETER population; all associated wire-bond machines retained.",
        "analysis_assumption": "Within the selected package / wire type / wire diameter population, device variants are treated as comparable for machine-level analysis; device is retained for traceability.",
        "selected_package": str(chosen["PACKAGE"]),
        "selected_wire_type": str(chosen["WIRE_TYPE"]),
        "selected_wire_diameter": float(chosen["WIRE_DIAMETER"]),
        "study_start_date_filter": STUDY_START_DATE,
        "machine_count": int(study["MACH_NO"].nunique()),
        "machine_families": sorted(study["MACHINE_FAMILY"].dropna().unique().tolist()),
        "machine_family_counts": {k: int(v) for k, v in study[["MACHINE_FAMILY", "MACH_NO"]].drop_duplicates().groupby("MACHINE_FAMILY").size().to_dict().items()},
        "device_count": int(study["DEVICE"].nunique()),
        "machines": sorted(study["MACH_NO"].unique().tolist()),
        "events_by_ctq": {ctq: int(counts[ctq]) for ctq in CTQS},
        "total_events": int(len(study)),
        "date_start": study["DATE_TIME"].min().isoformat(),
        "date_end": study["DATE_TIME"].max().isoformat(),
    }
    study = study.sort_values(["DATE_TIME", "MACH_NO", "CTQ", "EVENT_ID"]).reset_index(drop=True)
    return study, config, audit.reset_index(drop=True), machine_audit


def save_study_config(config: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
