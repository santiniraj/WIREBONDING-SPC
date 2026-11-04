from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

PUBLIC_ANCHOR = pd.Timestamp('2020-01-01 00:00:00')
SENSITIVE_ID_COLUMNS = {'TID','LOT_ID','DEVICE','MACHINE_FAMILY'}
RAW_MEASUREMENT_COLUMNS = {
    'SPEC','SPEC_KEY','LSL','USL','AVG','MIN_VALUE','MAX_VALUE',
    'CENTER_LINE','PROCESS_STD','UCL_XBAR','LCL_XBAR','RBAR','UCL_R','LCL_R',
    'SIGMA','RANGE','CV','MIN_VALUE','MAX_VALUE',
}
BREAK_MODE_COLUMNS = {'BREAK_MODE_PATTERN','SHIFT'}


def build_machine_map(values) -> dict[str, str]:
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    return {value: f'WB{i:02d}' for i, value in enumerate(vals, start=1)}


def anonymize_study(study: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Anonymize identifiers before modelling while preserving analytical relationships.

    Machine labels are replaced one-to-one, absolute dates are rebased to a neutral anchor,
    and product/traceability labels are masked. Numeric SPC measurements/specifications are
    left unchanged in memory because they are required for SPC/capability calculations; they
    are removed from public exports later.
    """
    out = study.copy()
    machine_map = build_machine_map(out['MACH_NO'])
    out['MACH_NO'] = out['MACH_NO'].astype(str).map(machine_map)
    if 'MACHINE_FAMILY' in out:
        out['MACHINE_FAMILY'] = 'WB'

    # Preserve chronology and elapsed-time relationships without exposing real dates.
    if 'DATE_TIME' in out and out['DATE_TIME'].notna().any():
        dt = pd.to_datetime(out['DATE_TIME'], errors='coerce')
        origin = dt.min()
        out['DATE_TIME'] = PUBLIC_ANCHOR + (dt - origin)

    if 'DEVICE' in out:
        out['DEVICE'] = 'REDACTED'
    if 'PACKAGE' in out:
        out['PACKAGE'] = 'PACKAGE_A'
    if 'WIRE_TYPE' in out:
        out['WIRE_TYPE'] = 'WIRE_TYPE_A'
    if 'WIRE_DIAMETER' in out:
        out['WIRE_DIAMETER'] = np.nan
    if 'LOT_ID' in out:
        lots = pd.Series(out['LOT_ID'].astype(str).unique())
        lot_map = {v: f'LOT{i:06d}' for i, v in enumerate(lots, start=1)}
        out['LOT_ID'] = out['LOT_ID'].astype(str).map(lot_map)
    if 'TID' in out:
        tids = pd.Series(out['TID'].astype(str).unique())
        tid_map = {v: f'REC{i:06d}' for i, v in enumerate(tids, start=1)}
        out['TID'] = out['TID'].astype(str).map(tid_map)

    # Re-key events after identifiers/dates have been anonymized.
    out = out.sort_values(['DATE_TIME','MACH_NO','CTQ']).reset_index(drop=True)
    out['EVENT_ID'] = [f'EVT{i:07d}' for i in range(1, len(out)+1)]
    return out, machine_map


def public_study_config(config: dict, machine_map: dict[str, str], study: pd.DataFrame) -> dict:
    counts = study['CTQ'].value_counts().to_dict() if 'CTQ' in study else {}
    return {
        'selection_rule': 'Largest usable Package + Wire Type + Wire Diameter group; all associated anonymized wire-bond machines retained.',
        'analysis_assumption': 'Product identifiers are excluded from report-facing outputs; machine labels are anonymized one-to-one.',
        'selected_package': 'Package A',
        'selected_wire_type': 'Wire Type A',
        'selected_wire_diameter': 'Diameter Group A',
        'study_period': 'Anonymized historical production period',
        'machine_count': int(study['MACH_NO'].nunique()),
        'machines': sorted(study['MACH_NO'].dropna().astype(str).unique().tolist()),
        'events_by_ctq': {k: int(v) for k,v in counts.items()},
        'total_events': int(len(study)),
        'confidentiality_note': 'Company-sensitive equipment, product, lot, specification and calendar-date identifiers are anonymized or omitted.',
    }



def rebase_datetime_columns(df: pd.DataFrame, anchor: pd.Timestamp = PUBLIC_ANCHOR) -> pd.DataFrame:
    """Rebase all date/time-like public fields to a neutral timeline.

    Relative ordering and elapsed intervals are preserved. The earliest parseable value
    across date/time columns is mapped to the neutral anchor.
    """
    out = df.copy()
    date_cols = [c for c in out.columns if any(k in str(c).upper() for k in ("DATE", "TIME", "CUTOFF", "REFERENCE_START", "REFERENCE_END"))]
    parsed = {}
    minima = []
    for c in date_cols:
        ser = pd.to_datetime(out[c], errors="coerce")
        if ser.notna().any():
            parsed[c] = ser
            minima.append(ser.min())
    if not minima:
        return out
    origin = min(minima)
    for c, ser in parsed.items():
        out[c] = anchor + (ser - origin)
    return out

def sanitize_for_public_export(df: pd.DataFrame, *, keep_measurement_scale: bool = False) -> pd.DataFrame:
    """Return a thesis/dashboard-safe dataframe.

    Identifiers are removed. Raw product specifications and absolute measurement scale are
    omitted by default; derived/statistical fields such as standardized deviation, range ratio,
    capability index, Nelson rules and model predictions remain available.
    """
    out = rebase_datetime_columns(df)
    drop = set(SENSITIVE_ID_COLUMNS)
    if not keep_measurement_scale:
        drop |= RAW_MEASUREMENT_COLUMNS
        drop |= BREAK_MODE_COLUMNS
        drop |= {c for c in out.columns if str(c).startswith('VALUE_') or str(c).startswith('BREAK_MODE_') or str(c).startswith('MODE_')}
    out = out.drop(columns=[c for c in drop if c in out.columns], errors='ignore')

    if 'PACKAGE' in out:
        out['PACKAGE'] = 'Package A'
    if 'WIRE_TYPE' in out:
        out['WIRE_TYPE'] = 'Wire Type A'
    if 'WIRE_DIAMETER' in out:
        out['WIRE_DIAMETER'] = 'Diameter Group A'
    if 'DEVICE' in out:
        out = out.drop(columns=['DEVICE'])
    return out


def sanitize_population_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep population counts but replace commercial/product categories with generic groups."""
    out = df.copy().reset_index(drop=True)
    if 'PACKAGE' in out:
        out['PACKAGE'] = [f'Package Group {i+1:02d}' for i in range(len(out))]
    if 'WIRE_TYPE' in out:
        out['WIRE_TYPE'] = 'Wire Type Group'
    if 'WIRE_DIAMETER' in out:
        out['WIRE_DIAMETER'] = 'Diameter Group'
    return out


def save_private_mapping(machine_map: dict[str, str], path: Path) -> None:
    """Save local-only mapping. This path should stay outside public outputs/repositories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'machine_map': machine_map}, indent=2), encoding='utf-8')
