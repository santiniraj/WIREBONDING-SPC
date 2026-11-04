from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BASELINE_ABNORMAL_REVIEW_RATE,
    MIN_REFERENCE_EVENTS,
    MIN_SPC_EVENTS,
    NELSON_RULES,
    SUBGROUP_SIZE,
    XBAR_R_CONSTANTS,
)


def _nelson_rule_flags(z: pd.Series) -> pd.DataFrame:
    """Apply Nelson Rules 1-8 to one chronologically ordered standardized stream.

    Each flag is attached to the event where the required sequence becomes observable.
    Consecutive flags from a persistent pattern are later grouped into one Nelson episode.
    """
    x = pd.to_numeric(z, errors="coerce").reset_index(drop=True)
    n = len(x)
    flags = {f"N{i}": np.zeros(n, dtype=bool) for i in range(1, 9)}
    values = x.to_numpy(dtype=float)
    flags["N1"] = np.abs(values) > 3

    for i in range(n):
        if i >= 8:
            w = x.iloc[i - 8 : i + 1]
            flags["N2"][i] = bool((w > 0).all() or (w < 0).all())
        if i >= 5:
            w = x.iloc[i - 5 : i + 1].to_numpy(dtype=float)
            d = np.diff(w)
            flags["N3"][i] = bool((d > 0).all() or (d < 0).all())
        if i >= 13:
            w = x.iloc[i - 13 : i + 1].to_numpy(dtype=float)
            d = np.diff(w)
            flags["N4"][i] = bool(np.all(d[:-1] * d[1:] < 0))
        if i >= 2:
            w = x.iloc[i - 2 : i + 1]
            flags["N5"][i] = bool((w > 2).sum() >= 2 or (w < -2).sum() >= 2)
        if i >= 4:
            w = x.iloc[i - 4 : i + 1]
            flags["N6"][i] = bool((w > 1).sum() >= 4 or (w < -1).sum() >= 4)
        if i >= 14:
            w = x.iloc[i - 14 : i + 1]
            flags["N7"][i] = bool((np.abs(w) < 1).all())
        if i >= 7:
            w = x.iloc[i - 7 : i + 1]
            flags["N8"][i] = bool((np.abs(w) > 1).all() and (w > 1).any() and (w < -1).any())

    return pd.DataFrame(flags)


def _add_nelson_episode_columns(g: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive trigger points into interpretable Nelson episodes.

    Example: if N2 remains true for four consecutive events, it is four trigger points but
    one sustained N2 episode. Both measures are retained for transparency.
    """
    out = g.copy()
    start_cols: list[str] = []
    for rule in NELSON_RULES:
        flag = out[rule].fillna(False).astype(bool)
        start = flag & ~flag.shift(1, fill_value=False)
        start_col = f"{rule}_EPISODE_START"
        id_col = f"{rule}_EPISODE_ID"
        out[start_col] = start
        ids = start.cumsum()
        out[id_col] = np.where(flag, ids, np.nan)
        start_cols.append(start_col)

    out["NELSON_ANY"] = out[list(NELSON_RULES)].any(axis=1)
    out["NELSON_RULE_COUNT"] = out[list(NELSON_RULES)].sum(axis=1).astype(int)
    out["NELSON_RULES_TRIGGERED"] = out.apply(
        lambda r: ", ".join([rule for rule in NELSON_RULES if bool(r[rule])]), axis=1
    )
    out["NELSON_EPISODE_START_ANY"] = out[start_cols].any(axis=1)
    out["NELSON_EPISODES_STARTED"] = out[start_cols].sum(axis=1).astype(int)
    return out


def _capability_from_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cpk/Cpl/Cpu within machine + CTQ + specification set.

    Device is retained for traceability but not used as the primary process grouping. Different
    specification sets are never mixed into one capability value.
    """
    d2 = XBAR_R_CONSTANTS[SUBGROUP_SIZE]["D2"]
    rows: list[dict[str, object]] = []

    for spec_key, s in reference.groupby("SPEC_KEY", dropna=False):
        mean = float(s["AVG"].mean()) if not s.empty else np.nan
        rbar = float(s["RANGE"].mean()) if not s.empty else np.nan
        sigma_within = rbar / d2 if np.isfinite(rbar) and rbar > 0 else np.nan
        lsl = s["LSL"].dropna().iloc[0] if s["LSL"].notna().any() else np.nan
        usl = s["USL"].dropna().iloc[0] if s["USL"].notna().any() else np.nan

        cpl = (mean - lsl) / (3 * sigma_within) if np.isfinite(lsl) and np.isfinite(sigma_within) and sigma_within > 0 else np.nan
        cpu = (usl - mean) / (3 * sigma_within) if np.isfinite(usl) and np.isfinite(sigma_within) and sigma_within > 0 else np.nan
        if np.isfinite(cpl) and np.isfinite(cpu):
            capability_type = "CPK"
            capability_value = min(cpl, cpu)
            cpk = capability_value
        elif np.isfinite(cpl):
            capability_type = "CPL"
            capability_value = cpl
            cpk = np.nan
        elif np.isfinite(cpu):
            capability_type = "CPU"
            capability_value = cpu
            cpk = np.nan
        else:
            capability_type = "N/A"
            capability_value = np.nan
            cpk = np.nan

        rows.append({
            "SPEC_KEY": spec_key,
            "CAPABILITY_EVENTS": int(len(s)),
            "CAPABILITY_MEAN": mean,
            "WITHIN_SIGMA": sigma_within,
            "LSL": lsl,
            "USL": usl,
            "CPL": cpl,
            "CPU": cpu,
            "CPK": cpk,
            "CAPABILITY_TYPE": capability_type,
            "CAPABILITY_VALUE": capability_value,
        })
    return pd.DataFrame(rows)


def apply_spc(
    study: pd.DataFrame,
    split_meta: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply machine + CTQ X-bar/R SPC, Nelson Rules and capability analysis.

    RQ1: current process abnormalities are detected by X-bar/R and Nelson N1-N8.
    RQ2 leakage control: when split_meta is supplied, control limits/capability are estimated
    only from the chronological TRAINING reference period and then frozen for later data.
    RQ3: output summaries contain machine-level Nelson episode/rate evidence for the DSS.
    """
    if split_meta is None:
        # Backward-compatible fallback for notebook use. Final run_all supplies split_meta.
        from src.features import derive_time_split_meta
        split_meta = derive_time_split_meta(study)

    train_cutoff = pd.Timestamp(split_meta["train_cutoff"])
    validation_cutoff = pd.Timestamp(split_meta["validation_cutoff"])
    constants = XBAR_R_CONSTANTS[SUBGROUP_SIZE]

    result_frames: list[pd.DataFrame] = []
    limit_rows: list[dict[str, object]] = []
    capability_rows: list[pd.DataFrame] = []

    for (machine, ctq), group in study.groupby(["MACH_NO", "CTQ"], sort=True):
        g = group.sort_values("DATE_TIME").copy().reset_index(drop=True)
        reference = g[g["DATE_TIME"].le(train_cutoff)].copy()
        available = len(g) >= MIN_SPC_EVENTS and len(reference) >= MIN_REFERENCE_EVENTS
        g["SPC_AVAILABLE"] = available

        if not available:
            g["PHASE"] = "INSUFFICIENT"
            for c in ["CENTER_LINE", "UCL_XBAR", "LCL_XBAR", "RBAR", "UCL_R", "LCL_R", "STANDARDIZED_DEVIATION", "RANGE_RATIO"]:
                g[c] = np.nan
            for rule in NELSON_RULES:
                g[rule] = False
                g[f"{rule}_EPISODE_START"] = False
                g[f"{rule}_EPISODE_ID"] = np.nan
            g["NELSON_ANY"] = False
            g["NELSON_RULE_COUNT"] = 0
            g["NELSON_RULES_TRIGGERED"] = ""
            g["NELSON_EPISODE_START_ANY"] = False
            g["NELSON_EPISODES_STARTED"] = 0
            g["CONTROL_LIMIT_VIOLATION"] = False
            g["SPEC_VIOLATION"] = False
            g["PROCESS_STATE"] = "INSUFFICIENT"
            for c in ["CPL", "CPU", "CPK", "CAPABILITY_VALUE", "WITHIN_SIGMA"]:
                g[c] = np.nan
            g["CAPABILITY_TYPE"] = "N/A"
            g["BASELINE_STATUS"] = "INSUFFICIENT"
            result_frames.append(g)
            limit_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "TOTAL_EVENTS": len(g), "REFERENCE_EVENTS": len(reference),
                "SPC_AVAILABLE": False, "BASELINE_STATUS": "INSUFFICIENT",
                "REFERENCE_START": reference["DATE_TIME"].min() if not reference.empty else pd.NaT,
                "REFERENCE_END": reference["DATE_TIME"].max() if not reference.empty else pd.NaT,
                "CENTER_LINE": np.nan, "UCL_XBAR": np.nan, "LCL_XBAR": np.nan,
                "RBAR": np.nan, "UCL_R": np.nan, "LCL_R": np.nan,
                "REFERENCE_ABNORMAL_RATE_%": np.nan,
            })
            continue

        center = float(reference["AVG"].mean())
        rbar = float(reference["RANGE"].mean())
        ucl_x = center + constants["A2"] * rbar
        lcl_x = center - constants["A2"] * rbar
        ucl_r = constants["D4"] * rbar
        lcl_r = constants["D3"] * rbar

        sigma_xbar = (ucl_x - center) / 3.0
        if not np.isfinite(sigma_xbar) or sigma_xbar <= 0:
            sigma_xbar = float(reference["AVG"].std(ddof=1))
        if not np.isfinite(sigma_xbar) or sigma_xbar <= 0:
            sigma_xbar = 1e-9

        g["PHASE"] = np.select(
            [g["DATE_TIME"].le(train_cutoff), g["DATE_TIME"].le(validation_cutoff)],
            ["REFERENCE_TRAIN", "VALIDATION"],
            default="TEST",
        )
        g["CENTER_LINE"] = center
        g["UCL_XBAR"] = ucl_x
        g["LCL_XBAR"] = lcl_x
        g["RBAR"] = rbar
        g["UCL_R"] = ucl_r
        g["LCL_R"] = lcl_r
        g["STANDARDIZED_DEVIATION"] = (g["AVG"] - center) / sigma_xbar
        g["RANGE_RATIO"] = g["RANGE"] / rbar if rbar > 0 else np.nan

        flags = _nelson_rule_flags(g["STANDARDIZED_DEVIATION"])
        for rule in flags.columns:
            g[rule] = flags[rule].to_numpy()
        g = _add_nelson_episode_columns(g)

        g["CONTROL_LIMIT_VIOLATION"] = (
            g["AVG"].gt(ucl_x) | g["AVG"].lt(lcl_x) | g["RANGE"].gt(ucl_r) | g["RANGE"].lt(lcl_r)
        )
        lower_fail = g["LSL"].notna() & g["MIN_VALUE"].lt(g["LSL"])
        upper_fail = g["USL"].notna() & g["MAX_VALUE"].gt(g["USL"])
        g["SPEC_VIOLATION"] = lower_fail | upper_fail

        # N1 is treated as a strong abnormal signal; N2-N8 are warning signals requiring review.
        warning = g[["N2", "N3", "N4", "N5", "N6", "N7", "N8"]].any(axis=1)
        abnormal = g["N1"] | g["CONTROL_LIMIT_VIOLATION"] | g["SPEC_VIOLATION"]
        g["PROCESS_STATE"] = np.select([abnormal, warning], ["ABNORMAL", "WARNING"], default="NORMAL")

        cap = _capability_from_reference(reference)
        cap["MACH_NO"] = machine
        cap["CTQ"] = ctq
        cap["REFERENCE_END"] = train_cutoff
        capability_rows.append(cap)
        g = g.merge(
            cap[["SPEC_KEY", "CPL", "CPU", "CPK", "CAPABILITY_TYPE", "CAPABILITY_VALUE", "WITHIN_SIGMA"]],
            on="SPEC_KEY", how="left", validate="many_to_one",
        )

        reference_mask = g["DATE_TIME"].le(train_cutoff)
        reference_abnormal_rate = float(g.loc[reference_mask, "PROCESS_STATE"].eq("ABNORMAL").mean())
        baseline_status = "REVIEW" if reference_abnormal_rate > BASELINE_ABNORMAL_REVIEW_RATE else "ACCEPTABLE_REFERENCE"
        g["BASELINE_STATUS"] = baseline_status

        result_frames.append(g)
        limit_rows.append({
            "MACH_NO": machine, "CTQ": ctq, "TOTAL_EVENTS": len(g), "REFERENCE_EVENTS": len(reference),
            "SPC_AVAILABLE": True, "BASELINE_STATUS": baseline_status,
            "REFERENCE_START": reference["DATE_TIME"].min(), "REFERENCE_END": reference["DATE_TIME"].max(),
            "CENTER_LINE": center, "UCL_XBAR": ucl_x, "LCL_XBAR": lcl_x,
            "RBAR": rbar, "UCL_R": ucl_r, "LCL_R": lcl_r,
            "REFERENCE_ABNORMAL_RATE_%": 100 * reference_abnormal_rate,
        })

    spc = pd.concat(result_frames, ignore_index=True).sort_values(["DATE_TIME", "MACH_NO", "CTQ"])
    limits = pd.DataFrame(limit_rows).sort_values(["MACH_NO", "CTQ"]).reset_index(drop=True)
    capability = pd.concat(capability_rows, ignore_index=True) if capability_rows else pd.DataFrame()

    # Rule-level summary: trigger points and sustained episodes are both reported.
    rule_rows: list[dict[str, object]] = []
    available_spc = spc[spc["SPC_AVAILABLE"].astype(bool)].copy()
    for (machine, ctq), group in available_spc.groupby(["MACH_NO", "CTQ"], sort=True):
        events = len(group)
        for rule, (definition, meaning) in NELSON_RULES.items():
            trigger_count = int(group[rule].sum())
            episode_count = int(group[f"{rule}_EPISODE_START"].sum())
            last_trigger = group.loc[group[rule], "DATE_TIME"].max() if trigger_count else pd.NaT
            rule_rows.append({
                "MACH_NO": machine,
                "CTQ": ctq,
                "RULE": rule,
                "EVENTS": events,
                "TRIGGER_COUNT": trigger_count,
                "EPISODE_COUNT": episode_count,
                "TRIGGER_RATE_PER_100": 100 * trigger_count / max(events, 1),
                "EPISODE_RATE_PER_100": 100 * episode_count / max(events, 1),
                "LAST_TRIGGER_TIME": last_trigger,
                "DEFINITION": definition,
                "SIMPLE_MEANING": meaning,
            })
    nelson_summary = pd.DataFrame(rule_rows)

    if available_spc.empty:
        machine_summary = pd.DataFrame()
    else:
        machine_summary = (
            available_spc.groupby(["MACH_NO", "CTQ"])
            .agg(
                EVENTS=("EVENT_ID", "size"),
                NORMAL=("PROCESS_STATE", lambda s: int((s == "NORMAL").sum())),
                WARNING=("PROCESS_STATE", lambda s: int((s == "WARNING").sum())),
                ABNORMAL=("PROCESS_STATE", lambda s: int((s == "ABNORMAL").sum())),
                NELSON_TRIGGER_EVENTS=("NELSON_ANY", "sum"),
                NELSON_RULE_TRIGGERS=("NELSON_RULE_COUNT", "sum"),
                NELSON_EPISODES=("NELSON_EPISODES_STARTED", "sum"),
                AVG_STANDARDIZED_DEVIATION=("STANDARDIZED_DEVIATION", "mean"),
                WORST_CAPABILITY=("CAPABILITY_VALUE", "min"),
                CURRENT_STATE=("PROCESS_STATE", "last"),
                BASELINE_STATUS=("BASELINE_STATUS", "last"),
            ).reset_index()
        )
        machine_summary["WARNING_RATE_%"] = 100 * machine_summary["WARNING"] / machine_summary["EVENTS"]
        machine_summary["ABNORMAL_RATE_%"] = 100 * machine_summary["ABNORMAL"] / machine_summary["EVENTS"]
        machine_summary["NELSON_TRIGGER_EVENT_RATE_%"] = 100 * machine_summary["NELSON_TRIGGER_EVENTS"] / machine_summary["EVENTS"]
        machine_summary["NELSON_EPISODE_RATE_PER_100"] = 100 * machine_summary["NELSON_EPISODES"] / machine_summary["EVENTS"]

        if not nelson_summary.empty:
            dominant = (
                nelson_summary.sort_values(["MACH_NO", "CTQ", "EPISODE_COUNT", "TRIGGER_COUNT"], ascending=[True, True, False, False])
                .groupby(["MACH_NO", "CTQ"], as_index=False)
                .first()[["MACH_NO", "CTQ", "RULE", "EPISODE_COUNT"]]
                .rename(columns={"RULE": "DOMINANT_NELSON_RULE", "EPISODE_COUNT": "DOMINANT_RULE_EPISODES"})
            )
            machine_summary = machine_summary.merge(dominant, on=["MACH_NO", "CTQ"], how="left")

    return spc.reset_index(drop=True), limits, nelson_summary, machine_summary, capability


def nelson_rule_reference() -> pd.DataFrame:
    return pd.DataFrame([
        {"RULE": rule, "DEFINITION": text[0], "SIMPLE_MEANING": text[1]}
        for rule, text in NELSON_RULES.items()
    ])
