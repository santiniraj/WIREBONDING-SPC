from __future__ import annotations

import numpy as np
import pandas as pd

from config import FORECAST_HORIZON, STATE_ORDER, TRAIN_FRACTION, VALIDATION_FRACTION

# Features intentionally remain compact and explainable. Nelson-rule history is included
# because RQ1 produces those signals and RQ2 tests whether they help future-state prediction.
NUMERIC_FEATURES = [
    "AVG",
    "SIGMA",
    "RANGE",
    "CV",
    "STANDARDIZED_DEVIATION",
    "RANGE_RATIO",
    "CAPABILITY_VALUE",
    "NELSON_RULE_COUNT",
    "NELSON_ANY",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8",
    "MODE_A_COUNT", "MODE_B_COUNT", "MODE_C_COUNT", "MODE_D_COUNT", "MODE_E_COUNT",
    "AVG_LAG_1",
    "Z_LAG_1", "Z_LAG_2", "Z_LAG_3",
    "RANGE_RATIO_LAG_1",
    "MOVING_AVG_5",
    "ROLLING_Z_MEAN_3",
    "ROLLING_Z_STD_5",
    "PREVIOUS_WARNING_RATE_5",
    "PREVIOUS_ABNORMAL_RATE_5",
    "PREVIOUS_NELSON_RATE_5",
    "PREVIOUS_NELSON_RATE_10",
    "PREVIOUS_NELSON_EPISODE_RATE_10",
    "EVENTS_SINCE_PREVIOUS_NELSON",
    "HOURS_SINCE_PREVIOUS",
    "CURRENT_STATE_RANK",
]

# DEVICE remains available for traceability but is not a model feature, following the
# project scope: population is fixed by package/wire/diameter and machines are compared.
CATEGORICAL_FEATURES = ["CTQ", "MACH_NO"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def derive_time_split_meta(data: pd.DataFrame) -> dict[str, object]:
    """Derive global chronological train/validation/test cut-offs before SPC modelling.

    The cut-offs are calculated from the selected study population before control limits are
    estimated. This allows SPC reference limits and capability values to be built only from
    historical training-period data, avoiding future information entering model evaluation.
    """
    times = np.array(sorted(pd.to_datetime(data["DATE_TIME"], errors="coerce").dropna().unique()))
    if len(times) < 10:
        raise ValueError("Too few chronological observations for train/validation/test splitting.")

    train_idx = min(len(times) - 3, max(1, int(len(times) * TRAIN_FRACTION) - 1))
    val_idx = min(
        len(times) - 2,
        max(train_idx + 1, int(len(times) * (TRAIN_FRACTION + VALIDATION_FRACTION)) - 1),
    )
    train_cutoff = pd.Timestamp(times[train_idx])
    validation_cutoff = pd.Timestamp(times[val_idx])
    return {
        "train_cutoff": train_cutoff.isoformat(),
        "validation_cutoff": validation_cutoff.isoformat(),
        "forecast_horizon_events": FORECAST_HORIZON,
        "split_method": "Chronological global 60/20/20 split with +5 target-boundary purging",
        "spc_reference_policy": "SPC limits and capability use training-period history only",
    }


def _events_since_previous_signal(signal: pd.Series) -> pd.Series:
    """Number of events since the previous Nelson-trigger event, excluding current event."""
    sig = signal.fillna(False).astype(bool).to_numpy()
    out = np.full(len(sig), np.nan, dtype=float)
    last_idx: int | None = None
    for i in range(len(sig)):
        if last_idx is not None:
            out[i] = float(i - last_idx)
        if sig[i]:
            last_idx = i
    return pd.Series(out, index=signal.index, dtype=float)


def build_features(spc: pd.DataFrame) -> pd.DataFrame:
    """Create proposal-aligned statistical, SPC, Nelson, break-mode and time features."""
    source = spc[spc["SPC_AVAILABLE"].astype(bool)].copy()
    frames: list[pd.DataFrame] = []

    for (machine, ctq), group in source.groupby(["MACH_NO", "CTQ"], sort=True):
        g = group.sort_values("DATE_TIME").copy().reset_index(drop=True)
        z = pd.to_numeric(g["STANDARDIZED_DEVIATION"], errors="coerce")

        # Historical/lag features use information available before the current event.
        g["AVG_LAG_1"] = pd.to_numeric(g["AVG"], errors="coerce").shift(1)
        g["Z_LAG_1"] = z.shift(1)
        g["Z_LAG_2"] = z.shift(2)
        g["Z_LAG_3"] = z.shift(3)
        g["RANGE_RATIO_LAG_1"] = pd.to_numeric(g["RANGE_RATIO"], errors="coerce").shift(1)
        g["MOVING_AVG_5"] = pd.to_numeric(g["AVG"], errors="coerce").shift(1).rolling(5, min_periods=1).mean()
        g["ROLLING_Z_MEAN_3"] = z.shift(1).rolling(3, min_periods=1).mean()
        g["ROLLING_Z_STD_5"] = z.shift(1).rolling(5, min_periods=2).std()

        previous_warning = g["PROCESS_STATE"].eq("WARNING").astype(float).shift(1)
        previous_abnormal = g["PROCESS_STATE"].eq("ABNORMAL").astype(float).shift(1)
        previous_nelson = g["NELSON_ANY"].astype(float).shift(1)
        previous_episode = g["NELSON_EPISODE_START_ANY"].astype(float).shift(1)

        g["PREVIOUS_WARNING_RATE_5"] = previous_warning.rolling(5, min_periods=1).mean()
        g["PREVIOUS_ABNORMAL_RATE_5"] = previous_abnormal.rolling(5, min_periods=1).mean()
        g["PREVIOUS_NELSON_RATE_5"] = previous_nelson.rolling(5, min_periods=1).mean()
        g["PREVIOUS_NELSON_RATE_10"] = previous_nelson.rolling(10, min_periods=1).mean()
        g["PREVIOUS_NELSON_EPISODE_RATE_10"] = previous_episode.rolling(10, min_periods=1).sum()
        g["EVENTS_SINCE_PREVIOUS_NELSON"] = _events_since_previous_signal(g["NELSON_ANY"])
        g["HOURS_SINCE_PREVIOUS"] = pd.to_datetime(g["DATE_TIME"]).diff().dt.total_seconds().div(3600)
        g["CURRENT_STATE_RANK"] = g["PROCESS_STATE"].map(STATE_ORDER).astype(float)

        # MLP target: fifth next event of the same CTQ on the same machine.
        g["TARGET_FUTURE_STATE"] = g["PROCESS_STATE"].shift(-FORECAST_HORIZON)
        g["TARGET_FUTURE_TIME"] = g["DATE_TIME"].shift(-FORECAST_HORIZON)
        g["TARGET_FUTURE_EVENT_ID"] = g["EVENT_ID"].shift(-FORECAST_HORIZON)
        frames.append(g)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["DATE_TIME", "MACH_NO", "CTQ"]).reset_index(drop=True)


def chronological_split(
    features: pd.DataFrame,
    split_meta: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split with target-boundary purging.

    If split_meta is supplied, its cut-offs are reused so the same chronology controls both
    SPC reference construction and predictive evaluation.
    """
    labelled = features[features["TARGET_FUTURE_STATE"].isin(STATE_ORDER)].copy()
    if labelled.empty:
        raise ValueError("No labelled future-state rows are available for modelling.")

    if split_meta is None:
        split_meta = derive_time_split_meta(labelled)

    train_cutoff = pd.Timestamp(split_meta["train_cutoff"])
    validation_cutoff = pd.Timestamp(split_meta["validation_cutoff"])

    labelled["SPLIT"] = "PURGED"
    train_mask = labelled["DATE_TIME"].le(train_cutoff) & labelled["TARGET_FUTURE_TIME"].le(train_cutoff)
    validation_mask = (
        labelled["DATE_TIME"].gt(train_cutoff)
        & labelled["DATE_TIME"].le(validation_cutoff)
        & labelled["TARGET_FUTURE_TIME"].le(validation_cutoff)
    )
    test_mask = labelled["DATE_TIME"].gt(validation_cutoff)

    labelled.loc[train_mask, "SPLIT"] = "TRAIN"
    labelled.loc[validation_mask, "SPLIT"] = "VALIDATION"
    labelled.loc[test_mask, "SPLIT"] = "TEST"

    summary = (
        labelled["SPLIT"].value_counts()
        .reindex(["TRAIN", "VALIDATION", "TEST", "PURGED"], fill_value=0)
        .rename_axis("SPLIT").reset_index(name="ROWS")
    )
    summary["PERCENT_OF_LABELLED"] = 100 * summary["ROWS"] / max(1, len(labelled))

    class_rows = []
    for split_name in ["TRAIN", "VALIDATION", "TEST"]:
        s = labelled[labelled["SPLIT"].eq(split_name)]
        for state in ["NORMAL", "WARNING", "ABNORMAL"]:
            class_rows.append({
                "SPLIT": split_name,
                "STATE": state,
                "ROWS": int(s["TARGET_FUTURE_STATE"].eq(state).sum()),
            })
    class_summary = pd.DataFrame(class_rows)

    meta = dict(split_meta)
    meta["forecast_horizon_events"] = FORECAST_HORIZON
    meta["split_method"] = "Chronological global split with +5 target-boundary purging"
    return labelled, meta, summary, class_summary
