from __future__ import annotations

"""Convenience helpers for the VS Code stage files.

These functions call the same source modules used by ``run_all.py``. They are not a
second modelling implementation; they simply make each stage easier to demonstrate.
"""

from config import DATA_FILE
from src.anonymization import anonymize_study
from src.data_cleaning import load_and_clean_data, select_study_population
from src.features import build_features, chronological_split, derive_time_split_meta
from src.spc import apply_spc


def prepare_study():
    cleaned, cleaning_audit, normalization_audit = load_and_clean_data(DATA_FILE)
    study_raw, study_config_raw, population_audit, machine_population_raw = select_study_population(cleaned)
    study, machine_map = anonymize_study(study_raw)
    return cleaned, cleaning_audit, normalization_audit, study, study_config_raw, population_audit, machine_map


def prepare_spc():
    cleaned, cleaning_audit, normalization_audit, study, study_config_raw, population_audit, machine_map = prepare_study()
    split_meta = derive_time_split_meta(study)
    spc, limits, nelson_summary, machine_spc_summary, capability = apply_spc(study, split_meta)
    return study, spc, limits, nelson_summary, machine_spc_summary, capability, split_meta


def prepare_features():
    study, spc, limits, nelson_summary, machine_spc_summary, capability, split_meta = prepare_spc()
    features = build_features(spc)
    labelled, split_meta, split_summary, split_class_summary = chronological_split(features, split_meta)
    return features, labelled, split_meta, split_summary, split_class_summary
