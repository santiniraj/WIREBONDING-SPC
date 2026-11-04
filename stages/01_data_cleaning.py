# %% [markdown]
# # Stage 1 — Data Preparation and Cleaning
#
# Research alignment: RQ1 / RO1.
#
# The unified project uses an anonymized event-level source dataset so that the complete
# analysis can be reproduced without the original company workbook. This stage still
# performs the data-quality work needed before SPC: required-column checks, categorical
# normalization, numeric conversion, subgroup validation, specification parsing,
# duplicate removal, event-statistic recalculation and chronological ordering.
#
# The reusable `src/data_cleaning.py` module also retains the original raw-workbook
# ITEM_VAL / FAILURE_CODE pairing logic for controlled internal use if a private workbook
# is supplied, but the anonymous project does not need that workbook.

# %%
from config import DATA_FILE
from src.data_cleaning import load_and_clean_data

cleaned, cleaning_audit, normalization_audit = load_and_clean_data(DATA_FILE)

print("Cleaning audit")
print(cleaning_audit.to_string(index=False))
print("\nCleaned event-level shape:", cleaned.shape)
print("\nColumns used later for SPC:")
print([c for c in [
    "CTQ", "MACH_NO", "DATE_TIME", "AVG", "SIGMA", "RANGE",
    "LSL", "USL", "BREAK_MODE_PATTERN"
] if c in cleaned.columns])

# %% [markdown]
# ## Outputs used in the report
#
# The complete pipeline writes:
# - `outputs/data/data_cleaning_audit.csv`
# - `outputs/data/cleaned_all_events_anonymous.csv`
# - `outputs/thesis/figure_data_cleaning_summary.png`
# - `outputs/thesis/table_data_cleaning_audit.csv`
