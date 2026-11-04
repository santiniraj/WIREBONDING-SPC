# %% [markdown]
# # Stage 2 — Study Population Selection
#
# The study population is selected by the Package + Wire Type + Wire Diameter
# combination with the highest usable record count. Machine ID is not used to
# choose the population; all machines associated with the selected population remain.

# %%
from src.stage_helpers import prepare_study

cleaned, cleaning_audit, normalization_audit, study, study_config_raw, population_audit, machine_map = prepare_study()

print("Selected source population (private values shown only in this local script):")
print(study_config_raw)
print("\nAnonymized study shape:", study.shape)
print("Machines:", study["MACH_NO"].nunique())
print("Events by CTQ:")
print(study["CTQ"].value_counts())

# %% [markdown]
# ## Thesis-safe outputs
# - `outputs/thesis/figure_selected_population_by_ctq.png`
# - `outputs/thesis/figure_machine_event_distribution.png`
# - `outputs/thesis/table_selected_study_population.csv`
