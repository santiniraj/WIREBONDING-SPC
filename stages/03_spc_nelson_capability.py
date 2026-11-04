# %% [markdown]
# # Stage 3 — X̄–R SPC, Nelson Rules, Capability and Break Mode
#
# X̄ monitors subgroup mean. R monitors within-subgroup variation.
# Nelson N1–N8 are applied on each Machine–CTQ stream.
#
# State logic:
# - ABNORMAL: N1, control-limit violation, or specification violation
# - WARNING: N2–N8 only, without an Abnormal condition
# - NORMAL: none of the above

# %%
from src.stage_helpers import prepare_spc

study, spc, limits, nelson_summary, machine_spc_summary, capability, split_meta = prepare_spc()

print("SPC available:", int(spc["SPC_AVAILABLE"].sum()), "/", len(spc))
print("\nProcess states:")
print(spc.loc[spc["SPC_AVAILABLE"], "PROCESS_STATE"].value_counts())
print("\nNelson summary by rule:")
print(nelson_summary.groupby("RULE")[["TRIGGER_COUNT", "EPISODE_COUNT"]].sum())
print("\nCapability sample:")
print(capability.head(10).to_string(index=False))

# %% [markdown]
# ## Thesis-safe outputs
# - `figure_representative_xbar_nelson.png`
# - `figure_representative_r_chart.png`
# - `figure_nelson_rules.png`
# - `figure_process_state_distribution.png`
# - `figure_capability_by_stream.png`
# - `figure_break_mode_distribution.png`
