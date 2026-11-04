# %% [markdown]
# # Stage 4 — Feature Engineering and Model Dataset
#
# MLP uses current SPC evidence plus historical lag/rolling features.
# The prediction target is the process state at the fifth next event of the same
# Machine–CTQ stream.
#
# The dataset is separated by production sequence. Rows near a boundary are removed
# when their event+5 target crosses into the next dataset.

# %%
from src.features import MODEL_FEATURES
from src.stage_helpers import prepare_features

features, labelled, split_meta, split_summary, split_class_summary = prepare_features()

print("Model features:")
for c in MODEL_FEATURES:
    print(" -", c)

print("\nSplit summary:")
print(split_summary.to_string(index=False))
print("\nFuture-state class distribution by split:")
print(split_class_summary.to_string(index=False))

# %% [markdown]
# ## Report outputs
# - `table_feature_groups.csv`
# - `figure_dataset_split.png`
# - `figure_split_class_distribution.png`
