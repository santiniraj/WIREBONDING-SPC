# %% [markdown]
# # Stage 5B — MLP Future-State Classification Branch
#
# MLP is independent of ARIMA. It receives engineered SPC features and predicts the
# state of event +5: Normal, Warning or Abnormal.
# Random oversampling is applied only to development data. Validation Macro F1 selects
# the network configuration; test data remain untouched until final evaluation.

# %%
from pathlib import Path
from src.stage_helpers import prepare_features
from src.models import train_mlp

features, labelled, split_meta, split_summary, split_class_summary = prepare_features()
mlp = train_mlp(features, labelled, Path("outputs/models"))

print("MLP validation:")
print(mlp.validation.to_string(index=False))
print("\nMLP test metrics:")
print(mlp.metrics.to_string(index=False))
print("\nClass-level performance:")
print(mlp.class_report.to_string(index=False))
print("\nConfusion matrix (long form):")
print(mlp.confusion.to_string(index=False))

# %% [markdown]
# ## Report outputs after `run_all.py`
# - `figure_mlp_validation.png`
# - `figure_mlp_test_metrics.png`
# - `figure_mlp_confusion_matrix.png`
# - `figure_mlp_class_performance.png`
