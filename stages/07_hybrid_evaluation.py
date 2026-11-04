# %% [markdown]
# # Stage 6 — Common-Test Evaluation and Fork Integration
#
# The Fork comparison uses only test observations that are available to both ARIMA
# and MLP. This gives a fair comparison among:
#
# 1. SPC Persistence
# 2. ARIMA
# 3. MLP
# 4. Hybrid
#
# Hybrid adopts the more severe state between the two predictive branches.

# %%
from pathlib import Path
from src.stage_helpers import prepare_features
from src.models import train_arima, train_mlp, evaluate_fork

features, labelled, split_meta, split_summary, split_class_summary = prepare_features()
mlp = train_mlp(features, labelled, Path("outputs/models"))
arima = train_arima(features, split_meta)
comparison, hybrid_predictions = evaluate_fork(mlp.predictions, arima.predictions)

print(comparison.to_string(index=False))

# %% [markdown]
# ## Interpretation
# The Hybrid is not assumed to be numerically superior. Its research contribution is
# decision-level integration of complementary trend and state information.
#
# Report outputs:
# - `figure_model_comparison.png`
# - `table_model_comparison.csv`
