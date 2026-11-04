# %% [markdown]
# # Stage 5A — ARIMA Forecasting Branch
#
# ARIMA is fitted separately to each Machine–CTQ stream using standardized SPC
# deviation. Candidate orders are selected on validation event+5 RMSE.
# Test evaluation uses expanding-origin forecasts and remains unseen during selection.

# %%
from src.stage_helpers import prepare_features
from src.models import train_arima

features, labelled, split_meta, split_summary, split_class_summary = prepare_features()
arima = train_arima(features, split_meta)

print("ARIMA validation candidates:")
print(arima.selection.head(20).to_string(index=False))
print("\nARIMA test metrics:")
print(arima.metrics.to_string(index=False))
print("\nNext five-event forecast sample:")
print(arima.forecast.head(10).to_string(index=False))

# %% [markdown]
# ## Report outputs after `run_all.py`
# - `table_arima_validation_orders.csv`
# - `table_arima_test_performance.csv`
# - `figure_arima_rmse_vs_naive.png`
# - `figure_arima_next5_on_spc.png`
