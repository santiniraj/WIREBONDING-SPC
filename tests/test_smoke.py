from __future__ import annotations

from config import DATA_FILE, FORECAST_HORIZON
from src.data_cleaning import load_and_clean_data
from src.features import CATEGORICAL_FEATURES


def test_anonymous_source_loads_and_contains_all_ctqs():
    cleaned, audit, _ = load_and_clean_data(DATA_FILE)
    assert not cleaned.empty
    assert set(cleaned["CTQ"].unique()) == {"WP", "BS", "SP"}
    assert not audit.empty


def test_forecast_horizon_is_fifth_event():
    assert FORECAST_HORIZON == 5


def test_mlp_does_not_use_device_or_arima_output():
    assert "DEVICE" not in CATEGORICAL_FEATURES
    assert "ARIMA" not in " ".join(CATEGORICAL_FEATURES).upper()
