from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "SPC_Events_Anonymous.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Study scope
CTQS = ("WP", "BS", "SP")
STUDY_START_DATE = "2000-01-01"  # public default; override locally if required
MIN_EVENTS_PER_CTQ_FOR_POPULATION = 80

# Raw subgroup structure
SUBGROUP_SIZE = 5
XBAR_R_CONSTANTS = {
    5: {"A2": 0.577, "D3": 0.0, "D4": 2.114, "D2": 2.326},
}

# SPC reference / control limits
# Reference limits are estimated only from the chronological TRAINING period.
MIN_SPC_EVENTS = 20
MIN_REFERENCE_EVENTS = 10
# Backward-compatible alias used by older notebooks.
MIN_PHASE1_EVENTS = MIN_REFERENCE_EVENTS
BASELINE_ABNORMAL_REVIEW_RATE = 0.10

# Forecast/classification horizon: fifth next event for the same machine + CTQ
FORECAST_HORIZON = 5
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
MIN_ARIMA_TRAIN_EVENTS = 25
MIN_ARIMA_TEST_EVENTS_FOR_STRONG_EVIDENCE = 10

STATE_ORDER = {"NORMAL": 0, "WARNING": 1, "ABNORMAL": 2}
STATE_LABELS = ("NORMAL", "WARNING", "ABNORMAL")

NELSON_RULES = {
    "N1": ("One point beyond 3 sigma", "Strong unusual point; investigate a special cause."),
    "N2": ("Nine points on the same side of the center line", "Possible sustained process shift."),
    "N3": ("Six points continuously increasing or decreasing", "Possible continuous process drift."),
    "N4": ("Fourteen points alternating up and down", "Possible oscillation or alternating influence."),
    "N5": ("Two of three points beyond 2 sigma on the same side", "Possible moderate process shift."),
    "N6": ("Four of five points beyond 1 sigma on the same side", "Possible small persistent process shift."),
    "N7": ("Fifteen points within 1 sigma", "Variation is unusually small; check stratification or measurement behaviour."),
    "N8": ("Eight points outside 1 sigma with points on both sides", "Possible mixture or other non-random behaviour."),
}

# The distributed project uses an anonymized event-level source dataset.
# No company-specific correction map is required for the default run.
PACKAGE_CORRECTIONS = {}
MACHINE_CORRECTIONS = {}


ARIMA_CANDIDATES = (
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
)

MLP_CANDIDATES = (
    {"hidden_layer_sizes": (16,), "alpha": 0.001},
    {"hidden_layer_sizes": (32, 16), "alpha": 0.001},
    {"hidden_layer_sizes": (32, 16), "alpha": 0.01},
)
