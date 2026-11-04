from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from statsmodels.tsa.arima.model import ARIMA

from config import (
    ARIMA_CANDIDATES,
    FORECAST_HORIZON,
    MIN_ARIMA_TEST_EVENTS_FOR_STRONG_EVIDENCE,
    MIN_ARIMA_TRAIN_EVENTS,
    MLP_CANDIDATES,
    RANDOM_SEED,
    STATE_LABELS,
    STATE_ORDER,
)
from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def _classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probabilities: np.ndarray | None = None,
    probability_classes: list[str] | None = None,
) -> dict[str, float]:
    y_true = actual.astype(str)
    y_pred = predicted.astype(str)
    recalls = recall_score(y_true, y_pred, labels=list(STATE_LABELS), average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=list(STATE_LABELS), average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=list(STATE_LABELS), average=None, zero_division=0)

    metrics = {
        "ACCURACY": float(accuracy_score(y_true, y_pred)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y_true, y_pred)),
        "MACRO_PRECISION": float(precision_score(y_true, y_pred, labels=list(STATE_LABELS), average="macro", zero_division=0)),
        "MACRO_RECALL": float(recall_score(y_true, y_pred, labels=list(STATE_LABELS), average="macro", zero_division=0)),
        "MACRO_F1": float(f1_score(y_true, y_pred, labels=list(STATE_LABELS), average="macro", zero_division=0)),
        "PRECISION_NORMAL": float(precisions[0]),
        "PRECISION_WARNING": float(precisions[1]),
        "PRECISION_ABNORMAL": float(precisions[2]),
        "RECALL_NORMAL": float(recalls[0]),
        "RECALL_WARNING": float(recalls[1]),
        "RECALL_ABNORMAL": float(recalls[2]),
        "F1_NORMAL": float(f1s[0]),
        "F1_WARNING": float(f1s[1]),
        "F1_ABNORMAL": float(f1s[2]),
        "ROC_AUC_OVR_MACRO": np.nan,
    }

    if probabilities is not None and probability_classes is not None:
        try:
            aligned = np.zeros((len(y_true), len(STATE_LABELS)), dtype=float)
            for j, cls in enumerate(probability_classes):
                if cls in STATE_LABELS:
                    aligned[:, list(STATE_LABELS).index(cls)] = probabilities[:, j]
            y_bin = label_binarize(y_true, classes=list(STATE_LABELS))
            if y_bin.shape[1] == len(STATE_LABELS):
                metrics["ROC_AUC_OVR_MACRO"] = float(
                    roc_auc_score(y_bin, aligned, average="macro", multi_class="ovr")
                )
        except Exception:
            pass
    return metrics


def _make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def _oversample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Randomly oversample minority classes in training data only."""
    rng = np.random.default_rng(RANDOM_SEED)
    labels, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    indices: list[np.ndarray] = []
    for label, count in zip(labels, counts):
        idx = np.flatnonzero(y == label)
        if count < target:
            idx = np.concatenate([idx, rng.choice(idx, size=target - count, replace=True)])
        indices.append(idx)
    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return x[all_idx], y[all_idx]


def _probability_frame(model: MLPClassifier, probabilities: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({f"P_{state}": np.zeros(len(probabilities)) for state in STATE_LABELS})
    for i, cls in enumerate(model.classes_):
        if cls in STATE_LABELS:
            out[f"P_{cls}"] = probabilities[:, i]
    return out


@dataclass
class MLPResult:
    validation: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion: pd.DataFrame
    class_report: pd.DataFrame
    latest: pd.DataFrame


def train_mlp(features: pd.DataFrame, labelled: pd.DataFrame, model_dir: Path) -> MLPResult:
    """Three-class MLP: NORMAL / WARNING / ABNORMAL at the fifth next event."""
    train = labelled[labelled["SPLIT"].eq("TRAIN")].copy()
    validation = labelled[labelled["SPLIT"].eq("VALIDATION")].copy()
    test = labelled[labelled["SPLIT"].eq("TEST")].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Train, validation and test splits must all contain rows.")

    prep = _make_preprocessor()
    x_train = prep.fit_transform(train[MODEL_FEATURES])
    x_val = prep.transform(validation[MODEL_FEATURES])
    y_train = train["TARGET_FUTURE_STATE"].astype(str).to_numpy()
    y_val = validation["TARGET_FUTURE_STATE"].astype(str).to_numpy()
    x_bal, y_bal = _oversample(np.asarray(x_train), y_train)

    validation_rows: list[dict[str, object]] = []
    best_score = -np.inf
    best_params: dict[str, object] | None = None

    for params in MLP_CANDIDATES:
        model = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"], alpha=params["alpha"],
            activation="relu", solver="adam", learning_rate_init=0.001,
            max_iter=1200, early_stopping=False, random_state=RANDOM_SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_bal, y_bal)
        pred = model.predict(x_val)
        score = f1_score(y_val, pred, labels=list(STATE_LABELS), average="macro", zero_division=0)
        validation_rows.append({
            "HIDDEN_LAYERS": str(params["hidden_layer_sizes"]),
            "ALPHA": params["alpha"],
            "VALIDATION_MACRO_F1": float(score),
        })
        if score > best_score:
            best_score = float(score)
            best_params = params

    assert best_params is not None

    # Refit selected architecture using train + validation. Test remains untouched.
    development = pd.concat([train, validation], ignore_index=True).sort_values("DATE_TIME")
    final_prep = _make_preprocessor()
    x_dev = final_prep.fit_transform(development[MODEL_FEATURES])
    y_dev = development["TARGET_FUTURE_STATE"].astype(str).to_numpy()
    x_dev_bal, y_dev_bal = _oversample(np.asarray(x_dev), y_dev)

    selected_model = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"], alpha=best_params["alpha"],
        activation="relu", solver="adam", learning_rate_init=0.001,
        max_iter=1500, early_stopping=False, random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected_model.fit(x_dev_bal, y_dev_bal)

    x_test = final_prep.transform(test[MODEL_FEATURES])
    test_pred = selected_model.predict(x_test)
    test_prob = selected_model.predict_proba(x_test)
    prob_df = _probability_frame(selected_model, test_prob)

    predictions = test[[
        "EVENT_ID", "MACH_NO", "CTQ", "DEVICE", "DATE_TIME", "PROCESS_STATE",
        "TARGET_FUTURE_STATE", "TARGET_FUTURE_TIME",
    ]].reset_index(drop=True)
    predictions["MLP_PRED_STATE"] = test_pred
    predictions = pd.concat([predictions, prob_df], axis=1)

    metric = _classification_metrics(
        predictions["TARGET_FUTURE_STATE"], predictions["MLP_PRED_STATE"],
        test_prob, [str(x) for x in selected_model.classes_],
    )
    metrics = pd.DataFrame([{
        "MODEL": "MLP", "TEST_N": len(predictions),
        "SELECTED_HIDDEN_LAYERS": str(best_params["hidden_layer_sizes"]),
        "SELECTED_ALPHA": best_params["alpha"], **metric,
    }])

    cm = confusion_matrix(predictions["TARGET_FUTURE_STATE"], predictions["MLP_PRED_STATE"], labels=list(STATE_LABELS))
    confusion = (
        pd.DataFrame(cm, index=STATE_LABELS, columns=STATE_LABELS)
        .rename_axis("ACTUAL_STATE").reset_index()
        .melt(id_vars="ACTUAL_STATE", var_name="PREDICTED_STATE", value_name="COUNT")
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["TARGET_FUTURE_STATE"], predictions["MLP_PRED_STATE"],
        labels=list(STATE_LABELS), zero_division=0,
    )
    class_report = pd.DataFrame({
        "STATE": list(STATE_LABELS), "PRECISION": precision, "RECALL": recall,
        "F1": f1, "SUPPORT": support,
    })

    # Deployment refit after final test metrics are frozen.
    deploy_data = labelled[labelled["SPLIT"].ne("PURGED")].copy()
    deploy_prep = _make_preprocessor()
    x_all = deploy_prep.fit_transform(deploy_data[MODEL_FEATURES])
    y_all = deploy_data["TARGET_FUTURE_STATE"].astype(str).to_numpy()
    x_all_bal, y_all_bal = _oversample(np.asarray(x_all), y_all)
    deploy_model = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"], alpha=best_params["alpha"],
        activation="relu", solver="adam", learning_rate_init=0.001,
        max_iter=1500, early_stopping=False, random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deploy_model.fit(x_all_bal, y_all_bal)

    latest_rows = features.sort_values("DATE_TIME").groupby(["MACH_NO", "CTQ"], as_index=False).tail(1).copy().reset_index(drop=True)
    latest_x = deploy_prep.transform(latest_rows[MODEL_FEATURES])
    latest_pred = deploy_model.predict(latest_x)
    latest_prob = _probability_frame(deploy_model, deploy_model.predict_proba(latest_x))
    latest = latest_rows[["EVENT_ID", "MACH_NO", "CTQ", "DATE_TIME", "PROCESS_STATE"]].copy()
    latest["MLP_PRED_STATE"] = latest_pred
    latest = pd.concat([latest.reset_index(drop=True), latest_prob], axis=1)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": deploy_prep,
        "model": deploy_model,
        "features": MODEL_FEATURES,
        "forecast_horizon": FORECAST_HORIZON,
        "selected_params": best_params,
    }, model_dir / "mlp_future_state.joblib")

    return MLPResult(
        validation=pd.DataFrame(validation_rows).sort_values("VALIDATION_MACRO_F1", ascending=False),
        metrics=metrics,
        predictions=predictions,
        confusion=confusion,
        class_report=class_report,
        latest=latest,
    )


def _fit_arima(values: np.ndarray, order: tuple[int, int, int]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(values, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()


def _state_from_z(z: float, low: float | None = None, high: float | None = None) -> str:
    """Map the ARIMA point forecast to an SPC-style state. Intervals remain uncertainty evidence, not the class itself."""
    if not np.isfinite(z):
        return "NORMAL"
    if abs(z) > 3:
        return "ABNORMAL"
    if abs(z) > 2:
        return "WARNING"
    return "NORMAL"


def _arima_origin_predictions(
    group: pd.DataFrame,
    order: tuple[int, int, int],
    fit_end: pd.Timestamp,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Expanding-origin event+5 ARIMA forecasts for validation/test."""
    g = group.sort_values("DATE_TIME").reset_index(drop=True)
    initial = g[g["DATE_TIME"].le(fit_end)]
    if len(initial) < MIN_ARIMA_TRAIN_EVENTS:
        return pd.DataFrame()
    try:
        result = _fit_arima(initial["STANDARDIZED_DEVIATION"].to_numpy(float), order)
    except Exception:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    start_pos = len(initial)
    for pos in range(start_pos, len(g)):
        row = g.iloc[pos]
        try:
            result = result.append([float(row["STANDARDIZED_DEVIATION"])], refit=False)
        except Exception:
            continue

        in_window = row["DATE_TIME"] > origin_start and (origin_end is None or row["DATE_TIME"] <= origin_end)
        target_pos = pos + FORECAST_HORIZON
        if not in_window or target_pos >= len(g):
            continue
        target_row = g.iloc[target_pos]
        if origin_end is not None and target_row["DATE_TIME"] > origin_end:
            continue

        try:
            fc = result.get_forecast(steps=FORECAST_HORIZON)
            mean = np.asarray(fc.predicted_mean, dtype=float)
            ci95 = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        except Exception:
            continue

        records.append({
            "EVENT_ID": row["EVENT_ID"], "MACH_NO": row["MACH_NO"], "CTQ": row["CTQ"],
            "DATE_TIME": row["DATE_TIME"], "TARGET_FUTURE_STATE": target_row["PROCESS_STATE"],
            "TARGET_FUTURE_TIME": target_row["DATE_TIME"], "ARIMA_ORDER": str(order),
            "FORECAST_H5_Z": float(mean[-1]), "LOWER_95_H5": float(ci95[-1, 0]),
            "UPPER_95_H5": float(ci95[-1, 1]), "ACTUAL_H5_Z": float(target_row["STANDARDIZED_DEVIATION"]),
            "ARIMA_PRED_STATE": _state_from_z(float(mean[-1]), float(ci95[-1, 0]), float(ci95[-1, 1])),
            "NAIVE_H5_Z": float(row["STANDARDIZED_DEVIATION"]),
        })
    return pd.DataFrame(records)


@dataclass
class ARIMAResult:
    selection: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    forecast: pd.DataFrame
    latest: pd.DataFrame


def _validation_h5_rmse(
    group: pd.DataFrame,
    order: tuple[int, int, int],
    train_cut: pd.Timestamp,
    val_cut: pd.Timestamp,
) -> tuple[float, int]:
    """Select ARIMA order using the same event+5 objective used in final evaluation."""
    pred = _arima_origin_predictions(
        group=group,
        order=order,
        fit_end=train_cut,
        origin_start=train_cut,
        origin_end=val_cut,
    )
    if pred.empty:
        return np.inf, 0
    err = pred["ACTUAL_H5_Z"] - pred["FORECAST_H5_Z"]
    return float(np.sqrt(np.mean(err ** 2))), int(len(pred))


def train_arima(features: pd.DataFrame, split_meta: dict[str, object]) -> ARIMAResult:
    """ARIMA validation and event+5 test evaluation for each machine + CTQ stream."""
    train_cut = pd.Timestamp(split_meta["train_cutoff"])
    val_cut = pd.Timestamp(split_meta["validation_cutoff"])

    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    test_frames: list[pd.DataFrame] = []
    forecast_rows: list[dict[str, object]] = []
    latest_rows: list[dict[str, object]] = []

    for (machine, ctq), group in features.groupby(["MACH_NO", "CTQ"], sort=True):
        g = group.sort_values("DATE_TIME").copy()
        if len(g[g["DATE_TIME"].le(train_cut)]) < MIN_ARIMA_TRAIN_EVENTS:
            selection_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "ORDER": "NOT_AVAILABLE",
                "VALIDATION_RMSE": np.nan, "VALIDATION_N": 0,
                "STATUS": "Insufficient training history",
            })
            continue

        best_order: tuple[int, int, int] | None = None
        best_rmse = np.inf
        for order in ARIMA_CANDIDATES:
            rmse, n = _validation_h5_rmse(g, order, train_cut, val_cut)
            selection_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "ORDER": str(order),
                "VALIDATION_RMSE": rmse if np.isfinite(rmse) else np.nan,
                "VALIDATION_N": n, "STATUS": "Candidate",
            })
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse, best_order = rmse, order

        if best_order is None:
            best_order = (1, 0, 0)

        # Test uses expanding-origin +5 forecasts after model/order selection is frozen.
        test_pred = _arima_origin_predictions(g, best_order, val_cut, val_cut, None)
        if not test_pred.empty:
            err = test_pred["ACTUAL_H5_Z"] - test_pred["FORECAST_H5_Z"]
            naive_err = test_pred["ACTUAL_H5_Z"] - test_pred["NAIVE_H5_Z"]
            n_test = len(test_pred)
            metric_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "ORDER": str(best_order), "TEST_N": n_test,
                "H5_MAE": float(np.mean(np.abs(err))),
                "H5_MSE": float(np.mean(err ** 2)),
                "H5_RMSE": float(np.sqrt(np.mean(err ** 2))),
                "NAIVE_H5_RMSE": float(np.sqrt(np.mean(naive_err ** 2))),
                "H5_INTERVAL_COVERAGE_%": float(100 * np.mean(
                    (test_pred["ACTUAL_H5_Z"] >= test_pred["LOWER_95_H5"])
                    & (test_pred["ACTUAL_H5_Z"] <= test_pred["UPPER_95_H5"])
                )),
                "EVIDENCE_LEVEL": "ADEQUATE" if n_test >= MIN_ARIMA_TEST_EVENTS_FOR_STRONG_EVIDENCE else "LIMITED",
            })
            test_frames.append(test_pred)

        try:
            final = _fit_arima(g["STANDARDIZED_DEVIATION"].to_numpy(float), best_order)
            fc = final.get_forecast(steps=FORECAST_HORIZON)
            mean = np.asarray(fc.predicted_mean, dtype=float)
            ci95 = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
            future_state = _state_from_z(float(mean[-1]), float(ci95[-1, 0]), float(ci95[-1, 1]))
            for h in range(FORECAST_HORIZON):
                forecast_rows.append({
                    "MACH_NO": machine, "CTQ": ctq, "HORIZON_EVENT": h + 1,
                    "FORECAST_Z": float(mean[h]), "LOWER_95": float(ci95[h, 0]), "UPPER_95": float(ci95[h, 1]),
                    "ARIMA_ORDER": str(best_order), "LAST_OBSERVED_TIME": g.iloc[-1]["DATE_TIME"],
                })
            latest_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "ARIMA_PRED_STATE": future_state,
                "ARIMA_ORDER": str(best_order), "FORECAST_H5_Z": float(mean[-1]),
            })
        except Exception:
            latest_rows.append({
                "MACH_NO": machine, "CTQ": ctq, "ARIMA_PRED_STATE": pd.NA,
                "ARIMA_ORDER": str(best_order), "FORECAST_H5_Z": np.nan,
            })

    return ARIMAResult(
        selection=pd.DataFrame(selection_rows), metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(),
        forecast=pd.DataFrame(forecast_rows), latest=pd.DataFrame(latest_rows),
    )


def _more_severe(a: object, b: object) -> str:
    a_s = str(a) if pd.notna(a) else "NORMAL"
    b_s = str(b) if pd.notna(b) else "NORMAL"
    return a_s if STATE_ORDER.get(a_s, 0) >= STATE_ORDER.get(b_s, 0) else b_s


def evaluate_fork(mlp_predictions: pd.DataFrame, arima_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision-level fork integration and fair comparison on common test rows."""
    if mlp_predictions.empty or arima_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = mlp_predictions.merge(
        arima_predictions[["EVENT_ID", "ARIMA_PRED_STATE", "FORECAST_H5_Z", "ACTUAL_H5_Z"]],
        on="EVENT_ID", how="inner", validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["SPC_PERSISTENCE_STATE"] = merged["PROCESS_STATE"]
    merged["HYBRID_PRED_STATE"] = [
        _more_severe(a, b) for a, b in zip(merged["MLP_PRED_STATE"], merged["ARIMA_PRED_STATE"])
    ]

    rows: list[dict[str, object]] = []
    for name, col in [
        ("SPC Persistence", "SPC_PERSISTENCE_STATE"),
        ("ARIMA", "ARIMA_PRED_STATE"),
        ("MLP", "MLP_PRED_STATE"),
        ("Hybrid", "HYBRID_PRED_STATE"),
    ]:
        rows.append({
            "MODEL": name, "TEST_N": len(merged),
            **_classification_metrics(merged["TARGET_FUTURE_STATE"], merged[col]),
        })
    return pd.DataFrame(rows), merged


def build_latest_status(features: pd.DataFrame, mlp_latest: pd.DataFrame, arima_latest: pd.DataFrame) -> pd.DataFrame:
    current_columns = [
        "MACH_NO", "CTQ", "DATE_TIME", "PROCESS_STATE", "NELSON_RULE_COUNT",
        "NELSON_ANY", "NELSON_RULES_TRIGGERED", "NELSON_EPISODES_STARTED",
        "CAPABILITY_TYPE", "CAPABILITY_VALUE", "BASELINE_STATUS",
    ]
    current_columns = [c for c in current_columns if c in features.columns]
    current = (
        features.sort_values("DATE_TIME")
        .groupby(["MACH_NO", "CTQ"], as_index=False)
        .tail(1)[current_columns]
        .rename(columns={"PROCESS_STATE": "CURRENT_STATE", "DATE_TIME": "LATEST_TIME"})
    )
    out = current.merge(
        mlp_latest[["MACH_NO", "CTQ", "MLP_PRED_STATE", "P_NORMAL", "P_WARNING", "P_ABNORMAL"]],
        on=["MACH_NO", "CTQ"], how="left",
    )
    if not arima_latest.empty:
        out = out.merge(
            arima_latest[["MACH_NO", "CTQ", "ARIMA_PRED_STATE", "FORECAST_H5_Z"]],
            on=["MACH_NO", "CTQ"], how="left",
        )
    else:
        out["ARIMA_PRED_STATE"] = pd.NA
        out["FORECAST_H5_Z"] = np.nan

    out["HYBRID_DECISION"] = [_more_severe(a, b) for a, b in zip(out["MLP_PRED_STATE"], out["ARIMA_PRED_STATE"])]
    out["FUTURE_STATE"] = out["HYBRID_DECISION"]  # backward-compatible alias
    out["OVERALL_DECISION"] = [_more_severe(a, b) for a, b in zip(out["CURRENT_STATE"], out["HYBRID_DECISION"])]
    out["DECISION_PRIORITY"] = out["OVERALL_DECISION"].map({"NORMAL": "LOW", "WARNING": "MEDIUM", "ABNORMAL": "HIGH"}).fillna("REVIEW")

    def recommendation(row: pd.Series) -> str:
        strongest = str(row["OVERALL_DECISION"])
        if strongest == "ABNORMAL":
            return "Review recent SPC points, Nelson signals and break-mode pattern before the next run."
        if strongest == "WARNING":
            return "Review recent trend and lot history; continue closer engineering monitoring."
        return "Continue routine SPC monitoring."

    out["ENGINEERING_GUIDANCE"] = out.apply(recommendation, axis=1)
    return out.sort_values(["MACH_NO", "CTQ"]).reset_index(drop=True)
