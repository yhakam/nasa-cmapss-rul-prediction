import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")

VAL_SIZE = 20
RANDOM_STATE = 42


def load_data(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    rul_test = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", header=None, names=["RUL"])["RUL"]
    return train, test, rul_test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("sensor_")]


def split_by_unit(df: pd.DataFrame, val_size: int = VAL_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_units = df["unit_id"].unique()
    val_units = np.random.RandomState(RANDOM_STATE).choice(all_units, size=val_size, replace=False)

    train_df = df[~df["unit_id"].isin(val_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()

    return train_df, val_df

#Message à moi même : Review à partir d'ici

def get_last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = y_pred - y_true
    score = np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1)
    return float(np.sum(score))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float]:
    y_pred = np.maximum(y_pred, 0)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    score = nasa_score(y_true, y_pred)

    print(f"\n{label}")
    print(f"  RMSE       : {rmse:.2f}")
    print(f"  MAE        : {mae:.2f}")
    print(f"  Score NASA : {score:.2f}")

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "nasa_score": round(score, 2),
    }


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge()
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def add_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["predicted_RUL"] <= 30,
        df["predicted_RUL"] <= 60,
    ]

    choices = ["HIGH", "MEDIUM"]

    df["risk_level"] = np.select(conditions, choices, default="LOW")

    return df


def build_prediction_dataframe(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    predictions = pd.DataFrame({
        "unit_id": df["unit_id"].values,
        "last_cycle": df["cycle"].values,
        "true_RUL": y_true,
        "predicted_RUL": np.maximum(y_pred, 0),
    })

    predictions["absolute_error"] = (predictions["true_RUL"] - predictions["predicted_RUL"]).abs()
    predictions = add_risk_level(predictions)

    return predictions


def save_artifacts(
    rf_model: RandomForestRegressor,
    baseline_model: Ridge,
    scaler: StandardScaler,
    feature_cols: list[str],
    metrics: dict,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, MODELS_DIR / "random_forest_rul.joblib")
    joblib.dump(baseline_model, MODELS_DIR / "ridge_baseline.joblib")
    joblib.dump(scaler, MODELS_DIR / "ridge_scaler.joblib")

    with open(MODELS_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtefacts sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    train, test, rul_test = load_data("FD001")
    feature_cols = get_feature_cols(train)

    train_df, val_df = split_by_unit(train)

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    X_val = val_df[feature_cols]
    y_val = val_df["RUL"]

    val_last = get_last_cycle_per_unit(val_df)
    X_val_last = val_last[feature_cols]
    y_val_last = val_last["RUL"]

    print(f"Train : {len(train_df)} lignes, {train_df['unit_id'].nunique()} moteurs")
    print(f"Val   : {len(val_df)} lignes, {val_df['unit_id'].nunique()} moteurs")
    print(f"Features utilisées : {len(feature_cols)}")

    baseline_model, scaler = train_baseline(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    baseline_val_preds = baseline_model.predict(scaler.transform(X_val))
    baseline_last_preds = baseline_model.predict(scaler.transform(X_val_last))

    rf_val_preds = rf_model.predict(X_val)
    rf_last_preds = rf_model.predict(X_val_last)

    metrics = {
        "validation_all_cycles": {
            "ridge_baseline": evaluate(y_val.values, baseline_val_preds, "Ridge Baseline — Validation all cycles"),
            "random_forest": evaluate(y_val.values, rf_val_preds, "Random Forest — Validation all cycles"),
        },
        "validation_last_cycle": {
            "ridge_baseline": evaluate(y_val_last.values, baseline_last_preds, "Ridge Baseline — Validation last cycle"),
            "random_forest": evaluate(y_val_last.values, rf_last_preds, "Random Forest — Validation last cycle"),
        },
    }

    val_predictions = build_prediction_dataframe(val_last, y_val_last.values, rf_last_preds)
    val_predictions.to_csv(PROCESSED_DIR / "validation_predictions.csv", index=False)

    X_full = train[feature_cols]
    y_full = train["RUL"]

    final_rf_model = train_random_forest(X_full, y_full)

    test_last = get_last_cycle_per_unit(test)
    X_test_last = test_last[feature_cols]
    test_preds = final_rf_model.predict(X_test_last)

    test_predictions = build_prediction_dataframe(test_last, rul_test.values, test_preds)
    test_predictions.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)

    metrics["test_last_cycle"] = {
        "random_forest": evaluate(
            test_predictions["true_RUL"].values,
            test_predictions["predicted_RUL"].values,
            "Random Forest — NASA test last cycle",
        )
    }

    save_artifacts(final_rf_model, baseline_model, scaler, feature_cols, metrics)

    print("\nFichiers générés :")
    print(f"- {PROCESSED_DIR / 'validation_predictions.csv'}")
    print(f"- {PROCESSED_DIR / 'test_predictions.csv'}")
    print(f"- {MODELS_DIR / 'random_forest_rul.joblib'}")