import sqlite3
import json
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
DATABASE_DIR = Path("database")
DATABASE_PATH = DATABASE_DIR / "cmapss.db"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def load_predictions(conn: sqlite3.Connection) -> None:
    path = PROCESSED_DIR / "test_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "true_RUL": "true_rul",
            "predicted_RUL": "predicted_rul",
        }
    )

    df.to_sql("predictions", conn, if_exists="replace", index=False)

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unit_id
        ON predictions(unit_id);
        """
    )

    print(f"predictions : {len(df)} lignes chargées")


def load_metrics(conn: sqlite3.Connection) -> None:
    path = MODELS_DIR / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = []

    for evaluation, models in metrics.items():
        if not isinstance(models, dict):
            continue

        for model, values in models.items():
            if not isinstance(values, dict):
                continue

            rows.append(
                {
                    "model": model,
                    "evaluation": evaluation,
                    "rmse": values.get("rmse"),
                    "mae": values.get("mae"),
                    "nasa_score": values.get("nasa_score"),
                }
            )

    conn.execute("DROP TABLE IF EXISTS metrics;")

    conn.execute(
        """
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            evaluation TEXT NOT NULL,
            rmse REAL,
            mae REAL,
            nasa_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    df = pd.DataFrame(rows)
    df.to_sql("metrics", conn, if_exists="append", index=False)

    print(f"metrics : {len(df)} lignes chargées")


def load_sensor_readings(conn: sqlite3.Connection) -> None:
    path = PROCESSED_DIR / "test_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    df.to_sql("sensor_readings", conn, if_exists="replace", index=False)

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sensor_readings_unit_cycle
        ON sensor_readings(unit_id, cycle);
        """
    )

    print(f"sensor_readings : {len(df)} lignes chargées")


def build_database() -> None:
    conn = get_connection()

    try:
        load_predictions(conn)
        load_metrics(conn)
        load_sensor_readings(conn)
        conn.commit()
    finally:
        conn.close()

    print(f"\nBase créée : {DATABASE_PATH}")


if __name__ == "__main__":
    build_database()