import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
WINDOW = 5 # Choix initiale de fenêtre de lissage des capteurs, compromis entre réduction du bruit et sensibilité aux changements récents


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("sensor_") and 'rollmean' not in col and 'delta' not in col]


def build_features(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    df = _add_rolling_mean(df)
    df = _add_delta(df)
    return df

def _add_rolling_mean(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = get_sensor_columns(df)
    for sensor in sensor_cols :
        df[f"{sensor}_rollmean_{WINDOW}"] = (df.groupby("unit_id")[sensor].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean()))
    return df

def _add_delta(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = get_sensor_columns(df)
    for sensor in sensor_cols :
        df[f"{sensor}_delta"] = (df.groupby("unit_id")[sensor].transform(lambda x: x.diff().fillna(0)))
    return df

def save_features(train: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_features.csv", index=False)

if __name__ == "__main__":
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    train = build_features(train)
    test = build_features(test)
    save_features(train, test)
    new_features = [col for col in train.columns if 'rollmean' in col or 'delta' in col]
    print(f"Train : {len(train)} lignes, {train.shape[1]} colonnes")
    print(f"Test : {len(test)} lignes, {test.shape[1]} colonnes")
    print(f"Nouvelles features: {new_features}")