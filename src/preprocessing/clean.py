import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
RUL_CAP = 150 
STD_THRESHOLD = 0.5

def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DIR/"train_processed.csv")
    test = pd.read_csv(PROCESSED_DIR/"test_processed.csv")
    return train, test

def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("sensor_")]

def identify_low_variance_sensors(train: pd.DataFrame,threshold= STD_THRESHOLD) -> list[str]:
    sensor_cols = get_sensor_columns(train)
    sensor_std = train[sensor_cols].std()
    low_variance_sensors = sensor_std[sensor_std < threshold].index.tolist()
    return low_variance_sensors

def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    df = df.copy()
    existing_columns= [col for col in columns_to_drop if col in df.columns]
    df= df.drop(columns=existing_columns)
    return df

def cap_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df = df.copy()
    if "RUL" not in df.columns:
        raise ValueError("Column 'RUL' not found in the DataFrame")
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df

def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    low_variance_sensors = identify_low_variance_sensors(train)
    train = drop_columns(train, low_variance_sensors)
    test = drop_columns(test, low_variance_sensors)
    train = cap_rul(train)
    return train, test, low_variance_sensors

def save_clean_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR/"train_clean.csv",index=False)
    test.to_csv(PROCESSED_DIR/ "test_clean.csv",index=False)

if __name__ == "__main__":
    train, test = load_processed_data()
    train_clean, test_clean, dropped_sensors =  preprocess(train,test)
    save_clean_data(train_clean,test_clean) 
    
    print(f"Capteurs supprimés ({len(dropped_sensors)}) : {dropped_sensors}")
    print(f"Train : {len(train_clean)} lignes, {train_clean.shape[1]} colonnes")
    print(f"Test  : {len(test_clean)} lignes, {test_clean.shape[1]} colonnes")
    print(f"RUL max après cap : {train_clean['RUL'].max()}")