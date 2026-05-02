import pandas as pd
from pathlib import Path

COLUMNS= (["unit_id","cycle"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)])

def load_dataset(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = Path("data/raw")

    train = pd.read_csv(raw / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(raw / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul_test = pd.read_csv(raw / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])["RUL"]

    return train, test, rul_test

def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_cycle"]= df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns="max_cycle")
    return df

def save(train: pd.DataFrame, test: pd.DataFrame) -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train.to_csv("data/processed/train_processed.csv", index=False)
    test.to_csv("data/processed/test_processed.csv", index=False)

if __name__ == "__main__":
    train, test, rul_test = load_dataset("FD001")
    train = compute_train_rul(train)
    save(train,test)

    print(f"Train : {len(train)} lignes, {train['unit_id'].nunique()} moteurs")
    print(f"Test : {len(test)} lignes, {test['unit_id'].nunique()} moteurs")
    print(f"RUL test - min :{rul_test.min()}, max: {rul_test.max()}, moyenne: {rul_test.mean():.2f} cycles")
