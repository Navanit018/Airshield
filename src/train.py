import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import DATA_PROCESSED, MODELS

# Features available from build_dataset + features.py
FEATURES = [
    "aqi", "pm25", "pm10", "temp", "humidity", "wind_speed", "rain",
    "dow", "month", "is_weekend", "is_winter",
    "festival_season", "school_day", "traffic_level", "construction_activity"
]

def train():
    df = pd.read_csv(DATA_PROCESSED / "hybrid_dataset.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure all feature cols exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0

    df[FEATURES] = df[FEATURES].fillna(0)

    # Target
    if "target_aqi_next_day" not in df.columns:
        raise ValueError("target_aqi_next_day column missing. Re-run: python -m src.build_dataset")

    X = df[FEATURES]
    y = df["target_aqi_next_day"]

    # Model
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        random_state=42
    )

    # Time-series CV (optional but useful)
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        maes.append(mean_absolute_error(yte, pred))

    # Fit final model on all data
    model.fit(X, y)

    # Save
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "aqi_model.joblib")
    joblib.dump(FEATURES, MODELS / "features.joblib")

    print("✅ Model saved:", MODELS / "aqi_model.joblib")
    print("✅ Features saved:", MODELS / "features.joblib")
    print("✅ CV MAE (avg):", sum(maes) / len(maes))

if __name__ == "__main__":
    train()