import pandas as pd
import joblib
from src.config import MODELS

def predict_next_day(latest_row: dict) -> float:
    model = joblib.load(MODELS / "aqi_model.joblib")
    features = joblib.load(MODELS / "features.joblib")

    X = pd.DataFrame([latest_row])
    for c in features:
        if c not in X.columns:
            X[c] = 0
    X = X[features].fillna(0)

    return float(model.predict(X)[0])