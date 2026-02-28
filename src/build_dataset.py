import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED
from src.features import make_features

def build() -> pd.DataFrame:
    path = DATA_RAW / "aqi.csv"
    df = pd.read_csv(path)

    # Normalize column names: strip + lower
    df.columns = [c.strip() for c in df.columns]

    # Map Kaggle columns -> our standard names
    rename_map = {
        "Date": "date",
        "City": "city",
        "AQI": "aqi",
        "PM2.5 (Âµg/mÂ³)": "pm25",
        "PM10 (Âµg/mÂ³)": "pm10",
        "Temperature (Â°C)": "temp",
        "Humidity (%)": "humidity",
        "Wind Speed (m/s)": "wind_speed",
    }
    # Some files may have slightly different encoding; handle alternatives too
    for col in list(df.columns):
        if "PM2.5" in col and col not in rename_map:
            rename_map[col] = "pm25"
        if "PM10" in col and col not in rename_map:
            rename_map[col] = "pm10"
        if "Temperature" in col and col not in rename_map:
            rename_map[col] = "temp"
        if "Humidity" in col and col not in rename_map:
            rename_map[col] = "humidity"
        if "Wind Speed" in col and col not in rename_map:
            rename_map[col] = "wind_speed"

    df = df.rename(columns=rename_map)

    # Keep only needed columns (others can stay, but we standardize core ones)
    needed = ["date", "city", "aqi", "pm25", "pm10", "temp", "humidity", "wind_speed"]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[needed].dropna(subset=["date", "city", "aqi"])

    # Add a simple "rain" column (not present). Keep 0 for now.
    df["rain"] = 0.0

    # Feature engineering (festival/winter/weekend/traffic/construction)
    df = make_features(df)

    # Target: next-day AQI per city
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["city", "date"])
    df["target_aqi_next_day"] = df.groupby("city")["aqi"].shift(-1)
    df = df.dropna(subset=["target_aqi_next_day"]).reset_index(drop=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "hybrid_dataset.csv"
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("Rows:", len(df), "Cols:", len(df.columns))
    return df

if __name__ == "__main__":
    build()