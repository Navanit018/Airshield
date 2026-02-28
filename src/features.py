import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    return df

def add_festival_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple proxy: Oct-Nov as festival-heavy months (edit later for Nepal-specific)
    df["festival_season"] = df["month"].isin([10, 11]).astype(int)
    return df

def add_school_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Proxy: Mon-Fri school days
    df["school_day"] = (1 - df["is_weekend"]).astype(int)
    return df

def add_traffic_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Daily traffic proxy: weekdays higher + festival adds some extra movement
    df["traffic_level"] = (2 * df["school_day"] + 1 * df["festival_season"]).clip(0, 3)
    return df

def add_construction_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Placeholder (0-3). Improve later if you add real construction data.
    df["construction_activity"] = 2
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_festival_feature(df)
    df = add_school_day_feature(df)
    df = add_traffic_feature(df)
    df = add_construction_feature(df)
    return df