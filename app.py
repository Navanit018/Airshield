import streamlit as st
import pandas as pd
from src.config import DATA_PROCESSED, CITY_DEFAULT
from src.predict import predict_next_day
from src.policy import management_actions

st.title("🌫 AirShield AI — AQI Prediction & Management Dashboard")

dataset_path = DATA_PROCESSED / "hybrid_dataset.csv"
if not dataset_path.exists():
    st.warning("Build dataset first: python -m src.build_dataset")
    st.stop()

df = pd.read_csv(dataset_path)
df["date"] = pd.to_datetime(df["date"])

city = st.selectbox("City", sorted(df["city"].unique()), index=0 if CITY_DEFAULT not in df["city"].unique() else sorted(df["city"].unique()).index(CITY_DEFAULT))
city_df = df[df["city"] == city].sort_values("date")

st.subheader("📈 Historical AQI (proxy)")
st.line_chart(city_df.set_index("date")["aqi"])

st.subheader("🔮 Next-day AQI prediction")
latest = city_df.iloc[-1].to_dict()

# Use the latest features row
pred = predict_next_day(latest)
cat, actions = management_actions(pred)

st.metric("Predicted AQI (next day)", f"{pred:.1f}", help="Model prediction using hybrid engineered dataset")
st.write("Category:", f"**{cat}**")

st.subheader("🛠 Recommended Management Actions")
for a in actions:
    st.write("•", a)

st.subheader("🧾 Latest feature row used for prediction")
show_cols = ["date","aqi","temp","humidity","wind_speed","rain","traffic_level","construction_activity","festival_season","school_day","is_winter"]
st.dataframe(city_df[show_cols].tail(5))