import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🚀 XGBoost M&V Dashboard")

# Load model with pickle (no joblib needed)
@st.cache_resource
def load_model():
    model_path = 'models/xgboost_model.pkl'
    features_path = 'models/xgboost_features.txt'
    
    if not os.path.exists(model_path):
        st.error("Model not found")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success("✅ XGBoost loaded!")

# Sidebar
with st.sidebar:
    st.header("Parameters")
    temp = st.slider("Temperature (°C)", 22, 35, 28)
    humidity = st.slider("Humidity (%)", 60, 95, 80)
    hour = st.slider("Hour", 0, 23, 14)
    dayofweek = st.selectbox("Day", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month = st.selectbox("Month", list(range(1,13)))
    floor_area = st.number_input("Floor Area (m²)", 50, 300, 120)
    occupants = st.number_input("Occupants", 1, 8, 4)
    retrofit = st.selectbox("Retrofit", [0,1], format_func=lambda x: "Yes" if x else "No")

# Feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

X = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

if st.button("Predict", type="primary"):
    pred = model.predict(X)[0] / 15
    st.metric("Energy", f"{pred:.2f} kWh")
    
    if retrofit == 1:
        X_base = X.copy()
        X_base['retrofit'] = 0
        base = model.predict(X_base)[0] / 15
        savings = base - pred
        st.success(f"Savings: {savings:.2f} kWh ({savings/base*100:.1f}%)")
