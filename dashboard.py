import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🚀 XGBoost M&V Dashboard")
st.markdown("*AI-based Measurement & Verification using XGBoost Regressor*")

# Load model with pickle
@st.cache_resource
def load_model():
    model_path = 'models/xgboost_model.pkl'
    features_path = 'models/xgboost_features.txt'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found!")
        st.info("Please make sure 'xgboost_model.pkl' is in the 'models' folder")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success("✅ XGBoost model loaded successfully!")

# Scaling factor for Malaysia context
SCALING_FACTOR = 15

# Sidebar
with st.sidebar:
    st.header("📋 Building Parameters")
    st.markdown("---")
    
    temp = st.slider("🌡️ Temperature (°C)", 22, 35, 28)
    humidity = st.slider("💧 Humidity (%)", 60, 95, 80)
    hour = st.slider("⏰ Hour of Day", 0, 23, 14)
    dayofweek = st.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], 
                             format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
    month = st.selectbox("📆 Month", list(range(1,13)), 
                         format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    floor_area = st.number_input("🏠 Floor Area (m²)", 50, 300, 120)
    occupants = st.number_input("👥 Number of Occupants", 1, 8, 4)
    retrofit = st.selectbox("🔧 Retrofit Status", [0,1], 
                           format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

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

# Main content
st.info("📌 **Malaysia Context:** Energy values scaled for residential homes | TNB tariff: RM0.52/kWh")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy (XGBoost)", type="primary", use_container_width=True):
        # Predict
        raw_pred = model.predict(X)[0]
        pred = raw_pred / SCALING_FACTOR
        
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{pred:.2f} kWh")
        
        if retrofit == 1:
            # Baseline (without retrofit)
            X_base = X.copy()
            X_base['retrofit'] = 0
            raw_base = model.predict(X_base)[0]
            base_pred = raw_base / SCALING_FACTOR
            
            savings = base_pred - pred
            savings_pct = (savings / base_pred) * 100
            
            m2.metric("💰 Energy Savings", f"{savings:.2f} kWh", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
            
            # Monthly bill savings
            monthly_savings = savings * 24 * 30
            monthly_rm = monthly_savings * 0.52
            st.success(f"💡 Retrofit Savings: {savings:.2f} kWh ({savings_pct:.1f}%)")
            st.info(f"💰 Estimated Monthly Bill Savings: RM {monthly_rm:.2f}")
            
        else:
            # Not retrofitted - show potential
            X_retro = X.copy()
            X_retro['retrofit'] = 1
            raw_retro = model.predict(X_retro)[0]
            retro_pred = raw_retro / SCALING_FACTOR
            
            potential = pred - retro_pred
            potential_pct = (potential / pred) * 100
            
            m2.metric("💰 Potential Savings", f"{potential:.2f} kWh", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 If you retrofit this building: Would save ~{potential:.2f} kWh ({potential_pct:.1f}%)")
            st.caption("👉 Tip: Select 'Yes (Retrofitted)' above to see detailed savings analysis!")

with col2:
    st.markdown("""
    ### 📖 About XGBoost M&V System
    
    | Item | Details |
    |------|---------|
    | **Model** | XGBoost Regressor |
    | **Features** | Temperature, Humidity, Hour, Day, Month, Floor Area, Occupants, Retrofit |
    | **Scaling** | Malaysia residential context |
    | **Tariff** | TNB RM0.52/kWh |
    
    ### 💡 How to Use
    1. Adjust building parameters in sidebar
    2. Click "Predict Energy (XGBoost)"
    3. See potential savings from retrofit
    """)

st.markdown("---")
st.caption("🚀 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
