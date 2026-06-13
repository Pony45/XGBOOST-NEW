import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Try joblib, fallback to pickle if not available
try:
    import joblib
    print("Using joblib")
except ImportError:
    import pickle as joblib
    print("Using pickle as fallback")

# Page config
st.set_page_config(
    page_title="XGBoost M&V Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .savings-high {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
    }
    .savings-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
    }
    .savings-low {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">🚀 XGBoost M&V Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-based Measurement & Verification using XGBoost Regressor</p>', unsafe_allow_html=True)

# ==========================================
# LOAD XGBOOST MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Try multiple possible paths
    possible_paths = [
        'models/xgboost_model.pkl',
        'xgboost_model.pkl',
        'models/model.pkl',
        'model.pkl'
    ]
    
    model = None
    model_path_used = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_path_used = path
                break
            except:
                continue
    
    if model is None:
        st.error("❌ XGBoost model not found!")
        st.info("""
        **Troubleshooting:**
        1. Make sure 'xgboost_model.pkl' is in the 'models' folder
        2. Check that the file was uploaded correctly to GitHub
        """)
        
        # List files for debugging
        st.write("Files in current directory:")
        for f in os.listdir('.'):
            st.write(f"  - {f}")
        
        if os.path.exists('models'):
            st.write("Files in 'models' folder:")
            for f in os.listdir('models'):
                st.write(f"  - {f}")
        
        return None, None
    
    # Load features
    features_paths = [
        'models/xgboost_features.txt',
        'xgboost_features.txt',
        'models/features.txt',
        'features.txt'
    ]
    
    features = None
    for path in features_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            break
    
    if features is None:
        # Default features
        features = [
            'temperature', 'humidity', 'hour', 'dayofweek', 'month',
            'floor_area', 'occupants', 'retrofit',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_weekend', 'temp_humidity', 'occ_per_area'
        ]
    
    st.success(f"✅ XGBoost model loaded from {model_path_used}")
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

# ==========================================
# LOAD METRICS
# ==========================================
@st.cache_resource
def load_metrics():
    metrics_paths = [
        'models/xgboost_metrics.json',
        'xgboost_metrics.json',
        'models/metrics.json',
        'metrics.json'
    ]
    
    for path in metrics_paths:
        if os.path.exists(path):
            try:
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                continue
    return None

metrics = load_metrics()

# ==========================================
# SCALING FACTOR
# ==========================================
SCALING_FACTOR = 15

def scale_prediction(prediction):
    return prediction / SCALING_FACTOR

# ==========================================
# UNIT CONVERSION
# ==========================================
def convert_energy_unit(prediction_kwh, target_unit):
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 📋 Building Parameters")
    st.markdown("---")
    
    # Display Settings
    st.markdown("### ⚙️ Display Settings")
    unit_option = st.selectbox(
        "Energy Unit Display",
        ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"]
    )
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("### 📊 XGBoost Performance")
    
    if metrics:
        with st.expander("Performance Metrics", expanded=True):
            col_r2, col_mae = st.columns(2)
            col_r2.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
            col_mae.metric("MAE", f"{metrics.get('mae', 0):.2f} kWh")
            if 'rmse' in metrics:
                st.caption(f"RMSE: {metrics['rmse']:.2f} kWh")
            st.progress(metrics.get('r2_score', 0), text=f"Accuracy: {metrics.get('r2_score', 0)*100:.1f}%")
    else:
        with st.expander("Performance Metrics", expanded=True):
            st.info("Metrics file not found - model still works!")
    
    st.markdown("---")
    
    # Input Parameters
    st.markdown("### 🏠 Building Characteristics")
    
    temp = st.slider("🌡️ Temperature (°C)", 22, 35, 28)
    humidity = st.slider("💧 Humidity (%)", 60, 95, 80)
    hour = st.slider("⏰ Hour of Day", 0, 23, 14)
    
    col_dow, col_month = st.columns(2)
    with col_dow:
        dayofweek = st.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], 
                                 format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
    with col_month:
        month = st.selectbox("📆 Month", list(range(1,13)), 
                             format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    
    floor_area = st.number_input("🏠 Floor Area (m²)", 50, 300, 120)
    occupants = st.number_input("👥 Number of Occupants", 1, 8, 4)
    retrofit = st.selectbox("🔧 Retrofit Status", [0,1], 
                           format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# ==========================================
# MAIN CONTENT
# ==========================================
st.info("📌 **Malaysia Context:** Energy values scaled for residential homes | TNB tariff: RM0.52/kWh")

tab1, tab2, tab3 = st.tabs(["📊 Energy Prediction", "💰 Savings Analysis", "📈 Feature Impact"])

# ==========================================
# TAB 1: Energy Prediction
# ==========================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Energy (XGBoost)", type="primary", use_container_width=True):
            raw_pred = model.predict(features_df)[0]
            pred = scale_prediction(raw_pred)
            converted, unit = convert_energy_unit(pred, unit_option)
            
            st.subheader("📊 Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
            
            if retrofit == 1:
                base_df = features_df.copy()
                base_df['retrofit'] = 0
                raw_base = model.predict(base_df)[0]
                base_pred = scale_prediction(raw_base)
                
                savings = base_pred - pred
                savings_pct = (savings / base_pred) * 100
                savings_conv, _ = convert_energy_unit(savings, unit_option)
                
                res_col2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
                res_col3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
                
                if savings_pct >= 20:
                    st.markdown(f'<div class="savings-high">💡 <strong>Excellent!</strong> XGBoost predicts {savings_pct:.1f}% savings.</div>', unsafe_allow_html=True)
                elif savings_pct >= 10:
                    st.markdown(f'<div class="savings-medium">💡 <strong>Good!</strong> XGBoost predicts {savings_pct:.1f}% savings.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="savings-low">💡 <strong>Moderate</strong> XGBoost predicts {savings_pct:.1f}% savings.</div>', unsafe_allow_html=True)
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(8,5))
                ax.bar(['Baseline', 'Retrofitted'], [base_pred, pred], 
                       color=['#e74c3c', '#2ecc71'], edgecolor='black')
                ax.set_ylabel(f'Energy ({unit})')
                ax.set_title('XGBoost: Retrofit Impact')
                st.pyplot(fig)
                
            else:
                retro_df = features_df.copy()
                retro_df['retrofit'] = 1
                raw_retro = model.predict(retro_df)[0]
                retro_pred = scale_prediction(raw_retro)
                potential = pred - retro_pred
                potential_pct = (potential / pred) * 100
                
                res_col2.metric("💰 Potential Savings", f"{potential:.2f} kWh", delta=f"{potential_pct:.1f}%")
                res_col3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
                
                st.info(f"💡 If retrofitted: Save ~{potential:.2f} kWh ({potential_pct:.1f}%)")
                st.caption("👉 Select 'Yes' above to see detailed analysis")
    
    with col2:
        st.markdown("""
        ### 📖 About XGBoost
        
        | Item | Details |
        |------|---------|
        | **Model** | XGBoost Regressor |
        | **Features** | 15 parameters |
        | **Scaling** | Malaysia residential |
        
        **vs Random Forest:**
        - Sequential learning
        - Usually higher accuracy
        """)

# ==========================================
# TAB 2: Savings Analysis
# ==========================================
with tab2:
    st.subheader("💰 Savings Analysis")
    
    base_df = features_df.copy()
    base_df['retrofit'] = 0
    retro_df = features_df.copy()
    retro_df['retrofit'] = 1
    
    base_pred = scale_prediction(model.predict(base_df)[0])
    retro_pred = scale_prediction(model.predict(retro_df)[0])
    
    savings = base_pred - retro_pred
    savings_pct = (savings / base_pred) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline", f"{base_pred:.2f} kWh")
    c2.metric("Retrofitted", f"{retro_pred:.2f} kWh", delta=f"-{savings:.2f}")
    c3.metric("Savings", f"{savings:.2f} kWh", delta=f"{savings_pct:.1f}%")
    
    # Monthly bill
    monthly_rm = savings * 24 * 30 * 0.52
    st.info(f"💰 Monthly Bill Savings: RM {monthly_rm:.2f}")

# ==========================================
# TAB 3: Feature Impact
# ==========================================
with tab3:
    st.subheader("🔍 Feature Impact")
    
    feature = st.selectbox("Select feature", ['temperature', 'humidity', 'hour'])
    
    if feature == 'temperature':
        x_range = np.arange(22, 36, 1)
        x_label = "Temperature (°C)"
    elif feature == 'humidity':
        x_range = np.arange(60, 96, 5)
        x_label = "Humidity (%)"
    else:
        x_range = np.arange(0, 24, 1)
        x_label = "Hour"
    
    pred_base = []
    pred_retro = []
    
    for val in x_range:
        if feature == 'temperature':
            temp_val = val
            hum_val = 80
            hour_val = 14
        elif feature == 'humidity':
            temp_val = 28
            hum_val = val
            hour_val = 14
        else:
            temp_val = 28
            hum_val = 80
            hour_val = val
        
        h_sin = np.sin(2 * np.pi * hour_val / 24)
        h_cos = np.cos(2 * np.pi * hour_val / 24)
        t_h = temp_val * hum_val / 100
        
        # Baseline
        feat = [[temp_val, hum_val, hour_val, 0, 6, 120, 4, 0, h_sin, h_cos, 0, 0, 0, t_h, 0.033]]
        pred_base.append(scale_prediction(model.predict(pd.DataFrame(feat, columns=FEATURES))[0]))
        
        # Retrofit
        feat[0][7] = 1
        pred_retro.append(scale_prediction(model.predict(pd.DataFrame(feat, columns=FEATURES))[0]))
    
    fig, ax = plt.subplots()
    ax.plot(x_range, pred_base, 'o-', label='Baseline', color='red')
    ax.plot(x_range, pred_retro, 's-', label='Retrofitted', color='green')
    ax.fill_between(x_range, pred_base, pred_retro, alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Energy (kWh)')
    ax.legend()
    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("🚀 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
