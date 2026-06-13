# ==========================================
# TRAIN XGBOOST & SAVE PROPERLY
# ==========================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import json
from google.colab import files

print("="*60)
print("🚀 TRAINING XGBOOST")
print("="*60)

# Upload energy_data_full.csv
print("\n📁 Upload energy_data_full.csv")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print(f"✅ Loaded: {df.shape}")

# Features
features = [
    'temperature', 'humidity', 'hour', 'dayofweek', 'month',
    'floor_area', 'occupants', 'retrofit',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'temp_humidity', 'occ_per_area'
]

# Create engineered features if not exist
if 'hour_sin' not in df.columns:
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
    df['occ_per_area'] = df['occupants'] / df['floor_area']

X = df[features]
y = df['energy_consumption']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
print("\n🚀 Training XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ R²: {r2:.4f}")
print(f"✅ MAE: {mae:.4f}")

# Save model using pickle (simpler than joblib)
os.makedirs('models', exist_ok=True)

# Save with pickle
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved with pickle")

# Save features
with open('models/xgboost_features.txt', 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")

# Save metrics
metrics = {'r2_score': float(r2), 'mae': float(mae), 'rmse': 0}
with open('models/xgboost_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Create zip
!zip -r xgboost_models.zip models/
files.download('xgboost_models.zip')

print("\n✅ xgboost_models.zip downloaded!")
