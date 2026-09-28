import os
import time
import argparse
import warnings
import joblib
import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier

# Suppress scikit-learn feature name warnings for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning)

# Reuse the ANN architecture definition so we can load the weights
class IrrigationANN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(IrrigationANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_scaler(models_dir="models"):
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Did you run Phase 2?")
    return joblib.load(scaler_path)

def load_ann_model(models_dir="models"):
    model_path = os.path.join(models_dir, "smart_irrigation_ann.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ANN model not found at {model_path}. Did you run Phase 3?")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model = IrrigationANN(
        input_dim=checkpoint['input_dim'],
        hidden_dim1=checkpoint['hidden_dim1'],
        hidden_dim2=checkpoint['hidden_dim2'],
        dropout_rate=checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def load_xgb_model(models_dir="models"):
    model_path = os.path.join(models_dir, "smart_irrigation_xgb.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}. Did you run Phase 3?")
    return joblib.load(model_path)


def calculate_pump_runtime(moi_value, prediction):
    """
    Decision Engine Mapper:
    Class 0 -> 0 minutes
    Class 1 -> Dynamically scaled between 15 and 40 minutes based on Moisture (MOI).
    Assume critical low moisture = 10%, saturated threshold = 55%
    """
    if prediction == 0:
        return 0

    # We map MOI from [10%, 55%] inversely to [40 mins, 15 mins]
    min_moi, max_moi = 10.0, 55.0
    min_time, max_time = 15.0, 40.0
    
    # Clip MOI to our expected active range bounds
    clamped_moi = max(min_moi, min(moi_value, max_moi))
    
    # Linear interpolation (inverse)
    ratio = (clamped_moi - min_moi) / (max_moi - min_moi)
    runtime = max_time - (ratio * (max_time - min_time))
    
    return int(round(runtime))


def run_demo(moi, temp, humidity, use_model="ann", models_dir="models"):
    print("\n" + "="*55)
    print("      SMART IRRIGATION PREDICTIVE MODEL DEMO      ")
    print("="*55)
    print(f"[*] Input Sensor Readings:")
    print(f"    - Moisture (MOI) : {moi} %")
    print(f"    - Temperature    : {temp} °C")
    print(f"    - Humidity       : {humidity} %")
    print("-" * 55)

    # 1. Load Preprocessor
    t0 = time.time()
    scaler = load_scaler(models_dir)
    features = np.array([[moi, temp, humidity]])
    features_scaled = scaler.transform(features)
    load_time = (time.time() - t0) * 1000

    # 2. Load Model & Predict
    t_inf_start = time.time()
    if use_model.lower() == "xgb":
        print("[*] Active Engine: XGBoost Classifier")
        model = load_xgb_model(models_dir)
        prob = model.predict_proba(features_scaled)[0, 1]
        pred_class = 1 if prob >= 0.5 else 0
    else:
        print("[*] Active Engine: PyTorch Deep Neural Network")
        model = load_ann_model(models_dir)
        with torch.no_grad():
            tensor_in = torch.tensor(features_scaled, dtype=torch.float32)
            logit = model(tensor_in)
            prob = torch.sigmoid(logit).item()
            pred_class = 1 if prob >= 0.5 else 0
    
    inf_time = (time.time() - t_inf_start) * 1000

    # 3. Decision Engine
    pump_minutes = calculate_pump_runtime(moi, pred_class)

    # 4. Format Output
    decision_text = "IRRIGATION REQUIRED (Class 1)" if pred_class == 1 else "NO IRRIGATION (Class 0)"
    action_text = f"Run pump for {pump_minutes} minutes." if pred_class == 1 else "Pump remains OFF (0 mins)."
    confidence_pct = prob * 100 if pred_class == 1 else (1 - prob) * 100

    print("\n[RESULT]")
    print(f"  Decision   : {decision_text}")
    print(f"  Action     : {action_text}")
    print(f"  Confidence : {confidence_pct:.1f}%")
    print(f"  Latency    : {inf_time:.3f} ms")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Irrigation Demo CLI")
    parser.add_argument("--moi", type=float, default=32.0, help="Soil Moisture Percentage (e.g., 32.0)")
    parser.add_argument("--temp", type=float, default=35.0, help="Temperature in Celsius (e.g., 35.0)")
    parser.add_argument("--humidity", type=float, default=45.0, help="Relative Humidity Percentage (e.g., 45.0)")
    parser.add_argument("--model", type=str, choices=["ann", "xgb"], default="ann", help="Model to use: 'ann' or 'xgb'")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing trained models")
    
    args = parser.parse_args()
    
    run_demo(
        moi=args.moi,
        temp=args.temp,
        humidity=args.humidity,
        use_model=args.model,
        models_dir=args.models_dir
    )
