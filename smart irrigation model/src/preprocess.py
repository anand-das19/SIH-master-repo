import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_phase2_preprocessing(
    input_csv="dataset/cleaned_binary_irrigation.csv",
    dataset_dir="dataset",
    models_dir="models",
    random_state=42
):
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"[*] Loading dataset from {input_csv}...")
    if not os.path.exists(input_csv):
        from src.eda_ingestion import run_phase1_ingestion
        df = run_phase1_ingestion()
    else:
        df = pd.read_csv(input_csv)

    feature_cols = ["MOI", "temp", "humidity"]
    target_col = "need_irrigation"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    total_len = len(df)
    print(f"[+] Total samples: {total_len}")

    # Stratified Split: 70% Train, 30% Temp (which splits 15% Val / 15% Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=random_state
    )

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"[+] Splits created (Stratified):")
    print(f"    Train : {len(X_train_raw)} ({len(X_train_raw)/total_len*100:.1f}%) | Class 1: {y_train.mean()*100:.2f}%")
    print(f"    Val   : {len(X_val_raw)} ({len(X_val_raw)/total_len*100:.1f}%) | Class 1: {y_val.mean()*100:.2f}%")
    print(f"    Test  : {len(X_test_raw)} ({len(X_test_raw)/total_len*100:.1f}%) | Class 1: {y_test.mean()*100:.2f}%")

    # Fit Scaler strictly on Train split (leak-free)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Save Scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved feature scaler to: {scaler_path}")

    # Export split CSVs (Raw sensor values for transparency & demo inspection)
    train_df = X_train_raw.copy()
    train_df[target_col] = y_train.values
    val_df = X_val_raw.copy()
    val_df[target_col] = y_val.values
    test_df = X_test_raw.copy()
    test_df[target_col] = y_test.values

    train_path = os.path.join(dataset_dir, "train.csv")
    val_path = os.path.join(dataset_dir, "val.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[+] Saved raw split CSVs: {train_path}, {val_path}, {test_path}")

    # Export bundled scaled splits for fast ML & DL ingestion
    splits_payload = {
        "features": feature_cols,
        "target": target_col,
        "X_train_raw": X_train_raw.values,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train.values,
        "X_val_raw": X_val_raw.values,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val.values,
        "X_test_raw": X_test_raw.values,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test.values,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    processed_bundle_path = os.path.join(dataset_dir, "processed_splits.joblib")
    joblib.dump(splits_payload, processed_bundle_path)
    print(f"[+] Saved scaled splits bundle to: {processed_bundle_path}")

    print("\n================= PHASE 2 PREPROCESSING COMPLETE =================")
    print(f"Scaler Means (MOI, temp, humidity): {scaler.mean_.round(2).tolist()}")
    print(f"Scaler Scales (std dev)           : {scaler.scale_.round(2).tolist()}")
    print("==================================================================\n")

    return splits_payload

if __name__ == "__main__":
    run_phase2_preprocessing()
