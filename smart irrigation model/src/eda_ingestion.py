import os
import json
import pandas as pd
import numpy as np

def run_phase1_ingestion(data_path="dataset/data.csv", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    print(f"[*] Ingesting raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[+] Initial shape: {df.shape}")

    df.columns = [c.strip() for c in df.columns]

    # Binary target: 0 = Pump OFF, 1 = Pump ON (result > 0)
    df["need_irrigation"] = (df["result"] > 0).astype(int)

    features = ["MOI", "temp", "humidity"]
    total = len(df)
    class_counts = df["need_irrigation"].value_counts().to_dict()
    c0 = class_counts.get(0, 0)
    c1 = class_counts.get(1, 0)
    p0 = round((c0 / total) * 100, 2)
    p1 = round((c1 / total) * 100, 2)

    corr = df[features + ["need_irrigation"]].corr().round(4).to_dict()
    grouped = df.groupby("need_irrigation")[features].agg(["mean", "std", "min", "median", "max"]).round(2)

    profile = {
        "total_records": total,
        "missing_values": {c: int(v) for c, v in df[features + ["result"]].isnull().sum().to_dict().items()},
        "class_distribution": {"0_pump_off": int(c0), "1_pump_on": int(c1)},
        "class_percentages": {"0_pump_off": f"{p0}%", "1_pump_on": f"{p1}%"},
        "correlations_with_target": {feat: corr["need_irrigation"][feat] for feat in features}
    }

    summary_path = os.path.join(output_dir, "phase1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)
    print(f"[+] Exported profile metrics to {summary_path}")

    clean_csv = "dataset/cleaned_binary_irrigation.csv"
    df.to_csv(clean_csv, index=False)
    print(f"[+] Exported binary cleaned dataset to {clean_csv}")

    print("\n================= PHASE 1 INGESTION SUMMARY =================")
    print(f"Total Records       : {total}")
    print(f"Class 0 (Pump OFF)  : {c0} ({p0}%)")
    print(f"Class 1 (Pump ON)   : {c1} ({p1}%)")
    print("\nTarget-Feature Pearson Correlations:")
    for feat in features:
        print(f"  {feat:<12}: {corr['need_irrigation'][feat]:+.4f}")
    print("\nGrouped Feature Statistics by Irrigation Decision:")
    print(grouped)
    print("=============================================================\n")
    return df

if __name__ == "__main__":
    run_phase1_ingestion()
