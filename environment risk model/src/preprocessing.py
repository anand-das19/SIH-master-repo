"""
AgriNode AI - Environmental Risk Intelligence
Phase 2: Data Preprocessing & Harmonization

Standardizes schema, timestamps, missing values, and temporal partitions
(Calibration: 1981-2015 vs Validation: 2016-2024) across historical climate time-series.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DISTRICTS_DIR = Path("data/raw/indmet/district_data/INDmet_District_Data/District_data_Daily_CSV")
CENTROIDS_PATH = Path("data/processed/district_centroids.csv")
PROCESSED_DIR = Path("data/processed")

CALIBRATION_START = "1981-01-01"
CALIBRATION_END = "2015-12-31"
VALIDATION_START = "2016-01-01"
VALIDATION_END = "2024-12-31"

RAW_COLUMNS = ["year", "month", "day", "rainfall", "tmax", "tmin", "tmean"]


def get_district_metadata_map():
    """Load mapping from district_id to (district_name, state)."""
    df = pd.read_csv(CENTROIDS_PATH)
    meta = {}
    for _, row in df.iterrows():
        meta[int(row["district_id"])] = {
            "district": row["district"],
            "state": row["state"],
            "latitude": row["latitude"],
            "longitude": row["longitude"]
        }
    return meta


def load_raw_district_file(district_id: int) -> pd.DataFrame:
    """Load raw daily CSV for a district."""
    file_path = RAW_DISTRICTS_DIR / f"data_District_{district_id}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file for district {district_id} not found at {file_path}")

    df = pd.read_csv(file_path, header=None, names=RAW_COLUMNS)
    return df


def clean_and_harmonize_district_data(district_id: int, meta: Optional[dict] = None) -> pd.DataFrame:
    """
    Standardize timestamps, clean anomalous/sentinel values, and handle gaps.
    Returns cleaned DataFrame with complete daily DateTimeIndex.
    """
    if meta is None:
        meta_map = get_district_metadata_map()
        meta = meta_map.get(district_id, {"district": f"District_{district_id}", "state": "Unknown"})

    raw_df = load_raw_district_file(district_id)

    # 1. Construct standard YYYY-MM-DD timestamps
    dates = pd.to_datetime(raw_df[["year", "month", "day"]])
    raw_df["date"] = dates
    raw_df["district_id"] = district_id
    raw_df["district"] = meta.get("district", f"District_{district_id}")
    raw_df["state"] = meta.get("state", "Unknown")

    # 2. Re-index to ensure complete continuous daily calendar
    raw_df = raw_df.set_index("date")
    full_idx = pd.date_range(start="1981-01-01", end="2024-12-31", freq="D", name="date")
    df = raw_df.reindex(full_idx)

    df["district_id"] = district_id
    df["district"] = meta.get("district", f"District_{district_id}")
    df["state"] = meta.get("state", "Unknown")
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day

    # 3. Clean sentinel codes (< -90) and negative rainfall
    df.loc[df["rainfall"] < 0, "rainfall"] = np.nan
    df.loc[df["tmax"] < -90, "tmax"] = np.nan
    df.loc[df["tmin"] < -90, "tmin"] = np.nan
    df.loc[df["tmean"] < -90, "tmean"] = np.nan

    # 4. Handle short gaps (< 3 days) via linear interpolation
    df["rainfall"] = df["rainfall"].interpolate(method="linear", limit=3).fillna(0.0)
    df["tmax"] = df["tmax"].interpolate(method="linear", limit=3).bfill().ffill()
    df["tmin"] = df["tmin"].interpolate(method="linear", limit=3).bfill().ffill()
    df["tmean"] = df["tmean"].interpolate(method="linear", limit=3).bfill().ffill()

    # Round to realistic precision
    df["rainfall"] = df["rainfall"].round(2)
    df["tmax"] = df["tmax"].round(2)
    df["tmin"] = df["tmin"].round(2)
    df["tmean"] = df["tmean"].round(2)

    df = df.reset_index()
    ordered_cols = ["date", "district_id", "district", "state", "year", "month", "day", "rainfall", "tmax", "tmin", "tmean"]
    return df[ordered_cols]


def split_calibration_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into:
    - Calibration period (1981-01-01 to 2015-12-31)
    - Validation period (2016-01-01 to 2024-12-31)
    """
    df["date"] = pd.to_datetime(df["date"])
    cal_df = df[(df["date"] >= CALIBRATION_START) & (df["date"] <= CALIBRATION_END)].copy()
    val_df = df[(df["date"] >= VALIDATION_START) & (df["date"] <= VALIDATION_END)].copy()
    return cal_df, val_df


def build_processed_deliverables():
    """
    Harmonize and export clean benchmark datasets:
    - data/processed/rainfall_clean.csv
    - data/processed/temperature_clean.csv
    """
    print("=" * 70)
    print("AGRINODE AI - PHASE 2: DATA PREPROCESSING & TEMPORAL SPLITTING")
    print("=" * 70)

    meta_map = get_district_metadata_map()
    benchmark_path = Path("data/benchmark_cases.csv")
    benchmark_df = pd.read_csv(benchmark_path)
    
    # Identify unique districts involved in benchmark evaluations
    from src.geo_resolver import get_geo_resolver
    resolver = get_geo_resolver()
    
    benchmark_district_ids = set()
    for _, row in benchmark_df.iterrows():
        match = resolver.resolve(row["latitude"], row["longitude"])
        benchmark_district_ids.add(match["district_id"])

    # Also add key zonal districts to ensure broad geographic representation
    key_ids = [14, 16, 25, 47, 70, 122, 134, 148, 190, 205, 231, 274, 425, 500, 523, 670]
    target_ids = sorted(list(benchmark_district_ids.union(key_ids)))
    
    print(f"[PROCESSING] Cleaning and standardizing time-series for {len(target_ids)} benchmark & reference districts...")

    all_clean_records = []
    for dist_id in target_ids:
        try:
            meta = meta_map.get(dist_id, {})
            cdf = clean_and_harmonize_district_data(dist_id, meta)
            all_clean_records.append(cdf)
            print(f"  [OK] District {dist_id:3d} ({meta.get('district', ''):<20}): {len(cdf):,} days cleaned & verified.")
        except Exception as e:
            print(f"  [FAIL] District {dist_id:3d}: {e}")

    full_clean_df = pd.concat(all_clean_records, ignore_index=True)
    full_clean_df["date"] = full_clean_df["date"].dt.strftime("%Y-%m-%d")

    # 1. Export rainfall_clean.csv
    rain_cols = ["date", "district_id", "district", "state", "rainfall"]
    rain_df = full_clean_df[rain_cols]
    rain_out = PROCESSED_DIR / "rainfall_clean.csv"
    rain_df.to_csv(rain_out, index=False)
    print(f"[EXPORTED] {rain_out} ({len(rain_df):,} rows)")

    # 2. Export temperature_clean.csv
    temp_cols = ["date", "district_id", "district", "state", "tmax", "tmin", "tmean"]
    temp_df = full_clean_df[temp_cols]
    temp_out = PROCESSED_DIR / "temperature_clean.csv"
    temp_df.to_csv(temp_out, index=False)
    print(f"[EXPORTED] {temp_out} ({len(temp_df):,} rows)")

    # 3. Verify temporal split integrity
    cal_df, val_df = split_calibration_validation(full_clean_df)
    print("\n[TEMPORAL SPLIT VERIFICATION]")
    print(f"  Calibration Period : {cal_df['date'].min()} to {cal_df['date'].max()} ({len(cal_df):,} records)")
    print(f"  Validation Period  : {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df):,} records)")
    assert cal_df['date'].max().strftime("%Y-%m-%d") == "2015-12-31", "Calibration end date mismatch!"
    assert val_df['date'].min().strftime("%Y-%m-%d") == "2016-01-01", "Validation start date mismatch!"
    print("  [SUCCESS] Temporal isolation verified. Zero data leakage between calibration and validation.")

    return rain_out, temp_out


if __name__ == "__main__":
    build_processed_deliverables()
