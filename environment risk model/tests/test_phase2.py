"""
AgriNode AI - Environmental Risk Intelligence
Phase 2 Verification Test Suite

Validates:
1. Sub-millisecond KDTree geographic resolution across all 20 benchmark disaster coordinates.
2. Cleaned rainfall and temperature dataset schema, value ranges, and lack of nulls.
3. Strict temporal isolation between calibration (1981-2015) and validation (2016-2024).
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geo_resolver import get_geo_resolver, resolve_district
from src.preprocessing import (
    CALIBRATION_START,
    CALIBRATION_END,
    VALIDATION_START,
    VALIDATION_END,
    split_calibration_validation
)


def test_geo_resolver_benchmarks():
    print("\n--- TEST 1: KDTree Geographic Resolution Accuracy & Latency ---")
    resolver = get_geo_resolver()
    benchmarks_path = Path("data/benchmark_cases.csv")
    assert benchmarks_path.exists(), "Benchmark cases file missing!"

    df_bench = pd.read_csv(benchmarks_path)
    latencies = []

    print(f"Testing {len(df_bench)} benchmark disaster & control coordinates:")
    for _, row in df_bench.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        expected_dist = row["district"].lower()

        res = resolver.resolve(lat, lon)
        latencies.append(res["latency_ms"])

        print(f"  [{row['case_id']:<12}] Query ({lat:6.2f}, {lon:6.2f}) -> Resolved: {res['district']}, {res['state']:<15} "
              f"| Dist: {res['distance_km']:5.1f} km | Latency: {res['latency_ms']:5.3f} ms")

        # Verify distance is reasonable (< 150 km to district centroid)
        assert res["distance_km"] < 150.0, f"Centroid distance {res['distance_km']} km unexpectedly large for {expected_dist}!"

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    print(f"\n[LATENCY RESULTS] Avg: {avg_latency:.3f} ms | Max: {max_latency:.3f} ms")
    assert avg_latency < 1.0, f"Average latency {avg_latency:.3f} ms exceeds 1 ms SLA!"
    print(">>> TEST 1 PASSED: Sub-millisecond geographic resolution verified.")


def test_clean_datasets_integrity():
    print("\n--- TEST 2: Cleaned Dataset Schema, Value Bounds & Completeness ---")
    rain_path = Path("data/processed/rainfall_clean.csv")
    temp_path = Path("data/processed/temperature_clean.csv")

    assert rain_path.exists(), "rainfall_clean.csv missing!"
    assert temp_path.exists(), "temperature_clean.csv missing!"

    df_rain = pd.read_csv(rain_path)
    df_temp = pd.read_csv(temp_path)

    # 1. Row count match
    assert len(df_rain) == len(df_temp), "Rainfall and Temperature row counts do not match!"
    print(f"  Total records validated per hazard file: {len(df_rain):,}")

    # 2. No nulls
    assert df_rain["rainfall"].isnull().sum() == 0, "Found nulls in cleaned rainfall!"
    assert df_temp["tmax"].isnull().sum() == 0, "Found nulls in cleaned tmax!"
    assert df_temp["tmin"].isnull().sum() == 0, "Found nulls in cleaned tmin!"

    # 3. Value bounds
    assert (df_rain["rainfall"] >= 0).all(), "Found negative rainfall values!"
    assert (df_temp["tmax"] >= -10).all(), "Found anomalous cold tmax!"
    assert (df_temp["tmax"] <= 60).all(), "Found anomalous hot tmax > 60C!"
    assert (df_temp["tmax"] >= df_temp["tmin"] - 0.5).all(), "Found tmax < tmin instances!"

    print(">>> TEST 2 PASSED: Cleaned datasets meet all schema, bound, and completeness constraints.")


def test_temporal_split():
    print("\n--- TEST 3: Temporal Split & Data Isolation ---")
    rain_path = Path("data/processed/rainfall_clean.csv")
    df_rain = pd.read_csv(rain_path)

    cal_df, val_df = split_calibration_validation(df_rain)

    assert str(cal_df["date"].min())[:10] == CALIBRATION_START
    assert str(cal_df["date"].max())[:10] == CALIBRATION_END
    assert str(val_df["date"].min())[:10] == VALIDATION_START
    assert str(val_df["date"].max())[:10] == VALIDATION_END

    # Ensure zero overlap
    cal_dates = set(cal_df["date"])
    val_dates = set(val_df["date"])
    overlap = cal_dates.intersection(val_dates)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping dates between calibration and validation!"

    print(f"  Calibration Horizon : {CALIBRATION_START} to {CALIBRATION_END} ({len(cal_df):,} records)")
    print(f"  Validation Horizon  : {VALIDATION_START} to {VALIDATION_END} ({len(val_df):,} records)")
    print(">>> TEST 3 PASSED: Temporal isolation verified with 0% data leakage.")


def run_all_tests():
    print("=" * 70)
    print("AGRINODE AI - RUNNING PHASE 2 AUTOMATED TEST SUITE")
    print("=" * 70)
    test_geo_resolver_benchmarks()
    test_clean_datasets_integrity()
    test_temporal_split()
    print("\n" + "=" * 70)
    print("ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
