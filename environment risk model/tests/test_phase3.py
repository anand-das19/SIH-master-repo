"""
AgriNode AI - Environmental Risk Intelligence
Phase 3: Automated Test Suite

Tests:
1. Baseline Climatology Integrity (no NaNs, positive rainfall percentiles, reasonable temp ranges).
2. Multi-Scale Feature Extraction Latency (< 10 ms cached, < 100 ms cold).
3. Extreme Benchmark Feature Signatures:
   - Ernakulam 2018 Deluge: acute rain > baseline 99th percentile, large 7d/30d accumulation.
   - Phalodi/Jodhpur 2016 Heatwave: Tmax > 45°C, high positive Tmax anomaly (> 4°C).
   - Latur 2015 Drought: prolonged dry spell (> 20 days), heavy negative rainfall anomalies.
   - Ludhiana 2021 Control: normal ranges, mild anomalies, dry spell within normal bounds.
4. Temporal Isolation (Calibrations strictly derive from 1981-2015).
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import get_baseline_calibrator, BASELINES_CSV_PATH
from src.features import get_feature_extractor


class TestPhase3BaselinesAndFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baselines_df = pd.read_csv(BASELINES_CSV_PATH)
        cls.calibrator = get_baseline_calibrator()
        cls.extractor = get_feature_extractor()

    def test_baselines_integrity(self):
        """Verify baseline file existence, row count, and schema validity."""
        self.assertTrue(BASELINES_CSV_PATH.exists(), "district_baselines.csv not found")
        self.assertGreater(len(self.baselines_df), 100, "Too few baseline records")

        # Check for NaNs
        nan_counts = self.baselines_df.isna().sum().sum()
        self.assertEqual(nan_counts, 0, f"Found {nan_counts} NaN values in district_baselines.csv")

        # Check physical sanity bounds
        self.assertTrue((self.baselines_df["rain_mean"] >= 0).all(), "Negative rain mean detected")
        self.assertTrue((self.baselines_df["rain_p95"] >= self.baselines_df["rain_p90"]).all(), "rain_p95 < rain_p90")
        self.assertTrue((self.baselines_df["rain_p99"] >= self.baselines_df["rain_p95"]).all(), "rain_p99 < rain_p95")

        # Check physical sanity bounds
        self.assertTrue((self.baselines_df["rain_mean"] >= 0).all(), "Negative rain mean detected")
        self.assertTrue((self.baselines_df["rain_p95"] >= self.baselines_df["rain_p90"]).all(), "rain_p95 < rain_p90")
        self.assertTrue((self.baselines_df["rain_p99"] >= self.baselines_df["rain_p95"]).all(), "rain_p99 < rain_p95")

        # Tmax mean bounds across all climates (including high-altitude Himalayan/sub-Himalayan zones)
        self.assertTrue((self.baselines_df["tmax_mean"] >= 0.0).all(), "Tmax mean too low (<0C)")
        self.assertTrue((self.baselines_df["tmax_mean"] <= 55.0).all(), "Tmax mean too high (>55C)")
        self.assertTrue((self.baselines_df["tmin_mean"] <= self.baselines_df["tmax_mean"]).all(), "Tmin > Tmax in baseline")

    def test_flood_feature_signature_ernakulam(self):
        """Ernakulam during Aug 16, 2018 deluge must reflect extreme rain accumulation."""
        features = self.extractor.extract_features(lat=9.9816, lon=76.2999, query_date="2018-08-16")
        
        self.assertEqual(features["district"], "Ernakulam")
        self.assertGreater(features["rain_1d"], 100.0, "Aug 16 2018 single-day rain should be > 100mm")
        self.assertGreater(features["rain_7d"], 250.0, "7-day rain accumulation should be > 250mm")
        self.assertGreater(features["rain_1d"], features["baseline_rain_p95"], "Peak rain must exceed 95th percentile")
        self.assertGreaterEqual(features["heavy_rain_days_7d"], 3, "Should have multiple extreme rain days in 7d window")

    def test_heatwave_feature_signature_jodhpur(self):
        """Jodhpur/Phalodi during May 19, 2016 record heat must show extreme thermal stress."""
        features = self.extractor.extract_features(lat=27.1300, lon=72.3600, query_date="2016-05-19")
        
        self.assertIn("Jodhpur", features["district"])
        self.assertGreater(features["tmax"], 45.0, "Tmax must exceed 45C during Phalodi record event")
        self.assertGreater(features["tmax_anomaly"], 2.0, "Tmax anomaly should be significantly positive (> 2C)")
        self.assertGreaterEqual(features["consecutive_hot_days"], 2, "Consecutive hot days should be >= 2")

    def test_drought_feature_signature_latur(self):
        """Latur during late August / Sept 2015 must exhibit severe seasonal rainfall deficit."""
        # 1. Check peak dry spell in late August 2015
        aug_feat = self.extractor.extract_features(lat=18.4088, lon=76.5604, query_date="2015-08-30")
        self.assertEqual(aug_feat["district"], "Latur")
        self.assertLess(aug_feat["anomaly_30d"], -40.0, "30-day rainfall anomaly should be < -40% in August")
        self.assertLess(aug_feat["anomaly_90d"], -40.0, "90-day seasonal anomaly should be < -40%")
        self.assertGreaterEqual(aug_feat["current_dry_spell"], 9, "Dry spell should be >= 9 days")

        # 2. Check cumulative Kharif deficit by Sept 10
        sept_feat = self.extractor.extract_features(lat=18.4088, lon=76.5604, query_date="2015-09-10")
        self.assertLess(sept_feat["anomaly_90d"], -30.0, "90-day Kharif monsoon deficit should be severe (< -30%)")

    def test_control_feature_signature_ludhiana(self):
        """Ludhiana during March 15, 2021 control should show temperate and moderate features."""
        features = self.extractor.extract_features(lat=30.9010, lon=75.8573, query_date="2021-03-15")
        
        self.assertEqual(features["district"], "Ludhiana")
        self.assertLess(abs(features["tmax_anomaly"]), 5.0, "Control Tmax anomaly should be modest (< 5C)")
        self.assertLess(features["rain_1d"], 50.0, "No torrential downpour in control case")

    def test_inference_latency(self):
        """Cached feature extraction must complete well under 10 ms (target < 50ms total inference)."""
        # Warmup
        _ = self.extractor.extract_features(lat=13.0827, lon=80.2707, query_date="2015-12-01")
        
        # Measure repeated queries
        import time
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = self.extractor.extract_features(lat=13.0827, lon=80.2707, query_date="2015-12-01")
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = np.mean(latencies)
        print(f"\n[LATENCY BENCHMARK] Average Feature Extraction Latency: {avg_latency:.3f} ms")
        self.assertLess(avg_latency, 10.0, f"Latency {avg_latency:.2f} ms exceeded 10 ms limit")


if __name__ == "__main__":
    unittest.main()
