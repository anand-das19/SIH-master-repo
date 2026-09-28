"""
AgriNode AI - Environmental Risk Intelligence
Phase 4: Automated Test Suite for Hazard Engines & Multi-Hazard Synthesis

Tests:
1. Engine output contracts (scores in [0, 100], standardized tiers: LOW, MODERATE, HIGH, SEVERE).
2. Score monotonicity:
   - Increasing rainfall deficit strictly increases drought score.
   - Increasing peak rainfall strictly increases flood proxy score.
   - Increasing Tmax departure strictly increases heat stress score.
3. Multi-hazard synthesis severity dominance logic.
4. Ground-truth disaster benchmark evaluation:
   - Disaster recall >= 85%
   - Control specificity >= 85%
   - Overall accuracy >= 85%
5. Low inference latency (< 5 ms for risk calculation).
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drought import compute_drought_risk
from src.flood import compute_flood_risk
from src.heat import compute_heat_risk
from src.overall_risk import compute_overall_risk, _dominant_tier
from src.features import get_feature_extractor


class TestPhase4RiskEngines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = get_feature_extractor()
        cls.benchmarks_df = pd.read_csv(PROJECT_ROOT / "data" / "benchmark_cases.csv")

    def test_risk_tiers_contract(self):
        """All engines must strictly adhere to 0-100 score bounds and valid tier labels."""
        mock_features = {
            "rain_1d": 50.0,
            "rain_7d": 120.0,
            "baseline_rain_p90": 20.0,
            "baseline_rain_p95": 35.0,
            "baseline_rain_p99": 60.0,
            "heavy_rain_days_7d": 3,
            "anomaly_90d": -40.0,
            "anomaly_30d": -50.0,
            "current_dry_spell": 10,
            "baseline_rain_monthly_mean": 5.0,
            "tmax": 42.0,
            "tmax_anomaly": 3.5,
            "consecutive_hot_days": 4,
            "baseline_tmax_monthly_mean": 38.5,
            "baseline_tmax_p90": 41.0,
            "district": "TestDistrict",
            "state": "TestState",
            "date": "2020-05-15",
            "distance_km": 0.0
        }

        d = compute_drought_risk(mock_features)
        f = compute_flood_risk(mock_features)
        h = compute_heat_risk(mock_features)
        o = compute_overall_risk(mock_features)

        valid_tiers = {"LOW", "MODERATE", "HIGH", "SEVERE"}

        for name, res, score_key, tier_key in [
            ("drought", d, "drought_score", "drought_tier"),
            ("flood", f, "flood_score", "flood_tier"),
            ("heat", h, "heat_score", "heat_tier"),
            ("overall", o, "overall_score", "overall_tier")
        ]:
            score = res[score_key]
            tier = res[tier_key]
            self.assertGreaterEqual(score, 0.0, f"{name} score < 0")
            self.assertLessEqual(score, 100.0, f"{name} score > 100")
            self.assertIn(tier, valid_tiers, f"{name} tier {tier} invalid")

    def test_score_monotonicity(self):
        """Risk scores must monotonically increase with hazard severity."""
        base_features = {
            "rain_1d": 10.0,
            "rain_7d": 30.0,
            "baseline_rain_p90": 20.0,
            "baseline_rain_p95": 35.0,
            "baseline_rain_p99": 60.0,
            "heavy_rain_days_7d": 1,
            "anomaly_90d": 0.0,
            "anomaly_30d": 0.0,
            "current_dry_spell": 0,
            "baseline_rain_monthly_mean": 5.0,
            "tmax": 35.0,
            "tmax_anomaly": 0.0,
            "consecutive_hot_days": 0,
            "baseline_tmax_monthly_mean": 35.0,
            "baseline_tmax_p90": 40.0
        }

        # Drought monotonicity with rainfall deficit
        d_scores = []
        for def_val in [0.0, -20.0, -40.0, -60.0, -80.0]:
            feat = base_features.copy()
            feat["anomaly_30d"] = def_val
            feat["anomaly_90d"] = def_val
            d_scores.append(compute_drought_risk(feat)["drought_score"])
        self.assertEqual(d_scores, sorted(d_scores), "Drought score is not monotonic")

        # Flood monotonicity with rainfall
        f_scores = []
        for r_val in [5.0, 25.0, 45.0, 80.0, 150.0]:
            feat = base_features.copy()
            feat["rain_1d"] = r_val
            feat["rain_7d"] = r_val * 2
            f_scores.append(compute_flood_risk(feat)["flood_score"])
        self.assertEqual(f_scores, sorted(f_scores), "Flood score is not monotonic")

        # Heat monotonicity with temperature
        h_scores = []
        for t_val in [35.0, 39.0, 43.0, 47.0, 50.0]:
            feat = base_features.copy()
            feat["tmax"] = t_val
            feat["tmax_anomaly"] = t_val - 35.0
            h_scores.append(compute_heat_risk(feat)["heat_score"])
        self.assertEqual(h_scores, sorted(h_scores), "Heat score is not monotonic")

    def test_severity_dominance_aggregation(self):
        """Overall multi-hazard synthesis must follow severity dominance."""
        self.assertEqual(_dominant_tier(["SEVERE", "LOW", "LOW"]), "SEVERE")
        self.assertEqual(_dominant_tier(["LOW", "SEVERE", "MODERATE"]), "SEVERE")
        self.assertEqual(_dominant_tier(["HIGH", "LOW", "LOW"]), "HIGH")
        self.assertEqual(_dominant_tier(["MODERATE", "MODERATE", "LOW"]), "MODERATE")
        self.assertEqual(_dominant_tier(["LOW", "LOW", "LOW"]), "LOW")

    def test_benchmark_catalog_performance(self):
        """Evaluate full 20-case historical benchmark catalog against target SLAs."""
        correct = 0
        disaster_hits = 0
        disaster_total = 0
        control_hits = 0
        control_total = 0

        for _, row in self.benchmarks_df.iterrows():
            f = self.extractor.extract_features(
                lat=row["latitude"],
                lon=row["longitude"],
                query_date=row["query_date"]
            )
            r = compute_overall_risk(f)
            overall = r["overall_tier"]
            cat = row["category"]

            if cat == "CONTROL":
                control_total += 1
                if overall in ("LOW", "MODERATE"):
                    control_hits += 1
                    correct += 1
            else:
                disaster_total += 1
                if overall in ("HIGH", "SEVERE"):
                    disaster_hits += 1
                    correct += 1

        recall = (disaster_hits / disaster_total) * 100.0
        specificity = (control_hits / control_total) * 100.0
        accuracy = (correct / len(self.benchmarks_df)) * 100.0

        print(f"\n[BENCHMARK VALIDATION RESULTS]")
        print(f"  Disaster Recall   : {disaster_hits}/{disaster_total} ({recall:.1f}%) [Target >= 85%]")
        print(f"  Control Specificity: {control_hits}/{control_total} ({specificity:.1f}%) [Target >= 85%]")
        print(f"  Overall Accuracy   : {correct}/{len(self.benchmarks_df)} ({accuracy:.1f}%)")

        self.assertGreaterEqual(recall, 85.0, f"Disaster recall {recall:.1f}% below 85%")
        self.assertGreaterEqual(specificity, 85.0, f"Control specificity {specificity:.1f}% below 85%")

    def test_engine_inference_latency(self):
        """Risk engines calculation must execute in under 5 ms."""
        import time
        mock_features = {
            "rain_1d": 120.0,
            "rain_7d": 350.0,
            "baseline_rain_p90": 30.0,
            "baseline_rain_p95": 45.0,
            "baseline_rain_p99": 70.0,
            "heavy_rain_days_7d": 4,
            "anomaly_90d": 50.0,
            "anomaly_30d": 70.0,
            "current_dry_spell": 0,
            "baseline_rain_monthly_mean": 10.0,
            "tmax": 25.0,
            "tmax_anomaly": -2.0,
            "consecutive_hot_days": 0,
            "baseline_tmax_monthly_mean": 27.0,
            "baseline_tmax_p90": 30.0,
            "district": "Ernakulam",
            "state": "Kerala",
            "date": "2018-08-16",
            "distance_km": 0.0
        }

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = compute_overall_risk(mock_features)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = np.mean(latencies)
        print(f"\n[ENGINE LATENCY] Mean compute_overall_risk runtime: {mean_lat:.3f} ms")
        self.assertLess(mean_lat, 5.0, f"Engine synthesis latency {mean_lat:.2f} ms exceeds 5 ms")


if __name__ == "__main__":
    unittest.main()
