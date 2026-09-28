"""Artifact-level smoke tests for the irrigation classifier.

Run from the repository root with: python -m unittest tests.test_irrigation_smoke
"""

import unittest

import numpy as np
import torch

from src.demo_predict import calculate_pump_runtime, load_ann_model, load_scaler, load_xgb_model


class IrrigationArtifactSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scaler = load_scaler()
        cls.xgb = load_xgb_model()
        cls.ann = load_ann_model()

    def probabilities(self, moi, temp, humidity):
        x = self.scaler.transform(np.array([[moi, temp, humidity]], dtype=float))
        xgb_probability = self.xgb.predict_proba(x)[0, 1]
        with torch.no_grad():
            ann_probability = torch.sigmoid(self.ann(torch.tensor(x, dtype=torch.float32))).item()
        return xgb_probability, ann_probability

    def test_dry_nominal_reading_requests_irrigation(self):
        for probability in self.probabilities(5, 35, 45):
            self.assertGreaterEqual(probability, 0.5)

    def test_wet_nominal_reading_does_not_request_irrigation(self):
        for probability in self.probabilities(95, 25, 80):
            self.assertLess(probability, 0.5)

    def test_runtime_mapping_is_bounded_and_monotonic(self):
        self.assertEqual(calculate_pump_runtime(5, 1), 40)
        self.assertEqual(calculate_pump_runtime(55, 1), 15)
        self.assertEqual(calculate_pump_runtime(95, 1), 15)
        self.assertEqual(calculate_pump_runtime(5, 0), 0)


if __name__ == "__main__":
    unittest.main()
