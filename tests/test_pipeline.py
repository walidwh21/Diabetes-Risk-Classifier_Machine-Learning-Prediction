import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from diabetes_svm import build_pipeline, load_dataset, predict_risk, split_features_target  # noqa: E402
from diabetes_svm.config import FEATURES, ZERO_AS_MISSING  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset()
        cls.X, cls.y = split_features_target(cls.frame)
        cls.model = build_pipeline("linear").fit(cls.X, cls.y)

    def test_shape_and_columns(self):
        self.assertEqual(self.X.shape, (768, 8))
        self.assertEqual(list(self.X.columns), FEATURES)

    def test_impossible_zeros_become_nan(self):
        for col in ZERO_AS_MISSING:
            self.assertEqual((self.frame[col] == 0).sum(), 0)
        self.assertGreater(self.frame["Insulin"].isna().sum(), 300)

    def test_model_beats_majority_baseline(self):
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=0)
        acc = build_pipeline("linear").fit(X_tr, y_tr).score(X_te, y_te)
        self.assertGreater(acc, 0.65)

    def test_prediction_handles_missing_inputs(self):
        report = predict_risk({"Glucose": 180, "BMI": 35, "Age": 55}, model=self.model)
        self.assertTrue(0.0 <= report.probability <= 1.0)
        self.assertIn(report.band, {"low", "moderate", "high"})

    def test_higher_glucose_raises_risk(self):
        low = predict_risk({"Glucose": 80}, model=self.model).probability
        high = predict_risk({"Glucose": 190}, model=self.model).probability
        self.assertGreater(high, low)

    def test_unknown_feature_rejected(self):
        with self.assertRaises(KeyError):
            predict_risk({"Shoe size": 42}, model=self.model)


if __name__ == "__main__":
    unittest.main()
