import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
model = joblib.load(model_path)

class TestModelTraining(unittest.TestCase):
    def test_model_type(self):
        """Check if model is a RandomForestClassifier"""
        self.assertIsInstance(model, RandomForestClassifier)

    def test_feature_importances(self):
        """Check if feature importances match churn dataset features"""
        # Churn dataset has 20 input features after preprocessing
        self.assertEqual(len(model.feature_importances_), 20)

if __name__ == "__main__":
    unittest.main()
