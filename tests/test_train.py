import unittest
import joblib
import numpy as np
from train import classifiers

class TestTrain(unittest.TestCase):
    def test_models_saved(self):
        # Check that the models are saved correctly
        models = ['logistic_regression.pkl', 'linear_svc.pkl', 'knn.pkl']
        for model in models:
            loaded_model = joblib.load(model)
            self.assertIsNotNone(loaded_model)

if __name__ == '__main__':
    unittest.main()