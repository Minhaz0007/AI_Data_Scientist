import unittest
import pandas as pd
import numpy as np
from utils.ml_engine import ml_engine as ml
import components.workflow

class TestAdvanced(unittest.TestCase):
    def setUp(self):
        # Small dataset
        self.df = pd.DataFrame({
            'X1': np.random.rand(50),
            'X2': np.random.rand(50),
            'y': np.random.randint(0, 2, 50)
        })
        self.ts = pd.Series(np.random.randn(50) + np.linspace(0, 10, 50)) # Trend

    def test_hyperparameter_tuning(self):
        # Use simple model to be fast
        res = ml.optimize_hyperparameters(self.df[['X1', 'X2']], self.df['y'], 'KNN', cv=2)
        print("Tuning result keys:", res.keys())
        self.assertIn('best_params', res)
        self.assertIn('best_score', res)

    def test_auto_arima(self):
        # Mocking or running simple search
        model, order = ml.auto_arima_search(self.ts, max_p=1, max_d=1, max_q=1)
        print("Best ARIMA order:", order)
        self.assertIsInstance(order, tuple)
        self.assertEqual(len(order), 3)

    def test_workflow_import(self):
        # Just checking if it imports without error
        self.assertTrue(hasattr(components.workflow, 'render'))

if __name__ == '__main__':
    unittest.main()
