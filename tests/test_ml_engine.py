import unittest
import pandas as pd
import numpy as np
from utils.ml_engine import MLEngine

class TestMLEngine(unittest.TestCase):

    def setUp(self):
        self.engine = MLEngine()
        # Regression Data
        np.random.seed(42)
        self.df_reg = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        # Classification Data
        self.df_cls = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], 100)
        })
        # Time Series Data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.df_ts = pd.DataFrame({
            'date': dates,
            'value': np.linspace(0, 10, 100) + np.random.normal(0, 0.1, 100)
        })

    def test_train_regression(self):
        X = self.df_reg[['feature1', 'feature2']]
        y = self.df_reg['target']
        result = self.engine.train_regression(X, y, model_name='Linear Regression')
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('predictions', result)
        self.assertEqual(result['model_name'], 'Linear Regression')

    def test_train_classification(self):
        X = self.df_cls[['feature1', 'feature2']]
        y = self.df_cls['target']
        result = self.engine.train_classification(X, y, model_name='Logistic Regression')
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('confusion_matrix', result)
        self.assertEqual(result['model_name'], 'Logistic Regression')

    def test_perform_pca(self):
        df = self.df_reg.copy()
        result = self.engine.perform_pca(df, ['feature1', 'feature2'], n_components=2)
        self.assertIn('pca_data', result)
        self.assertEqual(result['pca_data'].shape[1], 2)
        self.assertIn('explained_variance_ratio', result)

    def test_analyze_time_series(self):
        result = self.engine.analyze_time_series(self.df_ts, 'date', 'value')
        self.assertIn('series', result)
        self.assertIn('basic_stats', result)
        self.assertIn('trend', result['basic_stats'])

if __name__ == '__main__':
    unittest.main()
