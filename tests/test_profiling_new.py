import unittest
import pandas as pd
import numpy as np
from utils.data_processor import calculate_quality_score, detect_anomalies, detect_drift

class TestProfilingNew(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        # Add some missing values and duplicates
        self.df.loc[0, 'A'] = np.nan
        self.df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

    def test_quality_score(self):
        score, details = calculate_quality_score(self.df)
        print(f"Quality Score: {score}")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn('missing_score', details)

    def test_anomalies(self):
        df_anom, n = detect_anomalies(self.df)
        print(f"Anomalies detected: {n}")
        self.assertIn('is_anomaly', df_anom.columns)
        self.assertGreaterEqual(n, 0)

    def test_drift(self):
        report = detect_drift(self.df)
        print("Drift report generated.")
        self.assertIsInstance(report, dict)
        # Check if column A is in report
        self.assertIn('A', report)

if __name__ == '__main__':
    unittest.main()
