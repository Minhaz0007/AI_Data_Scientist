import unittest
import pandas as pd
import numpy as np
from components.visualization import suggest_charts
import plotly.graph_objects as go

class TestVizDashboard(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.choice(['X', 'Y', 'Z'], 100),
            'date': pd.date_range('2021-01-01', periods=100)
        })

    def test_suggest_charts(self):
        suggestions = suggest_charts(self.df)
        print("Suggestions keys:", suggestions.keys())
        # Should have trend since we have date
        self.assertTrue(any('Trend' in k for k in suggestions.keys()))
        # Should have correlation
        self.assertIn('Correlation Heatmap', suggestions)

if __name__ == '__main__':
    unittest.main()
