import unittest
import pandas as pd
import numpy as np
from utils.data_processor import profile_data, remove_duplicates, impute_missing, normalize_column_names, filter_data, group_and_aggregate, perform_clustering

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 1, 3, 4, 5],
            'C': ['x', 'y', 'z', 'x', 'y'],
            'D': [1.1, None, 3.3, 4.4, 5.5],
            'E F': [1, 2, 3, 4, 5]
        })
        # Add a duplicate row
        self.df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

    def test_profile_data(self):
        profile = profile_data(self.df)
        self.assertEqual(profile['rows'], 6)
        self.assertEqual(profile['duplicates'], 1)
        self.assertEqual(profile['missing_total'], 1)
        self.assertIn('A', profile['numeric_stats'])

    def test_remove_duplicates(self):
        cleaned = remove_duplicates(self.df)
        self.assertEqual(len(cleaned), 5)

    def test_impute_missing(self):
        # Mean imputation
        cleaned = impute_missing(self.df, ['D'], strategy='mean')
        self.assertFalse(cleaned['D'].isnull().any())

        # Constant imputation
        cleaned = impute_missing(self.df, ['D'], strategy='constant', fill_value=0)
        self.assertEqual(cleaned['D'].iloc[1], 0)

    def test_normalize_column_names(self):
        cleaned = normalize_column_names(self.df)
        self.assertIn('e_f', cleaned.columns)

    def test_filter_data(self):
        filtered = filter_data(self.df, 'A', 'greater_than', 3)
        self.assertTrue((filtered['A'] > 3).all())

    def test_group_and_aggregate(self):
        # A: 1, 2, 3, 4, 5, 1
        # C: x, y, z, x, y, x
        # Group by C, mean of A
        # x: (1 + 4 + 1) / 3 = 2
        agg = group_and_aggregate(self.df, 'C', 'A', 'mean')
        self.assertEqual(agg['x'], 2)

    def test_perform_clustering(self):
        # Clustering needs numeric data without NaN
        df_clean = impute_missing(self.df, ['D'], 'mean')
        df_clustered = perform_clustering(df_clean, ['A', 'B', 'D'], n_clusters=2)
        self.assertIn('cluster', df_clustered.columns)
        self.assertEqual(len(df_clustered['cluster'].unique()), 2)

if __name__ == '__main__':
    unittest.main()
