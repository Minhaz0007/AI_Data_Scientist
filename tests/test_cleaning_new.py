import unittest
import pandas as pd
import numpy as np
from utils.data_processor import get_cleaning_suggestions, auto_clean, get_transformation_suggestions

class TestCleaningNew(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 1, 2, np.nan],
            'B': ['x', 'x', 'x', 'x'], # Constant
            'C': [1, 100, 200, 300], # Skewed?
            'date_col': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']
        })
        self.df.loc[0, 'A'] = np.nan # More missing

    def test_suggestions(self):
        suggs = get_cleaning_suggestions(self.df)
        print("Cleaning suggestions:", suggs)
        self.assertTrue(any("Constant" in s or "constant" in s for s in suggs))

    def test_auto_clean(self):
        clean_df, log = auto_clean(self.df)
        print("Auto clean log:", log)
        # Constant col B should be gone
        self.assertNotIn('B', clean_df.columns)
        # Missing in A should be imputed
        self.assertEqual(clean_df['A'].isnull().sum(), 0)

    def test_transform_suggestions(self):
        suggs = get_transformation_suggestions(self.df)
        print("Transform suggestions:", suggs)
        # date_col should be suggested
        self.assertTrue(any(s['column'] == 'date_col' for s in suggs))

if __name__ == '__main__':
    unittest.main()
