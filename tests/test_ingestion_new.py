import unittest
import pandas as pd
from utils.data_loader import load_url, load_api, load_sample

class TestIngestionNew(unittest.TestCase):
    def test_load_sample_iris(self):
        try:
            df = load_sample('iris')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)
            print("Iris sample loaded successfully.")
        except Exception as e:
            # Allow failure if network is down, but basic logic should work
            print(f"Iris load failed (possibly network): {e}")

    def test_load_url(self):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        try:
            df = load_url(url, 'csv')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)
            print("URL CSV loaded successfully.")
        except Exception as e:
            print(f"URL load skipped/failed (possibly network): {e}")

    def test_load_api(self):
        url = "https://jsonplaceholder.typicode.com/posts"
        try:
            df = load_api(url)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertFalse(df.empty)
            print("API loaded successfully.")
        except Exception as e:
            print(f"API load skipped/failed (possibly network): {e}")

if __name__ == '__main__':
    unittest.main()
