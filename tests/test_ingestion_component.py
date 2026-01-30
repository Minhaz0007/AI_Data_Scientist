import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os
import importlib

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestIngestionComponent(unittest.TestCase):
    def setUp(self):
        # Mock streamlit using patch.dict on sys.modules
        self.patcher = patch.dict(sys.modules, {'streamlit': MagicMock()})
        self.patcher.start()

        # Mock st.file_uploader, st.tabs, etc.
        import streamlit as st
        st.file_uploader.return_value = None
        st.tabs.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        st.session_state = {}
        st.columns.return_value = (MagicMock(), MagicMock())

        # Ensure components.ingestion is (re)loaded with the mock
        if 'components.ingestion' in sys.modules:
            importlib.reload(sys.modules['components.ingestion'])
        else:
            import components.ingestion

        self.ingestion = sys.modules['components.ingestion']

    def tearDown(self):
        self.patcher.stop()
        # Clean up components.ingestion to avoid stale references
        if 'components.ingestion' in sys.modules:
            del sys.modules['components.ingestion']

    def test_load_from_url_csv(self):
        """Test load_from_url with CSV content using mocked requests."""
        url = "http://example.com/data.csv"
        csv_content = "col1,col2\n1,2\n3,4"

        mock_response = MagicMock()
        mock_response.content = csv_content.encode('utf-8')
        mock_response.headers = {'Content-Type': 'text/csv'}
        mock_response.raise_for_status.return_value = None

        # load_from_url uses requests.get, so we patch it.
        # Note: components.ingestion imports requests.
        # If it uses `import requests`, patching 'requests.get' works.
        with patch('requests.get', return_value=mock_response):
            # We access load_from_url from the module imported in setUp
            df, error = self.ingestion.load_from_url(url)

            if error:
                self.fail(f"load_from_url returned error: {error}")

            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertListEqual(list(df.columns), ['col1', 'col2'])
            print("load_from_url (CSV) passed verification.")

    def test_render_structure(self):
        """Verify render function runs without NameError (variable resolution)."""
        try:
            self.ingestion.render()
            print("render() executed structure without NameError.")
        except Exception as e:
            self.fail(f"render() raised exception: {e}")

    def test_auto_optimize_dtypes_exists(self):
        """Verify auto_optimize_dtypes is accessible and runs."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        try:
            df_opt = self.ingestion.auto_optimize_dtypes(df)
            self.assertIsInstance(df_opt, pd.DataFrame)
            print("auto_optimize_dtypes exists and runs.")
        except NameError as e:
            self.fail(f"auto_optimize_dtypes not found: {e}")
        except Exception as e:
            self.fail(f"auto_optimize_dtypes failed: {e}")

if __name__ == '__main__':
    unittest.main()
