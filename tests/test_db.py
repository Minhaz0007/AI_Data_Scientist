import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock streamlit for testing db.py without UI side effects
sys.modules['streamlit'] = MagicMock()

# Do not mock sqlalchemy globally as it breaks other tests
# sys.modules['sqlalchemy'] = MagicMock()
# sys.modules['sqlalchemy.create_engine'] = MagicMock()

from utils.db import init_db, save_project

class TestDB(unittest.TestCase):

    @patch('utils.db.create_engine')
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@host/db"})
    def test_init_db(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = init_db()
        self.assertTrue(result)
        mock_conn.execute.assert_called()

    @patch('utils.db.create_engine')
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@host/db"})
    def test_save_project(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = save_project("Test", {}, "Insights")
        self.assertTrue(result)

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_var(self):
        result = init_db()
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
