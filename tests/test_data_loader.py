import unittest
import pandas as pd
import io
import os
import sqlalchemy
from utils.data_loader import load_data, load_sql

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create dummy files
        self.df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        self.df.to_csv('test.csv', index=False)
        self.df.to_excel('test.xlsx', index=False)
        self.df.to_json('test.json')

    def tearDown(self):
        # Clean up
        for f in ['test.csv', 'test.xlsx', 'test.json']:
            if os.path.exists(f):
                os.remove(f)

    def test_load_csv(self):
        loaded_df = load_data('test.csv', 'csv')
        pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_load_excel(self):
        loaded_df = load_data('test.xlsx', 'excel')
        # Excel might read as object or int depending on engine, checking values
        pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_load_json(self):
        loaded_df = load_data('test.json', 'json')
        pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_load_sql_file(self):
        # Use a file based sqlite for testing
        db_file = 'test.db'
        engine = sqlalchemy.create_engine(f'sqlite:///{db_file}')
        self.df.to_sql('test_table', engine, index=False)
        engine.dispose()

        try:
            loaded_df = load_sql(f'sqlite:///{db_file}', 'SELECT * FROM test_table')
            pd.testing.assert_frame_equal(self.df, loaded_df)
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)

if __name__ == '__main__':
    unittest.main()
