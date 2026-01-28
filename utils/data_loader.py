import pandas as pd
import io
import sqlalchemy

def load_csv(file):
    """Load CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

def load_excel(file):
    """Load Excel file into a pandas DataFrame."""
    try:
        return pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Error loading Excel: {e}")

def load_json(file):
    """Load JSON file into a pandas DataFrame."""
    try:
        return pd.read_json(file)
    except Exception as e:
        raise ValueError(f"Error loading JSON: {e}")

def load_sql(connection_string, query):
    """Load data from SQL database."""
    try:
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)
    except Exception as e:
        raise ValueError(f"Error loading SQL: {e}")

def load_data(file, file_type):
    """
    Load data from a file based on its type.

    Args:
        file: The file object (UploadedFile or path).
        file_type: str, one of 'csv', 'excel', 'json'.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if file_type == 'csv':
        return load_csv(file)
    elif file_type == 'excel' or file_type == 'xlsx' or file_type == 'xls':
        return load_excel(file)
    elif file_type == 'json':
        return load_json(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
