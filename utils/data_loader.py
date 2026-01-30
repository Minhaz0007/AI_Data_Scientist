import pandas as pd
import io
import sqlalchemy
import streamlit as st

def _fix_mixed_types(df):
    """Fix columns with mixed types to prevent PyArrow serialization errors."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types and convert to string
            try:
                # Try to infer better types first
                df[col] = pd.to_numeric(df[col], errors='ignore')
                if df[col].dtype == 'object':
                    # Still object, convert to string to avoid PyArrow issues
                    df[col] = df[col].astype(str).replace('nan', pd.NA)
            except Exception:
                df[col] = df[col].astype(str).replace('nan', pd.NA)
    return df

def load_csv(file):
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file, low_memory=False)
        return _fix_mixed_types(df)
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")

def load_excel(file):
    """Load Excel file into a pandas DataFrame."""
    try:
        df = pd.read_excel(file)
        return _fix_mixed_types(df)
    except Exception as e:
        raise ValueError(f"Error loading Excel: {e}")

def load_json(file):
    """Load JSON file into a pandas DataFrame."""
    try:
        df = pd.read_json(file)
        return _fix_mixed_types(df)
    except Exception as e:
        raise ValueError(f"Error loading JSON: {e}")

def load_sql(connection_string, query):
    """Load data from SQL database."""
    try:
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        return _fix_mixed_types(df)
    except Exception as e:
        raise ValueError(f"Error loading SQL: {e}")

@st.cache_data(show_spinner=False)
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
