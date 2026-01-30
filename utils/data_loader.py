import pandas as pd
import io
import sqlalchemy
import streamlit as st
import requests

def _fix_mixed_types(df):
    """Fix columns with mixed types to prevent PyArrow serialization errors."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types and convert to string
            try:
                # Try to infer better types first
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

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

def load_url(url, file_type='csv'):
    """Load data from a direct URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if file_type == 'csv':
            return load_csv(io.BytesIO(response.content))
        elif file_type in ['json']:
            return load_json(io.BytesIO(response.content))
        elif file_type in ['excel', 'xlsx', 'xls']:
            return load_excel(io.BytesIO(response.content))
        else:
            raise ValueError(f"Unsupported file type for URL: {file_type}")
    except Exception as e:
        raise ValueError(f"Error loading from URL: {e}")

def load_api(url, params=None, headers=None, json_key=None):
    """Load data from a REST API returning JSON."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if json_key:
            # Navigate nested JSON
            keys = json_key.split('.')
            for k in keys:
                data = data.get(k, data)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("API response is not a list or dictionary")

        return _fix_mixed_types(df)
    except Exception as e:
        raise ValueError(f"Error loading from API: {e}")

def load_sample(dataset_name):
    """Load a sample dataset."""
    sample_urls = {
        'titanic': 'https://raw.githubusercontent.com/datasciencedoct/data-science-guide/master/data/titanic.csv',
        'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
        'housing': 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv',
        'wine': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    }

    if dataset_name not in sample_urls:
        raise ValueError(f"Unknown sample dataset: {dataset_name}")

    try:
        if dataset_name == 'wine':
            # Wine is ; separated
            df = pd.read_csv(sample_urls[dataset_name], sep=';')
            return _fix_mixed_types(df)
        return load_url(sample_urls[dataset_name], 'csv')
    except Exception as e:
        raise ValueError(f"Error loading sample dataset: {e}")

@st.cache_data
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
