"""
Enhanced Data Ingestion Component
Supports file upload, URL loading, SQL databases, sample datasets, and auto-detection.
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO
from utils.data_loader import load_data, load_sql
import os

# Sample datasets for quick start
SAMPLE_DATASETS = {
    "Iris (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Tips (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "Diamonds (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
    "Penguins (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "Flights (Time Series)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
    "Car Crashes (Analysis)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv",
    "MPG (Regression)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
}


def auto_detect_delimiter(content):
    """Auto-detect CSV delimiter."""
    delimiters = [',', ';', '\t', '|']
    sample = content[:5000] if len(content) > 5000 else content

    max_count = 0
    best_delimiter = ','

    for delim in delimiters:
        count = sample.count(delim)
        if count > max_count:
            max_count = count
            best_delimiter = delim

    return best_delimiter


def auto_detect_encoding(content_bytes):
    """Auto-detect file encoding."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

    for encoding in encodings:
        try:
            content_bytes.decode(encoding)
            return encoding
        except:
            continue

    return 'utf-8'


def load_from_url(url):
    """Load data from URL with auto-detection."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()

        # Determine file type from URL or content type
        if url.endswith('.csv') or 'csv' in content_type:
            encoding = auto_detect_encoding(response.content)
            content = response.content.decode(encoding)
            delimiter = auto_detect_delimiter(content)
            df = pd.read_csv(StringIO(content), sep=delimiter)

        elif url.endswith('.json') or 'json' in content_type:
            df = pd.read_json(BytesIO(response.content))

        elif url.endswith('.xlsx') or url.endswith('.xls') or 'excel' in content_type:
            df = pd.read_excel(BytesIO(response.content))

        elif url.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(response.content))

        else:
            # Try CSV as default
            encoding = auto_detect_encoding(response.content)
            content = response.content.decode(encoding)
            delimiter = auto_detect_delimiter(content)
            df = pd.read_csv(StringIO(content), sep=delimiter)

        return df, None

    except requests.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def auto_optimize_dtypes(df):
    """Automatically optimize data types to reduce memory usage."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Check if it's a date
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
                continue
            except:
                pass

            # Check if it should be categorical
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')

        elif col_type == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')

        elif col_type == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')

    return df


def get_data_summary(df):
    """Generate a quick data summary."""
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_total': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include='number').columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_cols': len(df.select_dtypes(include='datetime').columns),
    }
    return summary


def render():
    st.header("Data Ingestion")

    # Create tabs for different ingestion methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "File Upload",
        "URL / Web",
        "SQL Database",
        "Sample Datasets"
    ])

    with tab1:
        render_file_upload()

    with tab2:
        render_url_upload()

    with tab3:
        render_sql_connection()

    with tab4:
        render_sample_datasets()

    # Data preview and info
    if st.session_state['data'] is not None:
        render_data_preview()


def render_file_upload():
    st.info("Upload CSV, Excel (.xlsx, .xls, .xlsm), JSON, or Parquet files. No file size limits.")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (drag and drop supported)",
            type=['csv', 'xlsx', 'xls', 'xlsm', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )

    with col2:
        optimize_types = st.checkbox("Auto-optimize types", value=True, help="Automatically optimize data types for memory efficiency")

    if uploaded_file is not None:
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                encoding = st.selectbox("Encoding", ['auto', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252'])
            with col2:
                delimiter = st.selectbox("Delimiter (CSV)", ['auto', ',', ';', '\\t', '|'])

        if st.button("Load File", type="primary"):
            try:
                with st.spinner('Loading and analyzing data...'):
                    file_type = uploaded_file.name.split('.')[-1].lower()

                    if file_type == 'csv':
                        content = uploaded_file.read()

                        # Auto-detect encoding
                        if encoding == 'auto':
                            encoding = auto_detect_encoding(content)

                        content_str = content.decode(encoding)

                        # Auto-detect delimiter
                        if delimiter == 'auto':
                            delimiter = auto_detect_delimiter(content_str)
                        elif delimiter == '\\t':
                            delimiter = '\t'

                        df = pd.read_csv(StringIO(content_str), sep=delimiter)

                    elif 'xls' in file_type:
                        df = pd.read_excel(uploaded_file)

                    elif file_type == 'json':
                        df = pd.read_json(uploaded_file)

                    elif file_type == 'parquet':
                        df = pd.read_parquet(uploaded_file)
                    else:
                        df = load_data(uploaded_file, file_type)

                    if optimize_types:
                        df = auto_optimize_dtypes(df)

                    st.session_state['data'] = df
                    st.session_state['file_meta'] = {
                        'name': uploaded_file.name,
                        'size': uploaded_file.size,
                        'type': file_type,
                        'source': 'file_upload'
                    }
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")


def render_url_upload():
    st.info("Load data directly from a URL (CSV, JSON, Excel, or Parquet).")

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com/data.csv",
        help="Direct link to a data file"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        optimize_types = st.checkbox("Auto-optimize types", value=True, key="url_optimize")

    if url and st.button("Load from URL", type="primary"):
        with st.spinner("Fetching data from URL..."):
            df, error = load_from_url(url)

            if error:
                st.error(error)
            else:
                if optimize_types:
                    df = auto_optimize_dtypes(df)

                st.session_state['data'] = df
                st.session_state['file_meta'] = {
                    'name': url.split('/')[-1] or 'url_data',
                    'size': 0,
                    'type': 'url',
                    'source': 'url',
                    'url': url
                }
                st.success("Data loaded successfully from URL!")
                st.rerun()


def render_sql_connection():
    st.subheader("Connect to SQL Database")

    st.info("Connect to PostgreSQL, MySQL, SQLite, or other databases using SQLAlchemy connection strings.")

    # Connection string examples
    with st.expander("Connection String Examples"):
        st.code("""
# SQLite
sqlite:///path/to/database.db

# PostgreSQL
postgresql://user:password@host:5432/database

# MySQL
mysql+pymysql://user:password@host:3306/database

# Microsoft SQL Server
mssql+pyodbc://user:password@host/database?driver=ODBC+Driver+17+for+SQL+Server
        """)

    conn_string = st.text_input("Connection String", type="password")
    query = st.text_area("SQL Query", "SELECT * FROM my_table LIMIT 1000")

    if st.button("Load from SQL", type="primary"):
        if not conn_string or not query:
            st.error("Please provide both connection string and query.")
        else:
            try:
                with st.spinner("Executing query..."):
                    df = load_sql(conn_string, query)
                    st.session_state['data'] = df
                    st.session_state['file_meta'] = {
                        'name': 'SQL Query Result',
                        'size': 0,
                        'type': 'sql',
                        'source': 'sql'
                    }
                    st.success("Data loaded successfully from SQL!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading from SQL: {e}")


def render_sample_datasets():
    st.subheader("Quick Start with Sample Datasets")
    st.info("Load a sample dataset to explore the app's features.")

    # Display datasets in a grid
    cols = st.columns(2)

    for i, (name, url) in enumerate(SAMPLE_DATASETS.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{name}**")
                if st.button(f"Load", key=f"sample_{name}"):
                    with st.spinner(f"Loading {name}..."):
                        df, error = load_from_url(url)

                        if error:
                            st.error(error)
                        else:
                            df = auto_optimize_dtypes(df)
                            st.session_state['data'] = df
                            st.session_state['file_meta'] = {
                                'name': name,
                                'size': 0,
                                'type': 'sample',
                                'source': 'sample',
                                'url': url
                            }
                            st.success(f"Loaded {name} dataset!")
                            st.rerun()


def render_data_preview():
    st.markdown("---")
    st.subheader("Data Preview")

    df = st.session_state['data']
    meta = st.session_state.get('file_meta', {})

    # Summary metrics
    summary = get_data_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", summary['columns'])
    col3.metric("Missing Values", f"{summary['missing_total']:,}")
    col4.metric("Memory", f"{summary['memory_mb']:.2f} MB")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Numeric Cols", summary['numeric_cols'])
    col2.metric("Categorical Cols", summary['categorical_cols'])
    col3.metric("DateTime Cols", summary['datetime_cols'])
    col4.metric("Duplicates", summary['duplicates'])

    # Source info
    if meta.get('name'):
        st.caption(f"Source: {meta['name']} ({meta.get('source', 'unknown')})")

    # Data preview
    st.dataframe(df.head(100), use_container_width=True)

    # Column info
    with st.expander("Column Details"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(df[col].dtype) for col in df.columns],
            'Non-Null': [df[col].notna().sum() for col in df.columns],
            'Null': [df[col].isna().sum() for col in df.columns],
            'Unique': [df[col].nunique() for col in df.columns],
            'Sample': [str(df[col].dropna().iloc[0]) if df[col].notna().any() else 'N/A' for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    # Quick actions
    with st.expander("Quick Actions"):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Drop All Duplicates"):
                original_len = len(df)
                st.session_state['data'] = df.drop_duplicates()
                removed = original_len - len(st.session_state['data'])
                st.success(f"Removed {removed} duplicate rows")
                st.rerun()

        with col2:
            if st.button("Drop Rows with Any Missing"):
                original_len = len(df)
                st.session_state['data'] = df.dropna()
                removed = original_len - len(st.session_state['data'])
                st.success(f"Removed {removed} rows with missing values")
                st.rerun()

        with col3:
            if st.button("Reset Data"):
                st.session_state['data'] = None
                st.session_state['file_meta'] = None
                st.success("Data reset")
                st.rerun()
