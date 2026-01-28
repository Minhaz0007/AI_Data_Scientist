import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def profile_data(df):
    """
    Generate a comprehensive profile of the dataframe.
    """
    profile = {}

    # Basic info
    profile['rows'] = len(df)
    profile['columns'] = len(df.columns)
    profile['duplicates'] = df.duplicated().sum()
    profile['missing_total'] = df.isnull().sum().sum()

    # Column-wise info
    profile['missing_by_col'] = df.isnull().sum().to_dict()
    profile['dtypes'] = df.dtypes.astype(str).to_dict()

    # Numerical stats
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        profile['numeric_stats'] = numeric_df.describe().to_dict()
        profile['correlation'] = numeric_df.corr().to_dict()
    else:
        profile['numeric_stats'] = {}
        profile['correlation'] = {}

    return profile

def get_missing_summary(df):
    """Returns a summary of missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing.to_dict()

def remove_duplicates(df):
    """Removes duplicate rows from the dataframe."""
    return df.drop_duplicates()

def impute_missing(df, columns, strategy='mean', fill_value=None):
    """
    Impute missing values in specified columns.

    Args:
        df: pandas DataFrame
        columns: list of column names
        strategy: 'mean', 'median', 'mode', 'constant', 'drop'
        fill_value: value to use if strategy is 'constant'
    """
    df_clean = df.copy()
    for col in columns:
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif strategy == 'constant':
            df_clean[col] = df_clean[col].fillna(fill_value)

    return df_clean

def normalize_column_names(df):
    """Standardizes column names to snake_case."""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    return df_clean

def filter_data(df, column, condition, value):
    """
    Filter dataframe based on condition.
    condition: 'equals', 'greater_than', 'less_than', 'contains'
    """
    if condition == 'equals':
        return df[df[column] == value]
    elif condition == 'greater_than':
        return df[df[column] > float(value)]
    elif condition == 'less_than':
        return df[df[column] < float(value)]
    elif condition == 'contains':
        return df[df[column].astype(str).str.contains(value, na=False)]
    return df

def group_and_aggregate(df, group_col, agg_col, method='mean'):
    """
    Group by a column and aggregate another.
    method: 'mean', 'sum', 'count', 'min', 'max'
    """
    grouped = df.groupby(group_col)[agg_col]
    if method == 'mean':
        return grouped.mean()
    elif method == 'sum':
        return grouped.sum()
    elif method == 'count':
        return grouped.count()
    elif method == 'min':
        return grouped.min()
    elif method == 'max':
        return grouped.max()
    return grouped.mean()

def perform_clustering(df, columns, n_clusters=3):
    """
    Perform KMeans clustering.
    Returns the dataframe with a new 'cluster' column.
    """
    df_model = df.dropna(subset=columns)
    X = df_model[columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_result = df.copy()
    df_result['cluster'] = np.nan
    df_result.loc[df_model.index, 'cluster'] = clusters

    return df_result
