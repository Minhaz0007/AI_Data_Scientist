import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

@st.cache_data
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

        # Extended statistics
        profile['skewness'] = numeric_df.skew().to_dict()
        profile['kurtosis'] = numeric_df.kurtosis().to_dict()
    else:
        profile['numeric_stats'] = {}
        profile['correlation'] = {}
        profile['skewness'] = {}
        profile['kurtosis'] = {}

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
    condition: 'equals', 'greater_than', 'less_than', 'contains', 'not_equals', 'between'
    """
    if condition == 'equals':
        return df[df[column] == value]
    elif condition == 'not_equals':
        return df[df[column] != value]
    elif condition == 'greater_than':
        return df[df[column] > float(value)]
    elif condition == 'less_than':
        return df[df[column] < float(value)]
    elif condition == 'contains':
        return df[df[column].astype(str).str.contains(value, na=False)]
    elif condition == 'between':
        # value should be "min,max"
        min_val, max_val = map(float, value.split(','))
        return df[(df[column] >= min_val) & (df[column] <= max_val)]
    return df

def group_and_aggregate(df, group_col, agg_col, method='mean'):
    """
    Group by a column and aggregate another.
    method: 'mean', 'sum', 'count', 'min', 'max', 'std', 'median'
    """
    grouped = df.groupby(group_col)[agg_col]
    if method == 'mean':
        return grouped.mean().reset_index()
    elif method == 'sum':
        return grouped.sum().reset_index()
    elif method == 'count':
        return grouped.count().reset_index()
    elif method == 'min':
        return grouped.min().reset_index()
    elif method == 'max':
        return grouped.max().reset_index()
    elif method == 'std':
        return grouped.std().reset_index()
    elif method == 'median':
        return grouped.median().reset_index()
    return grouped.mean().reset_index()

def perform_clustering(df, columns, n_clusters=3):
    """
    Perform KMeans clustering.
    Returns the dataframe with a new 'cluster' column.
    """
    df_model = df.dropna(subset=columns)
    X = df_model[columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_result = df.copy()
    df_result['cluster'] = np.nan
    df_result.loc[df_model.index, 'cluster'] = clusters

    return df_result

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.

    Args:
        df: pandas DataFrame
        column: column name
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: IQR multiplier (default 1.5) or z-score threshold (default 3)

    Returns:
        DataFrame with outlier flag column and outlier statistics
    """
    df_result = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        return df_result, {"error": "Column must be numeric"}

    col_data = df[column].dropna()

    if method == 'iqr':
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(col_data))
        outlier_indices = col_data.index[z_scores > threshold]
        outlier_mask = df.index.isin(outlier_indices)
        lower_bound = col_data.mean() - threshold * col_data.std()
        upper_bound = col_data.mean() + threshold * col_data.std()
    else:
        return df_result, {"error": "Unknown method"}

    df_result[f'{column}_is_outlier'] = outlier_mask

    stats_dict = {
        'method': method,
        'threshold': threshold,
        'lower_bound': round(lower_bound, 4),
        'upper_bound': round(upper_bound, 4),
        'outlier_count': int(outlier_mask.sum()),
        'outlier_percentage': round(outlier_mask.sum() / len(df) * 100, 2),
        'outlier_values': df.loc[outlier_mask, column].tolist()[:20]  # First 20 outliers
    }

    return df_result, stats_dict

def treat_outliers(df, column, method='clip', lower_bound=None, upper_bound=None):
    """
    Treat outliers in a column.

    Args:
        method: 'clip' (cap at bounds), 'remove', or 'median' (replace with median)
    """
    df_result = df.copy()

    if method == 'clip':
        if lower_bound is not None:
            df_result.loc[df_result[column] < lower_bound, column] = lower_bound
        if upper_bound is not None:
            df_result.loc[df_result[column] > upper_bound, column] = upper_bound
    elif method == 'remove':
        if lower_bound is not None:
            df_result = df_result[df_result[column] >= lower_bound]
        if upper_bound is not None:
            df_result = df_result[df_result[column] <= upper_bound]
    elif method == 'median':
        median_val = df_result[column].median()
        if lower_bound is not None:
            df_result.loc[df_result[column] < lower_bound, column] = median_val
        if upper_bound is not None:
            df_result.loc[df_result[column] > upper_bound, column] = median_val

    return df_result

def perform_hypothesis_test(df, column1, column2=None, test_type='ttest'):
    """
    Perform statistical hypothesis testing.

    Args:
        test_type: 'ttest' (t-test), 'chi2' (chi-square), 'anova', 'correlation'
    """
    results = {}

    try:
        if test_type == 'ttest':
            # Two-sample t-test
            if column2 is None:
                return {"error": "Need two columns for t-test"}
            data1 = df[column1].dropna()
            data2 = df[column2].dropna()
            stat, p_value = stats.ttest_ind(data1, data2)
            results = {
                'test': 'Independent Samples T-Test',
                'statistic': round(stat, 4),
                'p_value': round(p_value, 6),
                'significant_at_05': p_value < 0.05,
                'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
            }

        elif test_type == 'chi2':
            # Chi-square test for independence
            if column2 is None:
                return {"error": "Need two columns for chi-square test"}
            contingency_table = pd.crosstab(df[column1], df[column2])
            stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            results = {
                'test': 'Chi-Square Test of Independence',
                'statistic': round(stat, 4),
                'p_value': round(p_value, 6),
                'degrees_of_freedom': dof,
                'significant_at_05': p_value < 0.05,
                'interpretation': 'Variables are dependent' if p_value < 0.05 else 'Variables are independent'
            }

        elif test_type == 'correlation':
            # Pearson correlation test
            if column2 is None:
                return {"error": "Need two columns for correlation test"}
            data1 = df[column1].dropna()
            data2 = df[column2].dropna()
            # Align data
            common_idx = data1.index.intersection(data2.index)
            stat, p_value = stats.pearsonr(data1.loc[common_idx], data2.loc[common_idx])
            results = {
                'test': 'Pearson Correlation Test',
                'correlation': round(stat, 4),
                'p_value': round(p_value, 6),
                'significant_at_05': p_value < 0.05,
                'interpretation': f"{'Strong' if abs(stat) > 0.7 else 'Moderate' if abs(stat) > 0.4 else 'Weak'} {'positive' if stat > 0 else 'negative'} correlation"
            }

        elif test_type == 'normality':
            # Shapiro-Wilk test for normality
            data = df[column1].dropna()
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)  # Sample for large datasets
            stat, p_value = stats.shapiro(data)
            results = {
                'test': 'Shapiro-Wilk Normality Test',
                'statistic': round(stat, 4),
                'p_value': round(p_value, 6),
                'significant_at_05': p_value < 0.05,
                'interpretation': 'Data is NOT normally distributed' if p_value < 0.05 else 'Data appears normally distributed'
            }

    except Exception as e:
        results = {"error": str(e)}

    return results

def merge_datasets(df1, df2, on, how='inner'):
    """
    Merge two datasets.

    Args:
        on: column(s) to merge on
        how: 'inner', 'left', 'right', 'outer'
    """
    return pd.merge(df1, df2, on=on, how=how)

def pivot_table(df, index, columns, values, aggfunc='mean'):
    """
    Create a pivot table.
    """
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)

def unpivot_table(df, id_vars, value_vars=None, var_name='variable', value_name='value'):
    """
    Unpivot (melt) a dataframe.
    """
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)


# ============================================================================
# DATA TYPE CONVERSION
# ============================================================================

def convert_column_type(df, column, target_type, datetime_format=None, errors='coerce'):
    """
    Convert a column to a specified data type.

    Args:
        df: pandas DataFrame
        column: column name to convert
        target_type: 'int', 'float', 'string', 'datetime', 'boolean', 'category'
        datetime_format: format string for datetime parsing (optional)
        errors: 'coerce' (invalid become NaN), 'raise' (raise exception), 'ignore' (return original)

    Returns:
        DataFrame with converted column
    """
    df_result = df.copy()

    try:
        if target_type == 'int':
            df_result[column] = pd.to_numeric(df_result[column], errors=errors).astype('Int64')
        elif target_type == 'float':
            df_result[column] = pd.to_numeric(df_result[column], errors=errors)
        elif target_type == 'string':
            df_result[column] = df_result[column].astype(str)
        elif target_type == 'datetime':
            if datetime_format:
                df_result[column] = pd.to_datetime(df_result[column], format=datetime_format, errors=errors)
            else:
                df_result[column] = pd.to_datetime(df_result[column], errors=errors)
        elif target_type == 'boolean':
            # Handle common boolean representations
            bool_map = {
                'true': True, 'false': False,
                'yes': True, 'no': False,
                'y': True, 'n': False,
                '1': True, '0': False,
                1: True, 0: False,
                1.0: True, 0.0: False
            }
            df_result[column] = df_result[column].map(
                lambda x: bool_map.get(str(x).lower().strip(), x) if pd.notna(x) else x
            ).astype('boolean')
        elif target_type == 'category':
            df_result[column] = df_result[column].astype('category')
    except Exception as e:
        if errors == 'raise':
            raise e
        # If ignore, return original

    return df_result


def get_column_types(df):
    """
    Get a summary of column types in the dataframe.

    Returns:
        dict with column names as keys and type info as values
    """
    type_info = {}
    for col in df.columns:
        dtype = df[col].dtype
        type_info[col] = {
            'dtype': str(dtype),
            'nullable': df[col].isnull().any(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
    return type_info


# ============================================================================
# STRING CLEANING
# ============================================================================

def clean_string_column(df, column, operations):
    """
    Apply string cleaning operations to a column.

    Args:
        df: pandas DataFrame
        column: column name to clean
        operations: list of operations to apply. Options:
            - 'trim': Remove leading/trailing whitespace
            - 'lower': Convert to lowercase
            - 'upper': Convert to uppercase
            - 'title': Convert to title case
            - 'remove_special': Remove special characters (keep alphanumeric and spaces)
            - 'remove_digits': Remove all digits
            - 'remove_whitespace': Remove all whitespace
            - 'normalize_whitespace': Replace multiple spaces with single space

    Returns:
        DataFrame with cleaned column
    """
    df_result = df.copy()
    col_data = df_result[column].astype(str)

    for op in operations:
        if op == 'trim':
            col_data = col_data.str.strip()
        elif op == 'lower':
            col_data = col_data.str.lower()
        elif op == 'upper':
            col_data = col_data.str.upper()
        elif op == 'title':
            col_data = col_data.str.title()
        elif op == 'remove_special':
            col_data = col_data.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        elif op == 'remove_digits':
            col_data = col_data.str.replace(r'\d', '', regex=True)
        elif op == 'remove_whitespace':
            col_data = col_data.str.replace(r'\s', '', regex=True)
        elif op == 'normalize_whitespace':
            col_data = col_data.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Handle 'nan' strings that result from NaN values
    col_data = col_data.replace('nan', np.nan)
    df_result[column] = col_data

    return df_result


# ============================================================================
# VALUE MAPPING / REPLACEMENT
# ============================================================================

def map_values(df, column, mapping, default=None):
    """
    Map/replace values in a column based on a mapping dictionary.

    Args:
        df: pandas DataFrame
        column: column name
        mapping: dict of {old_value: new_value}
        default: value to use for unmapped values (None keeps original)

    Returns:
        DataFrame with mapped values
    """
    df_result = df.copy()

    if default is not None:
        df_result[column] = df_result[column].map(mapping).fillna(default)
    else:
        df_result[column] = df_result[column].replace(mapping)

    return df_result


def replace_values(df, column, find_value, replace_value, match_type='exact'):
    """
    Find and replace values in a column.

    Args:
        match_type: 'exact', 'contains', 'startswith', 'endswith', 'regex'
    """
    df_result = df.copy()

    if match_type == 'exact':
        df_result[column] = df_result[column].replace(find_value, replace_value)
    elif match_type == 'contains':
        mask = df_result[column].astype(str).str.contains(str(find_value), na=False)
        df_result.loc[mask, column] = replace_value
    elif match_type == 'startswith':
        mask = df_result[column].astype(str).str.startswith(str(find_value), na=False)
        df_result.loc[mask, column] = replace_value
    elif match_type == 'endswith':
        mask = df_result[column].astype(str).str.endswith(str(find_value), na=False)
        df_result.loc[mask, column] = replace_value
    elif match_type == 'regex':
        df_result[column] = df_result[column].astype(str).str.replace(find_value, replace_value, regex=True)

    return df_result


# ============================================================================
# COLUMN MANAGEMENT
# ============================================================================

def drop_columns(df, columns):
    """
    Drop specified columns from the dataframe.

    Args:
        columns: list of column names to drop
    """
    return df.drop(columns=columns, errors='ignore')


def rename_columns(df, rename_map):
    """
    Rename columns based on a mapping.

    Args:
        rename_map: dict of {old_name: new_name}
    """
    return df.rename(columns=rename_map)


def reorder_columns(df, column_order):
    """
    Reorder columns in the dataframe.

    Args:
        column_order: list of column names in desired order
    """
    # Include any columns not in the order list at the end
    remaining = [c for c in df.columns if c not in column_order]
    return df[column_order + remaining]


def split_column(df, column, delimiter, new_column_names=None, expand=True):
    """
    Split a column into multiple columns.

    Args:
        column: column to split
        delimiter: string to split on
        new_column_names: list of names for new columns (optional)
        expand: if True, create separate columns; if False, create a list column
    """
    df_result = df.copy()

    if expand:
        split_data = df_result[column].astype(str).str.split(delimiter, expand=True)
        if new_column_names:
            # Use provided names for as many columns as we have
            for i, name in enumerate(new_column_names):
                if i < len(split_data.columns):
                    df_result[name] = split_data[i]
        else:
            # Auto-generate column names
            for i in range(len(split_data.columns)):
                df_result[f'{column}_part{i+1}'] = split_data[i]
    else:
        df_result[f'{column}_split'] = df_result[column].astype(str).str.split(delimiter)

    return df_result


# ============================================================================
# ENHANCED DEDUPLICATION
# ============================================================================

def remove_duplicates_subset(df, subset=None, keep='first'):
    """
    Remove duplicates based on a subset of columns.

    Args:
        subset: list of column names to consider for duplicates (None = all columns)
        keep: 'first', 'last', or False (remove all duplicates)
    """
    return df.drop_duplicates(subset=subset, keep=keep)


def get_duplicate_rows(df, subset=None):
    """
    Get rows that are duplicates.

    Args:
        subset: list of column names to consider

    Returns:
        DataFrame containing only duplicate rows
    """
    mask = df.duplicated(subset=subset, keep=False)
    return df[mask]


# ============================================================================
# ROW MANAGEMENT
# ============================================================================

def remove_rows_by_index(df, indices):
    """
    Remove rows by their index values.
    """
    return df.drop(index=indices, errors='ignore')


def remove_rows_by_condition(df, column, condition, value):
    """
    Remove rows based on a condition.

    Args:
        condition: 'equals', 'not_equals', 'greater_than', 'less_than', 'contains', 'is_null', 'is_not_null'
    """
    if condition == 'equals':
        return df[df[column] != value]
    elif condition == 'not_equals':
        return df[df[column] == value]
    elif condition == 'greater_than':
        return df[df[column] <= float(value)]
    elif condition == 'less_than':
        return df[df[column] >= float(value)]
    elif condition == 'contains':
        return df[~df[column].astype(str).str.contains(str(value), na=False)]
    elif condition == 'is_null':
        return df[df[column].notna()]
    elif condition == 'is_not_null':
        return df[df[column].isna()]
    return df


# ============================================================================
# ENCODING
# ============================================================================

def label_encode(df, column, mapping=None):
    """
    Apply label encoding to a categorical column.

    Args:
        column: column to encode
        mapping: optional dict of {value: code}. If None, auto-generate.

    Returns:
        tuple of (encoded DataFrame, mapping dict)
    """
    df_result = df.copy()

    if mapping is None:
        unique_values = df_result[column].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_values, key=str))}

    df_result[f'{column}_encoded'] = df_result[column].map(mapping)

    return df_result, mapping


def one_hot_encode(df, columns, drop_first=False, prefix=None):
    """
    Apply one-hot encoding to categorical columns.

    Args:
        columns: list of columns to encode
        drop_first: whether to drop the first category (avoid multicollinearity)
        prefix: prefix for new column names (uses original column name if None)
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, prefix=prefix)


# ============================================================================
# ADVANCED PROFILING (QUALITY, ANOMALIES, DRIFT)
# ============================================================================

def calculate_quality_score(df):
    """
    Calculate a data quality score (0-100).
    Based on missing values, duplicates, and outliers.
    """
    if df.empty:
        return 0, {}

    n_rows = len(df)
    n_cols = len(df.columns)

    # 1. Missing Values Score
    total_cells = n_rows * n_cols
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    missing_score = max(0, 100 - (missing_ratio * 100))

    # 2. Duplicate Rows Score
    duplicate_rows = df.duplicated().sum()
    duplicate_ratio = duplicate_rows / n_rows if n_rows > 0 else 0
    duplicate_score = max(0, 100 - (duplicate_ratio * 100))

    # 3. Outlier Score (simplified estimate using IQR on sample numeric cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_ratios = []

    if len(numeric_cols) > 0:
        # Check max 5 random numeric columns to save time
        cols_to_check = numeric_cols if len(numeric_cols) <= 5 else np.random.choice(numeric_cols, 5, replace=False)

        for col in cols_to_check:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_ratios.append(outliers / n_rows)
            except:
                pass

    avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else 0
    # Penalize outliers less severely as they might be valid
    outlier_score = max(0, 100 - (avg_outlier_ratio * 50))

    # Weighted Average
    # Missing: 40%, Duplicates: 30%, Outliers: 30%
    final_score = (missing_score * 0.4) + (duplicate_score * 0.3) + (outlier_score * 0.3)

    details = {
        'missing_score': round(missing_score, 1),
        'duplicate_score': round(duplicate_score, 1),
        'outlier_score': round(outlier_score, 1),
        'missing_ratio': round(missing_ratio * 100, 2),
        'duplicate_ratio': round(duplicate_ratio * 100, 2),
        'outlier_ratio_est': round(avg_outlier_ratio * 100, 2)
    }

    return round(final_score, 1), details

def detect_anomalies(df, contamination=0.05):
    """
    Detect anomalies using Isolation Forest on numeric columns.
    Returns DataFrame with 'is_anomaly' column and anomaly score.
    """
    df_result = df.copy()
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if numeric_df.empty or len(numeric_df.columns) < 1:
        return df_result, 0

    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        # fit_predict returns -1 for outliers, 1 for inliers
        preds = iso.fit_predict(numeric_df)

        # Initialize with False
        df_result['is_anomaly'] = False
        df_result['anomaly_score'] = 0.0

        # Map back to original indices
        anomaly_indices = numeric_df.index[preds == -1]
        df_result.loc[anomaly_indices, 'is_anomaly'] = True

        # Get decision function scores (lower is more anomalous)
        scores = iso.decision_function(numeric_df)
        df_result.loc[numeric_df.index, 'anomaly_score'] = scores

        n_anomalies = len(anomaly_indices)
        return df_result, n_anomalies

    except Exception as e:
        # Fallback if isolation forest fails
        print(f"Anomaly detection failed: {e}")
        return df_result, 0

def detect_drift(df, split_ratio=0.5):
    """
    Detect data drift by comparing the first half of data vs the second half.
    Uses KS test for continuous and Chi-square for categorical (simplified).
    """
    n = len(df)
    if n < 50:
        return {}

    split_idx = int(n * split_ratio)
    df1 = df.iloc[:split_idx]
    df2 = df.iloc[split_idx:]

    drift_report = {}

    # Numeric Drift (KS Test)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            stat, p_value = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
            is_drifted = p_value < 0.05
            drift_report[col] = {
                'type': 'numeric',
                'p_value': round(p_value, 4),
                'drift_detected': is_drifted,
                'stat': round(stat, 3)
            }
        except:
            pass

    # Categorical Drift (Chi-Square is tricky if categories differ, using simple distribution diff)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        try:
            # Compare top 5 categories distribution
            dist1 = df1[col].value_counts(normalize=True).head(5)
            dist2 = df2[col].value_counts(normalize=True).head(5)

            # Align
            all_cats = set(dist1.index) | set(dist2.index)
            diff_sum = 0
            for cat in all_cats:
                p1 = dist1.get(cat, 0)
                p2 = dist2.get(cat, 0)
                diff_sum += abs(p1 - p2)

            # Arbitrary threshold: if sum of differences > 0.3, flag drift
            is_drifted = diff_sum > 0.3
            drift_report[col] = {
                'type': 'categorical',
                'diff_score': round(diff_sum, 3),
                'drift_detected': is_drifted
            }
        except:
            pass

    return drift_report

# ============================================================================
# AUTOMATED CLEANING & TRANSFORMATION SUGGESTIONS
# ============================================================================

def get_cleaning_suggestions(df):
    """
    Analyze the dataframe and return a list of cleaning suggestions.
    """
    suggestions = []

    # Missing Values
    missing_cols = df.columns[df.isnull().any()].tolist()
    for col in missing_cols:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0.5:
            suggestions.append(f"Drop column '{col}' (>50% missing)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            suggestions.append(f"Impute missing values in '{col}' with median")
        else:
            suggestions.append(f"Impute missing values in '{col}' with mode")

    # Duplicates
    if df.duplicated().sum() > 0:
        suggestions.append(f"Remove {df.duplicated().sum()} duplicate rows")

    # Constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            suggestions.append(f"Drop constant column '{col}'")

    return suggestions

def auto_clean(df):
    """
    Automatically apply common cleaning operations.
    Returns cleaned dataframe and a log of actions.
    """
    df_clean = df.copy()
    log = []

    # 1. Drop high missing columns (>50%)
    cols_to_drop = [col for col in df_clean.columns if df_clean[col].isnull().mean() > 0.5]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        log.append(f"Dropped columns with >50% missing: {', '.join(cols_to_drop)}")

    # 2. Drop constant columns
    const_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if const_cols:
        df_clean = df_clean.drop(columns=const_cols)
        log.append(f"Dropped constant columns: {', '.join(const_cols)}")

    # 3. Remove duplicates
    n_dups = df_clean.duplicated().sum()
    if n_dups > 0:
        df_clean = df_clean.drop_duplicates()
        log.append(f"Removed {n_dups} duplicate rows")

    # 4. Impute remaining missing
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(val)
                log.append(f"Imputed '{col}' with median ({val})")
            else:
                val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(val)
                log.append(f"Imputed '{col}' with mode ('{val}')")

    return df_clean, log

def get_transformation_suggestions(df):
    """
    Suggest transformations based on data distribution.
    """
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Check skewness
        try:
            skew = df[col].skew()
            if abs(skew) > 1:
                suggestions.append({
                    'column': col,
                    'type': 'Log Transform',
                    'reason': f"Highly skewed ({skew:.2f})",
                    'action': 'log'
                })
        except:
            pass

    # Check for date components
    # (Simple check if column name contains 'date' or 'time' but isn't datetime yet)
    for col in df.columns:
        if df[col].dtype == 'object' and ('date' in col.lower() or 'time' in col.lower()):
             suggestions.append({
                'column': col,
                'type': 'Convert to DateTime',
                'reason': "Column name suggests date/time",
                'action': 'to_datetime'
            })

    return suggestions
