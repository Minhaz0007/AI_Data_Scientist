import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

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
