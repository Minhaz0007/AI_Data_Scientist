"""
Enhanced Data Profiling Component
Comprehensive data quality analysis with automated scoring, recommendations, and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processor import profile_data, calculate_quality_score, detect_anomalies, detect_drift


def calculate_data_quality_score(df):
    """Calculate an overall data quality score (0-100)."""
    scores = {}
    weights = {
        'completeness': 0.30,
        'uniqueness': 0.20,
        'consistency': 0.20,
        'validity': 0.15,
        'accuracy': 0.15
    }

    # Completeness: % of non-null values
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    scores['completeness'] = min(completeness, 100)

    # Uniqueness: % of unique rows (penalize high duplicates)
    duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
    scores['uniqueness'] = (1 - duplicate_ratio) * 100

    # Consistency: Check for consistent data types and patterns
    consistency_score = 100
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types or patterns
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Check if numeric strings mixed with text
                numeric_count = non_null.apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit()).sum()
                if 0 < numeric_count < len(non_null):
                    consistency_score -= 5
    scores['consistency'] = max(consistency_score, 0)

    # Validity: Check for reasonable values
    validity_score = 100
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        # Check for extreme outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
        if len(df) > 0:
            outlier_ratio = outlier_count / len(df)
            validity_score -= outlier_ratio * 20
    scores['validity'] = max(validity_score, 0)

    # Accuracy: Proxy based on data characteristics
    accuracy_score = 100
    # Check for suspiciously uniform distributions
    for col in numeric_cols:
        if df[col].std() == 0 and len(df) > 10:
            accuracy_score -= 10
    scores['accuracy'] = max(accuracy_score, 0)

    # Calculate weighted overall score
    overall = sum(scores[k] * weights[k] for k in weights)

    return {
        'overall': round(overall, 1),
        'completeness': round(scores['completeness'], 1),
        'uniqueness': round(scores['uniqueness'], 1),
        'consistency': round(scores['consistency'], 1),
        'validity': round(scores['validity'], 1),
        'accuracy': round(scores['accuracy'], 1)
    }


def generate_recommendations(df, quality_scores):
    """Generate automated recommendations based on data quality analysis."""
    recommendations = []

    # Completeness recommendations
    missing_cols = df.columns[df.isnull().sum() > 0].tolist()
    if missing_cols:
        high_missing = [col for col in missing_cols if df[col].isnull().sum() / len(df) > 0.5]
        low_missing = [col for col in missing_cols if df[col].isnull().sum() / len(df) <= 0.5]

        if high_missing:
            recommendations.append({
                'type': 'warning',
                'category': 'Missing Values',
                'message': f"Columns with >50% missing: {', '.join(high_missing)}. Consider dropping these columns.",
                'action': 'drop_columns',
                'columns': high_missing
            })

        if low_missing:
            for col in low_missing:
                if df[col].dtype in ['float64', 'int64']:
                    recommendations.append({
                        'type': 'info',
                        'category': 'Missing Values',
                        'message': f"Column '{col}' has missing values. Recommended: Impute with median.",
                        'action': 'impute',
                        'column': col,
                        'strategy': 'median'
                    })
                else:
                    recommendations.append({
                        'type': 'info',
                        'category': 'Missing Values',
                        'message': f"Column '{col}' has missing values. Recommended: Impute with mode.",
                        'action': 'impute',
                        'column': col,
                        'strategy': 'mode'
                    })

    # Duplicate recommendations
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        recommendations.append({
            'type': 'warning',
            'category': 'Duplicates',
            'message': f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%). Recommended: Remove duplicates.",
            'action': 'remove_duplicates'
        })

    # Data type recommendations
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Check if should be numeric
                try:
                    pd.to_numeric(non_null)
                    recommendations.append({
                        'type': 'info',
                        'category': 'Data Types',
                        'message': f"Column '{col}' appears to be numeric but stored as text. Consider converting.",
                        'action': 'convert_type',
                        'column': col,
                        'target_type': 'numeric'
                    })
                except:
                    pass

                # Check if should be datetime
                try:
                    pd.to_datetime(non_null.head(100))
                    recommendations.append({
                        'type': 'info',
                        'category': 'Data Types',
                        'message': f"Column '{col}' appears to be a date. Consider converting to datetime.",
                        'action': 'convert_type',
                        'column': col,
                        'target_type': 'datetime'
                    })
                except:
                    pass

                # Check if should be categorical
                if df[col].nunique() / len(df) < 0.05 and df[col].nunique() < 50:
                    recommendations.append({
                        'type': 'info',
                        'category': 'Data Types',
                        'message': f"Column '{col}' has few unique values ({df[col].nunique()}). Consider categorical encoding.",
                        'action': 'encode',
                        'column': col
                    })

    # Outlier recommendations
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > len(df) * 0.05:  # More than 5% outliers
            recommendations.append({
                'type': 'warning',
                'category': 'Outliers',
                'message': f"Column '{col}' has {outliers} potential outliers ({outliers/len(df)*100:.1f}%).",
                'action': 'handle_outliers',
                'column': col
            })

    # Correlation recommendations
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        if high_corr_pairs:
            pairs_str = ', '.join([f"({a}, {b})" for a, b in high_corr_pairs[:3]])
            recommendations.append({
                'type': 'info',
                'category': 'Multicollinearity',
                'message': f"Highly correlated columns detected: {pairs_str}. Consider removing one from each pair.",
                'action': 'review_correlations'
            })

    return recommendations


def detect_anomalies_simple(df, col):
    """Simple anomaly detection for a numeric column."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    return {
        'count': len(anomalies),
        'percentage': len(anomalies) / len(df) * 100 if len(df) > 0 else 0,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'indices': anomalies.index.tolist()[:100]
    }


def render():
    st.header("Data Profiling")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset in the 'Data Ingestion' page first.")
        return

    df = st.session_state['data']

    # Auto-generate profile on load
    if st.button("Generate Comprehensive Profile", type="primary") or 'profile_generated' in st.session_state:
        st.session_state['profile_generated'] = True

        with st.spinner("Analyzing data quality..."):
            # Calculate quality scores
            quality_scores = calculate_data_quality_score(df)

            # Generate profile
            profile = profile_data(df)
            quality_score, quality_details = calculate_quality_score(df)

            # --- Overview & Quality Score ---
            st.subheader("Overview & Quality")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Rows", profile['rows'])
            col2.metric("Columns", profile['columns'])
            col3.metric("Duplicates", profile['duplicates'])
            col4.metric("Missing Values", profile['missing_total'])

            # Quality Score Gauge
            col5.metric("Quality Score", f"{quality_score}/100", delta="High" if quality_score > 80 else "Low")

            with st.expander("Quality Score Details"):
                st.write(quality_details)

            # --- Tabs ---
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs(["Statistics", "Missing & Correlations", "Anomaly Detection", "Data Drift"])

            with tab1:
                # Numerical Stats
                st.subheader("Numerical Statistics")
                if profile['numeric_stats']:
                    st.dataframe(pd.DataFrame(profile['numeric_stats']))
                else:
                    st.info("No numerical columns found.")

                # Column Details
                st.subheader("Column Distribution")
                selected_col = st.selectbox("Select Column to visualize", df.columns)

                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    fig = px.histogram(df, x=selected_col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.bar(df[selected_col].value_counts().reset_index(), x=selected_col, y='count')
                    fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Missing Values
                st.subheader("Missing Values by Column")
                missing_df = pd.DataFrame(list(profile['missing_by_col'].items()), columns=['Column', 'Missing Count'])
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if not missing_df.empty:
                    st.bar_chart(missing_df.set_index('Column'))
                else:
                    st.success("No missing values found.")

                # Correlation Matrix
                st.subheader("Correlation Matrix")
                if profile['correlation']:
                    corr_df = pd.DataFrame(profile['correlation'])
                    fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numerical data for correlation.")

            with tab3:
                st.subheader("Automated Anomaly Detection")
                st.write("Uses Isolation Forest to detect anomalies in numerical data.")

                # We run this on demand inside the tab
                if st.button("Run Anomaly Detection", key="run_anomaly"):
                    with st.spinner("Detecting anomalies..."):
                        df_anom, n_anomalies = detect_anomalies(df)
                        if n_anomalies > 0:
                            st.warning(f"Detected {n_anomalies} anomalies ({round(n_anomalies/len(df)*100, 2)}%)")
                            st.write("Top Anomalies (sorted by anomaly score):")
                            # Show anomalies sorted by score
                            st.dataframe(df_anom[df_anom['is_anomaly']].sort_values('anomaly_score').head(20))

                            # Visualization
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            if len(num_cols) >= 2:
                                fig = px.scatter(df_anom, x=num_cols[0], y=num_cols[1], color='is_anomaly',
                                                title=f"Anomalies in {num_cols[0]} vs {num_cols[1]}",
                                                color_discrete_map={False: 'blue', True: 'red'})
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("No significant anomalies detected.")

            with tab4:
                st.subheader("Data Drift Analysis")
                st.info("Compares the first half of the dataset to the second half to check for distribution shifts.")

                drift_report = detect_drift(df)

                if drift_report:
                    drifted_cols = [col for col, data in drift_report.items() if data['drift_detected']]
                    if drifted_cols:
                        st.error(f"Drift detected in columns: {', '.join(drifted_cols)}")
                    else:
                        st.success("No significant drift detected.")

                    # Show details
                    st.write("Drift Metrics:")
                    st.json(drift_report)
                else:
                    st.info("Dataset too small for drift detection (need > 50 rows).")
