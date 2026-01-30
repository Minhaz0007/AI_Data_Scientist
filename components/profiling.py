"""
Enhanced Data Profiling Component
Comprehensive data quality analysis with automated scoring, recommendations, and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_processor import profile_data


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

            # Generate recommendations
            recommendations = generate_recommendations(df, quality_scores)

        # Store for later use
        st.session_state['quality_scores'] = quality_scores
        st.session_state['recommendations'] = recommendations

        # Data Quality Score Dashboard
        st.subheader("Data Quality Score")

        # Overall score with gauge
        col1, col2 = st.columns([1, 2])

        with col1:
            # Create gauge chart for overall score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_scores['overall'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Quality"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': quality_scores['overall']
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Quality dimensions breakdown
            dimensions = ['completeness', 'uniqueness', 'consistency', 'validity', 'accuracy']
            scores_list = [quality_scores[d] for d in dimensions]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=scores_list + [scores_list[0]],  # Close the polygon
                theta=['Completeness', 'Uniqueness', 'Consistency', 'Validity', 'Accuracy', 'Completeness'],
                fill='toself',
                line_color='blue'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=250,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Dimension scores as metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Completeness", f"{quality_scores['completeness']}%")
        col2.metric("Uniqueness", f"{quality_scores['uniqueness']}%")
        col3.metric("Consistency", f"{quality_scores['consistency']}%")
        col4.metric("Validity", f"{quality_scores['validity']}%")
        col5.metric("Accuracy", f"{quality_scores['accuracy']}%")

        st.markdown("---")

        # Automated Recommendations
        st.subheader("Automated Recommendations")

        if recommendations:
            # Group by type
            warnings = [r for r in recommendations if r['type'] == 'warning']
            infos = [r for r in recommendations if r['type'] == 'info']

            if warnings:
                st.markdown("#### Issues to Address")
                for rec in warnings:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.warning(f"**{rec['category']}**: {rec['message']}")
                        with col2:
                            if rec['action'] == 'remove_duplicates':
                                if st.button("Fix", key=f"fix_{rec['category']}"):
                                    st.session_state['data'] = df.drop_duplicates()
                                    st.success("Duplicates removed!")
                                    st.rerun()
                            elif rec['action'] == 'drop_columns':
                                if st.button("Drop", key=f"drop_{rec['category']}"):
                                    st.session_state['data'] = df.drop(columns=rec['columns'])
                                    st.success("Columns dropped!")
                                    st.rerun()

            if infos:
                with st.expander("Suggestions for Improvement", expanded=True):
                    for rec in infos:
                        st.info(f"**{rec['category']}**: {rec['message']}")

            # One-Click Auto-Fix
            st.markdown("---")
            st.subheader("One-Click Auto-Fix")

            if st.button("Apply All Safe Fixes", type="primary", help="Removes duplicates and imputes missing values"):
                df_fixed = df.copy()

                # Remove duplicates
                df_fixed = df_fixed.drop_duplicates()

                # Impute missing values
                for col in df_fixed.columns:
                    if df_fixed[col].isnull().sum() > 0:
                        if df_fixed[col].dtype in ['float64', 'int64']:
                            df_fixed[col] = df_fixed[col].fillna(df_fixed[col].median())
                        else:
                            df_fixed[col] = df_fixed[col].fillna(df_fixed[col].mode().iloc[0] if len(df_fixed[col].mode()) > 0 else 'Unknown')

                st.session_state['data'] = df_fixed
                st.success("Applied all safe fixes!")
                st.rerun()
        else:
            st.success("No issues detected! Your data quality is excellent.")

        st.markdown("---")

        # Overview Metrics
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{profile['rows']:,}")
        col2.metric("Columns", profile['columns'])
        col3.metric("Duplicates", f"{profile['duplicates']:,}")
        col4.metric("Missing Values", f"{profile['missing_total']:,}")

        # Missing Values Analysis
        st.subheader("Missing Values Analysis")
        missing_df = pd.DataFrame(list(profile['missing_by_col'].items()), columns=['Column', 'Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df) * 100).round(2)
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if not missing_df.empty:
            fig_missing = px.bar(
                missing_df,
                x='Column',
                y='Missing %',
                title='Missing Values by Column',
                color='Missing %',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values found!")

        # Numerical Statistics
        st.subheader("Numerical Statistics")
        if profile['numeric_stats']:
            stats_df = pd.DataFrame(profile['numeric_stats'])
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numerical columns found.")

        # Correlation Matrix
        st.subheader("Correlation Analysis")
        if profile['correlation']:
            corr_df = pd.DataFrame(profile['correlation'])
            fig = px.imshow(
                corr_df,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical data for correlation.")

        # Column Distribution Analysis
        st.subheader("Column Distribution Analysis")

        selected_col = st.selectbox("Select Column to Analyze", df.columns)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Statistics for {selected_col}**")
            col_stats = {
                'Data Type': str(df[selected_col].dtype),
                'Non-Null Count': df[selected_col].notna().sum(),
                'Null Count': df[selected_col].isna().sum(),
                'Unique Values': df[selected_col].nunique(),
                'Memory Usage': f"{df[selected_col].memory_usage(deep=True) / 1024:.2f} KB"
            }

            if pd.api.types.is_numeric_dtype(df[selected_col]):
                col_stats.update({
                    'Mean': f"{df[selected_col].mean():.4f}",
                    'Median': f"{df[selected_col].median():.4f}",
                    'Std Dev': f"{df[selected_col].std():.4f}",
                    'Min': f"{df[selected_col].min():.4f}",
                    'Max': f"{df[selected_col].max():.4f}"
                })

                # Anomaly detection
                anomalies = detect_anomalies_simple(df, selected_col)
                col_stats['Potential Outliers'] = f"{anomalies['count']} ({anomalies['percentage']:.1f}%)"

            for k, v in col_stats.items():
                st.write(f"**{k}:** {v}")

        with col2:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                fig = px.histogram(df, x=selected_col, marginal="box", title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                value_counts = df[selected_col].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title=f'Top 20 Values in {selected_col}',
                    labels={'x': selected_col, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Anomaly Detection Section
        st.markdown("---")
        st.subheader("Automated Anomaly Detection")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if numeric_cols:
            anomaly_col = st.selectbox("Select Column for Anomaly Detection", numeric_cols, key="anomaly_select")

            if st.button("Detect Anomalies"):
                anomalies = detect_anomalies_simple(df, anomaly_col)

                col1, col2, col3 = st.columns(3)
                col1.metric("Anomalies Found", anomalies['count'])
                col2.metric("Percentage", f"{anomalies['percentage']:.2f}%")
                col3.metric("Bounds", f"[{anomalies['lower_bound']:.2f}, {anomalies['upper_bound']:.2f}]")

                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[anomaly_col], name=anomaly_col, boxpoints='outliers'))
                fig.add_hline(y=anomalies['lower_bound'], line_dash="dash", line_color="red", annotation_text="Lower Bound")
                fig.add_hline(y=anomalies['upper_bound'], line_dash="dash", line_color="red", annotation_text="Upper Bound")
                fig.update_layout(title=f"Anomaly Detection: {anomaly_col}")
                st.plotly_chart(fig, use_container_width=True)

                if anomalies['count'] > 0:
                    with st.expander("View Anomalous Rows"):
                        st.dataframe(df.loc[anomalies['indices'][:50]], use_container_width=True)
        else:
            st.info("No numeric columns available for anomaly detection.")
