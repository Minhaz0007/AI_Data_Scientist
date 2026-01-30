"""
Enhanced Feature Engineering Component
Advanced feature creation with auto-generation, intelligent suggestions, and comprehensive feature tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ml_engine import MLEngine


def analyze_feature_opportunities(df):
    """Analyze data and suggest feature engineering opportunities."""
    suggestions = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()

    # DateTime features
    for col in datetime_cols:
        suggestions.append({
            'type': 'datetime',
            'column': col,
            'priority': 'high',
            'description': f"Extract year, month, day, weekday, hour from '{col}'",
            'features': ['year', 'month', 'day', 'weekday', 'quarter', 'is_weekend']
        })

    # Also check object columns that might be dates
    for col in categorical_cols:
        try:
            sample = df[col].dropna().head(100)
            pd.to_datetime(sample, errors='raise')
            suggestions.append({
                'type': 'datetime_convert',
                'column': col,
                'priority': 'high',
                'description': f"Column '{col}' appears to be a date - convert and extract features"
            })
        except:
            pass

    # Polynomial features for numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'polynomial',
            'columns': numeric_cols[:5],
            'priority': 'medium',
            'description': "Create polynomial features (x^2, x*y) for numeric columns"
        })

    # Interaction features
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'interaction',
            'columns': numeric_cols[:5],
            'priority': 'medium',
            'description': "Create interaction features (multiplication, division, ratios)"
        })

    # Log transform for skewed distributions
    for col in numeric_cols:
        if df[col].min() > 0:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                suggestions.append({
                    'type': 'log_transform',
                    'column': col,
                    'priority': 'high',
                    'description': f"Log transform '{col}' (skewness={skewness:.2f})"
                })

    # Binning for continuous variables
    for col in numeric_cols:
        if df[col].nunique() > 20:
            suggestions.append({
                'type': 'binning',
                'column': col,
                'priority': 'low',
                'description': f"Create bins for '{col}' to convert to categorical"
            })

    # Target encoding for high-cardinality categoricals
    for col in categorical_cols:
        if 10 < df[col].nunique() < 100:
            suggestions.append({
                'type': 'encoding',
                'column': col,
                'priority': 'medium',
                'description': f"Encode '{col}' ({df[col].nunique()} categories)"
            })

    # Aggregation features
    if categorical_cols and numeric_cols:
        suggestions.append({
            'type': 'aggregation',
            'categorical': categorical_cols[0],
            'numeric': numeric_cols[0],
            'priority': 'medium',
            'description': f"Create group statistics of '{numeric_cols[0]}' by '{categorical_cols[0]}'"
        })

    return suggestions


def auto_generate_features(df, suggestions):
    """Automatically generate features based on suggestions."""
    df_new = df.copy()
    generated = []

    for suggestion in suggestions:
        try:
            if suggestion['type'] == 'datetime':
                col = suggestion['column']
                dt = df_new[col]
                df_new[f'{col}_year'] = dt.dt.year
                df_new[f'{col}_month'] = dt.dt.month
                df_new[f'{col}_day'] = dt.dt.day
                df_new[f'{col}_weekday'] = dt.dt.weekday
                df_new[f'{col}_quarter'] = dt.dt.quarter
                df_new[f'{col}_is_weekend'] = dt.dt.weekday.isin([5, 6]).astype(int)
                generated.append(f"DateTime features from '{col}'")

            elif suggestion['type'] == 'log_transform':
                col = suggestion['column']
                df_new[f'{col}_log'] = np.log1p(df_new[col])
                generated.append(f"Log transform of '{col}'")

            elif suggestion['type'] == 'interaction':
                cols = suggestion['columns'][:3]  # Limit to 3 columns
                for i, col1 in enumerate(cols):
                    for col2 in cols[i+1:]:
                        df_new[f'{col1}_x_{col2}'] = df_new[col1] * df_new[col2]
                        df_new[f'{col1}_div_{col2}'] = df_new[col1] / df_new[col2].replace(0, np.nan)
                generated.append("Interaction features")

            elif suggestion['type'] == 'aggregation':
                cat_col = suggestion['categorical']
                num_col = suggestion['numeric']
                group_mean = df_new.groupby(cat_col)[num_col].transform('mean')
                group_std = df_new.groupby(cat_col)[num_col].transform('std')
                df_new[f'{num_col}_by_{cat_col}_mean'] = group_mean
                df_new[f'{num_col}_by_{cat_col}_std'] = group_std
                df_new[f'{num_col}_diff_from_group_mean'] = df_new[num_col] - group_mean
                generated.append(f"Group statistics: '{num_col}' by '{cat_col}'")

        except Exception as e:
            continue

    return df_new, generated


def render():
    """Render the Feature Engineering page."""
    if st.session_state['data'] is None:
        st.warning("Please load data first in the Data Ingestion page.")
        return

    df = st.session_state['data']
    ml = MLEngine()

    st.markdown("Create new features, select important features, and transform your data for better model performance.")

    # Tabs for different operations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Auto-Generate",
        "Polynomial Features",
        "Interaction Features",
        "DateTime Features",
        "Binning",
        "Feature Selection"
    ])

    with tab1:
        render_auto_generate(df, ml)

    with tab2:
        render_polynomial_features(df, ml)

    with tab3:
        render_interaction_features(df, ml)

    with tab4:
        render_datetime_features(df, ml)

    with tab5:
        render_binning(df, ml)

    with tab6:
        render_feature_selection(df, ml)

    # Current Dataset Info
    st.markdown("---")
    st.subheader("Current Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categorical", len(df.select_dtypes(include=['object', 'category']).columns))

    with st.expander("View All Columns"):
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            st.write(f"- **{col}** ({dtype}) - {missing} missing values")


def render_auto_generate(df, ml):
    """Render the auto-generate features tab."""
    st.subheader("Intelligent Auto Feature Generation")
    st.markdown("Automatically detect and generate useful features from your data.")

    # Analyze opportunities
    suggestions = analyze_feature_opportunities(df)

    if not suggestions:
        st.success("No obvious feature engineering opportunities detected.")
        return

    # Group by priority
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    low_priority = [s for s in suggestions if s['priority'] == 'low']

    col1, col2, col3 = st.columns(3)
    col1.metric("High Priority", len(high_priority))
    col2.metric("Medium Priority", len(medium_priority))
    col3.metric("Low Priority", len(low_priority))

    st.markdown("---")

    # Display suggestions
    st.markdown("### Feature Generation Opportunities")

    if high_priority:
        st.markdown("#### High Priority")
        for s in high_priority:
            st.success(f"**{s['type'].replace('_', ' ').title()}** - {s['description']}")

    if medium_priority:
        st.markdown("#### Medium Priority")
        for s in medium_priority:
            st.info(f"**{s['type'].replace('_', ' ').title()}** - {s['description']}")

    if low_priority:
        with st.expander("Low Priority Suggestions"):
            for s in low_priority:
                st.write(f"**{s['type'].replace('_', ' ').title()}** - {s['description']}")

    st.markdown("---")

    # Auto-generate options
    st.markdown("### Generate Features")

    col1, col2 = st.columns(2)

    with col1:
        include_datetime = st.checkbox("DateTime features", value=True, disabled=not any(s['type'] == 'datetime' for s in suggestions))
        include_log = st.checkbox("Log transforms (skewed columns)", value=True, disabled=not any(s['type'] == 'log_transform' for s in suggestions))

    with col2:
        include_interactions = st.checkbox("Interaction features", value=False)
        include_aggregations = st.checkbox("Group aggregation features", value=False)

    # Filter suggestions based on selection
    selected_suggestions = []
    if include_datetime:
        selected_suggestions.extend([s for s in suggestions if s['type'] == 'datetime'])
    if include_log:
        selected_suggestions.extend([s for s in suggestions if s['type'] == 'log_transform'])
    if include_interactions:
        selected_suggestions.extend([s for s in suggestions if s['type'] == 'interaction'])
    if include_aggregations:
        selected_suggestions.extend([s for s in suggestions if s['type'] == 'aggregation'])

    st.write(f"**{len(selected_suggestions)} feature operations selected**")

    if st.button("Generate Features", type="primary", disabled=not selected_suggestions):
        with st.spinner("Generating features..."):
            df_new, generated = auto_generate_features(df, selected_suggestions)

            new_cols = [c for c in df_new.columns if c not in df.columns]
            st.success(f"Generated {len(new_cols)} new features!")

            st.markdown("**Generated:**")
            for item in generated:
                st.write(f"- {item}")

            st.markdown("**New Features Preview:**")
            st.dataframe(df_new[new_cols].head(10), use_container_width=True)

            st.session_state['feature_preview'] = df_new

    if 'feature_preview' in st.session_state:
        if st.button("Apply All Features to Dataset"):
            st.session_state['data'] = st.session_state['feature_preview']
            del st.session_state['feature_preview']
            st.success("Features added to dataset!")
            st.rerun()


def render_polynomial_features(df, ml):
    st.subheader("Polynomial Features")
    st.markdown("Create polynomial and interaction terms from numeric columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available.")
        return

    cols_for_poly = st.multiselect(
        "Select Columns for Polynomial Features",
        options=numeric_cols,
        max_selections=5,
        help="Select up to 5 columns (more may cause memory issues)"
    )

    degree = st.slider("Polynomial Degree", 2, 4, 2)

    if cols_for_poly and st.button("Create Polynomial Features", key="poly_btn"):
        with st.spinner("Creating polynomial features..."):
            try:
                df_new = ml.create_polynomial_features(df, cols_for_poly, degree=degree)

                new_cols = [c for c in df_new.columns if c not in df.columns]
                st.success(f"Created {len(new_cols)} new polynomial features!")

                st.write("**New Features Preview:**")
                st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                if st.button("Apply Changes", key="apply_poly"):
                    st.session_state['data'] = df_new
                    st.success("Features added to dataset!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_interaction_features(df, ml):
    st.subheader("Interaction Features")
    st.markdown("Create multiplication and division interactions between numeric columns.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available.")
        return

    cols_for_interact = st.multiselect(
        "Select Columns for Interactions",
        options=numeric_cols,
        max_selections=5,
        help="Select columns to create interaction features"
    )

    # Quick interaction builder
    st.markdown("---")
    st.markdown("### Quick Interaction Builder")

    col1, col2, col3 = st.columns(3)
    with col1:
        col_a = st.selectbox("Column A", numeric_cols, key="interact_a")
    with col2:
        operation = st.selectbox("Operation", ['*', '/', '+', '-'])
    with col3:
        col_b = st.selectbox("Column B", [c for c in numeric_cols if c != col_a], key="interact_b")

    new_name = st.text_input("New Column Name", f"{col_a}_{operation}_{col_b}")

    if st.button("Create Single Interaction"):
        df_new = df.copy()
        if operation == '*':
            df_new[new_name] = df_new[col_a] * df_new[col_b]
        elif operation == '/':
            df_new[new_name] = df_new[col_a] / df_new[col_b].replace(0, np.nan)
        elif operation == '+':
            df_new[new_name] = df_new[col_a] + df_new[col_b]
        elif operation == '-':
            df_new[new_name] = df_new[col_a] - df_new[col_b]

        st.session_state['data'] = df_new
        st.success(f"Created '{new_name}'!")
        st.rerun()

    st.markdown("---")

    if cols_for_interact and len(cols_for_interact) >= 2:
        if st.button("Create All Interaction Features", key="interact_btn"):
            with st.spinner("Creating interaction features..."):
                try:
                    df_new = ml.create_interaction_features(df, cols_for_interact)

                    new_cols = [c for c in df_new.columns if c not in df.columns]
                    st.success(f"Created {len(new_cols)} new interaction features!")

                    st.write("**New Features Preview:**")
                    st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                    if st.button("Apply Changes", key="apply_interact"):
                        st.session_state['data'] = df_new
                        st.success("Features added to dataset!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    elif cols_for_interact:
        st.info("Select at least 2 columns for interactions.")


def render_datetime_features(df, ml):
    st.subheader("DateTime Feature Extraction")
    st.markdown("Extract useful features from datetime columns (year, month, day, weekday, etc.).")

    # Find potential datetime columns
    datetime_cols = []
    potential_datetime = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(10), errors='raise')
                potential_datetime.append(col)
            except:
                pass

    if not datetime_cols and not potential_datetime:
        st.warning("No datetime columns detected.")

        # Manual conversion
        st.subheader("Convert Column to DateTime")
        all_cols = df.columns.tolist()
        col_to_convert = st.selectbox("Select Column to Convert", all_cols)

        if st.button("Convert to DateTime"):
            try:
                df_new = df.copy()
                df_new[col_to_convert] = pd.to_datetime(df_new[col_to_convert], errors='coerce')
                st.session_state['data'] = df_new
                st.success(f"Converted {col_to_convert} to datetime!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        return

    # Show potential datetime columns
    if potential_datetime:
        st.info(f"Potential datetime columns detected: {', '.join(potential_datetime)}")

        if st.button("Convert All Potential DateTime Columns"):
            df_new = df.copy()
            for col in potential_datetime:
                try:
                    df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                except:
                    pass
            st.session_state['data'] = df_new
            st.success("Converted potential datetime columns!")
            st.rerun()

    # Extract features from confirmed datetime columns
    if datetime_cols:
        datetime_col = st.selectbox("Select DateTime Column", options=datetime_cols)

        # Feature selection
        st.markdown("**Select features to extract:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            extract_year = st.checkbox("Year", value=True)
            extract_month = st.checkbox("Month", value=True)
        with col2:
            extract_day = st.checkbox("Day", value=True)
            extract_weekday = st.checkbox("Weekday", value=True)
        with col3:
            extract_quarter = st.checkbox("Quarter", value=True)
            extract_weekend = st.checkbox("Is Weekend", value=True)

        if st.button("Extract DateTime Features", type="primary"):
            with st.spinner("Extracting datetime features..."):
                try:
                    df_new = df.copy()
                    dt = df_new[datetime_col]

                    if extract_year:
                        df_new[f'{datetime_col}_year'] = dt.dt.year
                    if extract_month:
                        df_new[f'{datetime_col}_month'] = dt.dt.month
                    if extract_day:
                        df_new[f'{datetime_col}_day'] = dt.dt.day
                    if extract_weekday:
                        df_new[f'{datetime_col}_weekday'] = dt.dt.weekday
                    if extract_quarter:
                        df_new[f'{datetime_col}_quarter'] = dt.dt.quarter
                    if extract_weekend:
                        df_new[f'{datetime_col}_is_weekend'] = dt.dt.weekday.isin([5, 6]).astype(int)

                    new_cols = [c for c in df_new.columns if c not in df.columns]
                    st.success(f"Created {len(new_cols)} new datetime features!")

                    st.write("**New Features:**")
                    for col in new_cols:
                        st.write(f"- {col}")

                    st.write("**Preview:**")
                    st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                    if st.button("Apply Changes", key="apply_datetime"):
                        st.session_state['data'] = df_new
                        st.success("Features added to dataset!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")


def render_binning(df, ml):
    st.subheader("Feature Binning")
    st.markdown("Convert continuous variables into categorical bins.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available.")
        return

    col_to_bin = st.selectbox("Select Column to Bin", options=numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        n_bins = st.slider("Number of Bins", 2, 20, 5)

    with col2:
        strategy = st.selectbox(
            "Binning Strategy",
            options=['quantile', 'uniform', 'kmeans'],
            format_func=lambda x: {
                'quantile': 'Quantile (Equal Frequency)',
                'uniform': 'Uniform (Equal Width)',
                'kmeans': 'K-Means Clustering'
            }.get(x, x)
        )

    # Show current distribution
    fig = px.histogram(df, x=col_to_bin, nbins=30, title=f'Distribution of {col_to_bin}')
    st.plotly_chart(fig, use_container_width=True)

    # Custom bin edges option
    use_custom = st.checkbox("Use custom bin edges")
    if use_custom:
        edges_str = st.text_input("Enter bin edges (comma-separated)", placeholder="0, 10, 25, 50, 100")

    if st.button("Create Binned Feature", type="primary"):
        with st.spinner("Creating binned feature..."):
            try:
                df_new = df.copy()

                if use_custom and edges_str:
                    edges = [float(x.strip()) for x in edges_str.split(',')]
                    df_new[f'{col_to_bin}_binned'] = pd.cut(df_new[col_to_bin], bins=edges, labels=False)
                else:
                    df_new = ml.create_binned_features(df, col_to_bin, n_bins=n_bins, strategy=strategy)

                new_col = f'{col_to_bin}_binned'
                st.success(f"Created binned feature: {new_col}")

                # Show bin distribution
                bin_counts = df_new[new_col].value_counts().sort_index()
                fig = px.bar(x=bin_counts.index.astype(str), y=bin_counts.values,
                            title=f'Bin Distribution for {new_col}',
                            labels={'x': 'Bin', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Apply Changes", key="apply_bin"):
                    st.session_state['data'] = df_new
                    st.success("Feature added to dataset!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_feature_selection(df, ml):
    st.subheader("Feature Selection")
    st.markdown("Identify the most important features for your prediction task.")

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            options=all_cols,
            default=numeric_cols[:min(10, len(numeric_cols))]
        )

    with col2:
        remaining = [c for c in all_cols if c not in feature_cols]
        target_col = st.selectbox(
            "Select Target Column (y)",
            options=remaining if remaining else all_cols
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        selection_method = st.selectbox(
            "Selection Method",
            options=['mutual_info', 'f_score', 'rfe', 'correlation'],
            format_func=lambda x: {
                'mutual_info': 'Mutual Information',
                'f_score': 'F-Score (ANOVA)',
                'rfe': 'Recursive Feature Elimination',
                'correlation': 'Correlation with Target'
            }.get(x, x)
        )

    with col2:
        k_features = st.slider(
            "Number of Features to Select",
            1, min(len(feature_cols), 20) if feature_cols else 10,
            min(5, len(feature_cols)) if feature_cols else 5
        )

    # Quick correlation analysis
    if feature_cols and target_col and target_col in numeric_cols:
        with st.expander("Quick Correlation Analysis"):
            correlations = {}
            for col in feature_cols:
                if col in numeric_cols:
                    corr = df[[col, target_col]].corr().iloc[0, 1]
                    correlations[col] = abs(corr)

            if correlations:
                corr_df = pd.DataFrame([
                    {'Feature': k, 'Abs Correlation': v}
                    for k, v in sorted(correlations.items(), key=lambda x: -x[1])
                ])
                fig = px.bar(corr_df, x='Abs Correlation', y='Feature', orientation='h',
                            title='Absolute Correlation with Target')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    if feature_cols and target_col and st.button("Run Feature Selection", type="primary"):
        with st.spinner("Analyzing feature importance..."):
            try:
                X = df[feature_cols]
                y = df[target_col].dropna()

                # Align indices
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]

                if selection_method == 'correlation':
                    # Simple correlation-based selection
                    correlations = {}
                    for col in feature_cols:
                        try:
                            corr = X[col].corr(y)
                            correlations[col] = abs(corr) if not np.isnan(corr) else 0
                        except:
                            correlations[col] = 0

                    sorted_features = sorted(correlations.items(), key=lambda x: -x[1])
                    selected = [f[0] for f in sorted_features[:k_features]]

                    results = {
                        'selected_features': selected,
                        'feature_scores': correlations
                    }
                else:
                    results = ml.select_features(X, y, method=selection_method, k=k_features)

                st.success(f"Selected {len(results['selected_features'])} features!")

                # Feature Scores
                st.subheader("Feature Scores")

                scores_df = pd.DataFrame([
                    {'Feature': k, 'Score': v}
                    for k, v in results['feature_scores'].items()
                ])

                if selection_method == 'rfe':
                    scores_df = scores_df.sort_values('Score', ascending=True)
                else:
                    scores_df = scores_df.sort_values('Score', ascending=False)

                scores_df['Selected'] = scores_df['Feature'].isin(results['selected_features'])

                # Visualization
                fig = px.bar(
                    scores_df,
                    x='Score',
                    y='Feature',
                    orientation='h',
                    color='Selected',
                    title='Feature Importance Scores',
                    color_discrete_map={True: 'green', False: 'gray'}
                )
                fig.update_layout(height=max(400, len(feature_cols) * 25))
                st.plotly_chart(fig, use_container_width=True)

                # Selected Features
                st.subheader("Selected Features")
                for i, f in enumerate(results['selected_features'], 1):
                    score = results['feature_scores'].get(f, 0)
                    st.write(f"{i}. **{f}** (score: {score:.4f})")

                # Option to create subset
                if st.button("Create Dataset with Selected Features Only"):
                    selected_cols = results['selected_features'] + [target_col]
                    df_new = df[selected_cols].copy()
                    st.session_state['data'] = df_new
                    st.success("Dataset updated with selected features only!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
