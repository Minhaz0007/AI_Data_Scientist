"""
Advanced Analysis Component
Includes Dimensionality Reduction (PCA, t-SNE), Anomaly Detection, and Text Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.ml_engine import MLEngine


def render():
    """Render the Advanced Analysis page."""
    if st.session_state['data'] is None:
        st.warning("Please load data first in the Data Ingestion page.")
        return

    df = st.session_state['data']
    ml = MLEngine()

    st.markdown("""
    Advanced analytical techniques including dimensionality reduction, anomaly detection, and text analysis.
    """)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "PCA",
        "t-SNE",
        "Anomaly Detection",
        "Text Analysis"
    ])

    with tab1:
        render_pca(df, ml)

    with tab2:
        render_tsne(df, ml)

    with tab3:
        render_anomaly_detection(df, ml)

    with tab4:
        render_text_analysis(df, ml)


def render_pca(df, ml):
    """Principal Component Analysis section."""
    st.subheader("Principal Component Analysis (PCA)")
    st.markdown("Reduce dimensionality while preserving variance. Great for visualization and feature reduction.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA.")
        return

    # Column selection
    selected_cols = st.multiselect(
        "Select Columns for PCA",
        options=numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )

    if len(selected_cols) < 2:
        st.info("Select at least 2 columns.")
        return

    col1, col2 = st.columns(2)

    with col1:
        n_components = st.slider(
            "Number of Components",
            2, min(len(selected_cols), 10),
            min(3, len(selected_cols))
        )

    with col2:
        # Color by column (optional)
        color_options = ['None'] + df.columns.tolist()
        color_by = st.selectbox("Color Points By", options=color_options)

    if st.button("Run PCA", type="primary", key="pca_btn"):
        with st.spinner("Running PCA..."):
            try:
                results = ml.perform_pca(df, selected_cols, n_components=n_components)

                # Explained Variance
                st.subheader("Explained Variance")

                exp_var = results['explained_variance_ratio']
                cum_var = results['cumulative_variance']

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(
                        x=[f'PC{i+1}' for i in range(len(exp_var))],
                        y=[v * 100 for v in exp_var],
                        name='Individual Variance %',
                        marker_color='blue'
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=[f'PC{i+1}' for i in range(len(cum_var))],
                        y=[v * 100 for v in cum_var],
                        name='Cumulative Variance %',
                        mode='lines+markers',
                        marker_color='red'
                    ),
                    secondary_y=True
                )

                fig.update_layout(title='Explained Variance by Principal Component')
                fig.update_yaxes(title_text="Individual %", secondary_y=False)
                fig.update_yaxes(title_text="Cumulative %", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)

                # PCA Scatter Plot
                st.subheader("PCA Visualization")

                pca_data = results['pca_data']

                if color_by != 'None':
                    # Align indices
                    color_values = df.loc[pca_data.index, color_by]
                else:
                    color_values = None

                if n_components >= 3:
                    fig_3d = px.scatter_3d(
                        pca_data,
                        x='PC1', y='PC2', z='PC3',
                        color=color_values,
                        title='3D PCA Visualization',
                        opacity=0.7
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                fig_2d = px.scatter(
                    pca_data,
                    x='PC1', y='PC2',
                    color=color_values,
                    title='2D PCA Visualization (PC1 vs PC2)',
                    opacity=0.7
                )
                st.plotly_chart(fig_2d, use_container_width=True)

                # Component Loadings
                st.subheader("Component Loadings")
                st.markdown("Shows how much each original feature contributes to each principal component.")

                loadings = results['components']
                fig_loadings = px.imshow(
                    loadings,
                    labels=dict(x="Original Features", y="Principal Components", color="Loading"),
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_loadings.update_layout(title='PCA Component Loadings')
                st.plotly_chart(fig_loadings, use_container_width=True)

                # Store in session state
                st.session_state['pca_result'] = results

                # Option to add PCA columns to dataset
                if st.button("Add PCA Components to Dataset"):
                    df_new = df.copy()
                    for col in pca_data.columns:
                        df_new.loc[pca_data.index, col] = pca_data[col]
                    st.session_state['data'] = df_new
                    st.success("PCA components added to dataset!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_tsne(df, ml):
    """t-SNE visualization section."""
    st.subheader("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    st.markdown("Non-linear dimensionality reduction for visualization. Great for finding clusters.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for t-SNE.")
        return

    selected_cols = st.multiselect(
        "Select Columns for t-SNE",
        options=numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))],
        key="tsne_cols"
    )

    if len(selected_cols) < 2:
        st.info("Select at least 2 columns.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        perplexity = st.slider(
            "Perplexity",
            5, 50, 30,
            help="Balance between local and global aspects of data"
        )

    with col2:
        n_iter = st.slider(
            "Iterations",
            250, 2000, 1000,
            step=250,
            help="More iterations = better convergence but slower"
        )

    with col3:
        color_options = ['None'] + df.columns.tolist()
        color_by = st.selectbox("Color Points By", options=color_options, key="tsne_color")

    if len(df) > 5000:
        st.warning("Dataset has >5000 rows. t-SNE will use a random sample of 5000 points.")

    if st.button("Run t-SNE", type="primary", key="tsne_btn"):
        with st.spinner("Running t-SNE (this may take a moment)..."):
            try:
                results = ml.perform_tsne(
                    df, selected_cols,
                    n_components=2,
                    perplexity=perplexity,
                    n_iter=n_iter
                )

                tsne_data = results['tsne_data']

                # Color values
                if color_by != 'None':
                    color_values = df.loc[tsne_data.index, color_by]
                else:
                    color_values = None

                # t-SNE Plot
                fig = px.scatter(
                    tsne_data,
                    x='tSNE1', y='tSNE2',
                    color=color_values,
                    title='t-SNE Visualization',
                    opacity=0.7
                )
                fig.update_layout(
                    xaxis_title='t-SNE Dimension 1',
                    yaxis_title='t-SNE Dimension 2'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"t-SNE completed on {len(tsne_data)} points!")

                # Store result
                st.session_state['tsne_result'] = results

                # Option to add t-SNE columns
                if st.button("Add t-SNE Components to Dataset"):
                    df_new = df.copy()
                    for col in tsne_data.columns:
                        df_new[col] = np.nan
                        df_new.loc[tsne_data.index, col] = tsne_data[col]
                    st.session_state['data'] = df_new
                    st.success("t-SNE components added to dataset!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_anomaly_detection(df, ml):
    """Anomaly detection section."""
    st.subheader("Anomaly Detection")
    st.markdown("Detect unusual data points using machine learning algorithms.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 1:
        st.warning("Need at least 1 numeric column for anomaly detection.")
        return

    selected_cols = st.multiselect(
        "Select Columns for Anomaly Detection",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        key="anomaly_cols"
    )

    if not selected_cols:
        st.info("Select at least 1 column.")
        return

    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Detection Method",
            options=['isolation_forest', 'lof'],
            format_func=lambda x: {
                'isolation_forest': 'Isolation Forest',
                'lof': 'Local Outlier Factor (LOF)'
            }.get(x, x)
        )

    with col2:
        contamination = st.slider(
            "Expected Anomaly Proportion",
            0.01, 0.3, 0.1, 0.01,
            help="Expected proportion of anomalies in the data"
        )

    if st.button("Detect Anomalies", type="primary", key="anomaly_btn"):
        with st.spinner("Detecting anomalies..."):
            try:
                results = ml.detect_anomalies(
                    df, selected_cols,
                    method=method,
                    contamination=contamination
                )

                # Summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Points", len(df))
                col2.metric("Anomalies Detected", results['anomaly_count'])
                col3.metric("Anomaly %", f"{results['anomaly_percentage']}%")

                # Visualization
                st.subheader("Anomaly Visualization")

                df_result = results['data_with_anomalies']

                if len(selected_cols) >= 2:
                    fig = px.scatter(
                        df_result,
                        x=selected_cols[0],
                        y=selected_cols[1],
                        color='is_anomaly',
                        color_discrete_map={True: 'red', False: 'blue'},
                        title='Anomaly Detection Results',
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 3D if 3+ columns
                if len(selected_cols) >= 3:
                    fig_3d = px.scatter_3d(
                        df_result,
                        x=selected_cols[0],
                        y=selected_cols[1],
                        z=selected_cols[2],
                        color='is_anomaly',
                        color_discrete_map={True: 'red', False: 'blue'},
                        title='3D Anomaly Visualization',
                        opacity=0.7
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # Show anomalous rows
                st.subheader("Anomalous Data Points")
                anomaly_df = df_result[df_result['is_anomaly']]
                st.dataframe(anomaly_df.head(50), use_container_width=True)

                # Options
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Add Anomaly Flag to Dataset"):
                        st.session_state['data'] = df_result
                        st.success("Anomaly flag added!")
                        st.rerun()

                with col2:
                    if st.button("Remove Anomalies from Dataset"):
                        df_clean = df_result[~df_result['is_anomaly']].drop(columns=['is_anomaly'])
                        st.session_state['data'] = df_clean
                        st.success(f"Removed {results['anomaly_count']} anomalies!")
                        st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def render_text_analysis(df, ml):
    """Text analysis section."""
    st.subheader("Text Analysis")
    st.markdown("Analyze text columns to extract insights and create features.")

    # Find text columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not text_cols:
        st.warning("No text columns found in the dataset.")
        return

    text_col = st.selectbox("Select Text Column", options=text_cols)

    # Preview
    st.write("**Sample Text:**")
    st.write(df[text_col].dropna().head(3).tolist())

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Analyze Text", type="primary"):
            with st.spinner("Analyzing text..."):
                try:
                    results = ml.analyze_text(df, text_col)

                    # Basic Stats
                    st.subheader("Text Statistics")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Documents", results['total_documents'])
                    col2.metric("Avg Word Count", results['avg_word_count'])
                    col3.metric("Avg Char Count", results['avg_char_count'])

                    # Word count stats
                    wc = results['word_count_stats']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Min Words", wc['min'])
                    col2.metric("Max Words", wc['max'])
                    col3.metric("Median Words", wc['median'])

                    # Sentiment Distribution
                    st.subheader("Sentiment Distribution")
                    sentiment = results['sentiment_distribution']
                    fig_sent = px.pie(
                        values=list(sentiment.values()),
                        names=list(sentiment.keys()),
                        title='Sentiment Distribution (Simple Analysis)',
                        color_discrete_sequence=['green', 'gray', 'red']
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)

                    # Top Words
                    st.subheader("Most Common Words")
                    top_words = results['top_words']
                    words_df = pd.DataFrame([
                        {'Word': k, 'Count': v}
                        for k, v in list(top_words.items())[:20]
                    ])

                    fig_words = px.bar(
                        words_df,
                        x='Count', y='Word',
                        orientation='h',
                        title='Top 20 Most Common Words'
                    )
                    fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_words, use_container_width=True)

                    st.session_state['text_analysis'] = results

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        if st.button("Create Text Features"):
            with st.spinner("Creating text features..."):
                try:
                    df_new = ml.create_text_features(df, text_col)

                    new_cols = [c for c in df_new.columns if c not in df.columns]
                    st.success(f"Created {len(new_cols)} text features!")

                    st.write("**New Features:**")
                    for col in new_cols:
                        st.write(f"- {col}")

                    st.write("**Preview:**")
                    st.dataframe(df_new[new_cols].head(10), use_container_width=True)

                    if st.button("Apply Text Features"):
                        st.session_state['data'] = df_new
                        st.success("Text features added!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Tips
    with st.expander("Text Analysis Tips"):
        st.markdown("""
        **What These Features Mean:**
        - **word_count**: Number of words in the text
        - **char_count**: Number of characters
        - **avg_word_length**: Average length of words
        - **uppercase_ratio**: Proportion of uppercase letters
        - **digit_ratio**: Proportion of digits

        **Sentiment Analysis:**
        - Uses a simple keyword-based approach
        - For production use, consider using specialized NLP libraries

        **Use Cases:**
        - Customer reviews analysis
        - Social media sentiment
        - Document classification features
        """)
