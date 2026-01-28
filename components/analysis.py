import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processor import perform_clustering

def render():
    st.header("Analysis Engine")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    st.subheader("Clustering (K-Means)")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
    else:
        selected_cols = st.multiselect("Select Features for Clustering", numeric_cols, default=numeric_cols[:2])
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("Run Clustering"):
            if len(selected_cols) < 2:
                st.error("Please select at least 2 columns.")
            else:
                try:
                    with st.spinner("Running clustering..."):
                        result_df = perform_clustering(df, selected_cols, n_clusters)
                        st.success("Clustering complete!")

                        # Visualize
                        st.subheader("Cluster Visualization")
                        if len(selected_cols) >= 2:
                            fig = px.scatter(
                                result_df,
                                x=selected_cols[0],
                                y=selected_cols[1],
                                color='cluster',
                                hover_data=selected_cols
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        st.write("Clustered Data Preview:")
                        st.dataframe(result_df.head())

                        # Option to save
                        if st.checkbox("Append cluster labels to current dataset"):
                            st.session_state['data'] = result_df
                except Exception as e:
                    st.error(f"Error during clustering: {e}")
