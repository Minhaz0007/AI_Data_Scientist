import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def suggest_charts(df):
    """
    Suggests chart types based on columns.
    Returns a dict of {name: figure}.
    """
    suggestions = {}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title="Correlation Heatmap")
        suggestions['Correlation Heatmap'] = fig

    # Scatter Plots for top correlated pairs? Too expensive to check all.
    # Just generic suggestions.

    # Univariate Distribution
    if numeric_cols:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        suggestions[f"Distribution of {col}"] = fig

    # Bar Chart
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        # Aggregate for bar chart to avoid overcrowding
        if df[cat].nunique() < 20:
            agg_df = df.groupby(cat)[num].mean().reset_index()
            fig = px.bar(agg_df, x=cat, y=num, title=f"Average {num} by {cat}")
            suggestions[f"Bar: {num} by {cat}"] = fig

    return suggestions

def render():
    st.header("Visualization Generator")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # Auto-suggestions
    st.subheader("AI-Suggested Visualizations")
    if st.checkbox("Show Suggestions"):
        suggestions = suggest_charts(df)
        if suggestions:
            for name, fig in suggestions.items():
                st.write(f"**{name}**")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data pattern for suggestions.")

    st.markdown("---")

    # Custom Builder
    st.subheader("Custom Visualization Builder")

    chart_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Heatmap"])

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()

    try:
        if chart_type == "Scatter":
            x_col = st.selectbox("X Axis", numeric_cols)
            y_col = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            color_col = st.selectbox("Color (Optional)", ["None"] + all_cols)

            if st.button("Generate Scatter Plot"):
                color = None if color_col == "None" else color_col
                fig = px.scatter(df, x=x_col, y=y_col, color=color, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line":
            x_col = st.selectbox("X Axis (Time/Sequence)", all_cols)
            y_col = st.selectbox("Y Axis", numeric_cols)

            if st.button("Generate Line Plot"):
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            x_col = st.selectbox("X Axis (Categorical)", all_cols)
            y_col = st.selectbox("Y Axis (Numerical)", numeric_cols)
            agg_func = st.selectbox("Aggregation", ["None", "Mean", "Sum", "Count"])

            if st.button("Generate Bar Chart"):
                if agg_func != "None":
                    if agg_func == "Mean":
                        plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                    elif agg_func == "Sum":
                        plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                    elif agg_func == "Count":
                        plot_df = df.groupby(x_col)[y_col].count().reset_index()
                    fig = px.bar(plot_df, x=x_col, y=y_col, title=f"{agg_func} of {y_col} by {x_col}")
                else:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols)
            bins = st.slider("Number of Bins", 5, 100, 20)

            if st.button("Generate Histogram"):
                fig = px.histogram(df, x=x_col, nbins=bins, title=f"Distribution of {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box":
            y_col = st.selectbox("Numerical Column", numeric_cols)
            x_col = st.selectbox("Categorical Column (Optional)", ["None"] + all_cols)

            if st.button("Generate Box Plot"):
                x = None if x_col == "None" else x_col
                fig = px.box(df, y=y_col, x=x, title=f"Box Plot of {y_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Heatmap":
            if len(numeric_cols) > 1:
                cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
                if st.button("Generate Heatmap"):
                    corr = df[cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns.")

    except Exception as e:
        st.error(f"Error generating chart: {e}")
