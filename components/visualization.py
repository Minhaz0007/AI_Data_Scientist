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

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title="Correlation Heatmap")
        suggestions['Correlation Heatmap'] = fig

    # Univariate Distribution
    if numeric_cols:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        suggestions[f"Distribution of {col}"] = fig

    # Bar Chart
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        if df[cat].nunique() < 20:
            agg_df = df.groupby(cat)[num].mean().reset_index()
            fig = px.bar(agg_df, x=cat, y=num, title=f"Average {num} by {cat}")
            suggestions[f"Bar: {num} by {cat}"] = fig

    return suggestions

def add_to_dashboard(fig, title, chart_type):
    """Add the chart to the dashboard session state."""
    if 'dashboard_charts' not in st.session_state:
        st.session_state['dashboard_charts'] = []

    st.session_state['dashboard_charts'].append({
        'figure': fig,
        'title': title,
        'type': chart_type
    })
    st.success(f"Added '{title}' to Dashboard!")

def render():
    st.header("Visualization Generator")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # Initialize dashboard charts list if needed
    if 'dashboard_charts' not in st.session_state:
        st.session_state['dashboard_charts'] = []

    # Auto-suggestions
    with st.expander("AI-Suggested Visualizations", expanded=False):
        suggestions = suggest_charts(df)
        if suggestions:
            for name, fig in suggestions.items():
                st.write(f"**{name}**")
                st.plotly_chart(fig, use_container_width=True)
                if st.button(f"Add to Dashboard", key=f"sugg_{name}"):
                    add_to_dashboard(fig, name, "AI Suggestion")
        else:
            st.info("Not enough data pattern for suggestions.")

    st.markdown("---")

    # Custom Builder
    st.subheader("Custom Visualization Builder")

    chart_types = [
        "Scatter", "Line", "Bar", "Histogram", "Box", "Heatmap",
        "Pie", "Donut", "Sunburst", "Treemap", "Funnel", "Radar", "Area", "Violin"
    ]

    chart_type = st.selectbox("Select Chart Type", chart_types)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    fig = None
    title = ""

    try:
        col1, col2 = st.columns(2)

        if chart_type == "Scatter":
            with col1: x_col = st.selectbox("X Axis", numeric_cols)
            with col2: y_col = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            color_col = st.selectbox("Color (Optional)", ["None"] + all_cols)
            size_col = st.selectbox("Size (Optional)", ["None"] + numeric_cols)

            if st.button("Generate Scatter Plot"):
                color = None if color_col == "None" else color_col
                size = None if size_col == "None" else size_col
                title = f"{y_col} vs {x_col}"
                fig = px.scatter(df, x=x_col, y=y_col, color=color, size=size, title=title)

        elif chart_type == "Line":
            with col1: x_col = st.selectbox("X Axis (Time/Sequence)", all_cols)
            with col2: y_col = st.selectbox("Y Axis", numeric_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            if st.button("Generate Line Plot"):
                color = None if color_col == "None" else color_col
                title = f"{y_col} over {x_col}"
                fig = px.line(df, x=x_col, y=y_col, color=color, title=title)

        elif chart_type == "Bar":
            with col1: x_col = st.selectbox("X Axis (Categorical)", all_cols)
            with col2: y_col = st.selectbox("Y Axis (Numerical)", numeric_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)
            barmode = st.selectbox("Bar Mode", ["group", "stack", "overlay", "relative"])

            if st.button("Generate Bar Chart"):
                color = None if color_col == "None" else color_col
                title = f"{y_col} by {x_col}"
                fig = px.bar(df, x=x_col, y=y_col, color=color, barmode=barmode, title=title)

        elif chart_type == "Histogram":
            x_col = st.selectbox("Column", numeric_cols)
            bins = st.slider("Number of Bins", 5, 100, 20)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            if st.button("Generate Histogram"):
                color = None if color_col == "None" else color_col
                title = f"Distribution of {x_col}"
                fig = px.histogram(df, x=x_col, nbins=bins, color=color, title=title)

        elif chart_type == "Box":
            with col1: y_col = st.selectbox("Numerical Column", numeric_cols)
            with col2: x_col = st.selectbox("Categorical Column (Optional)", ["None"] + all_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            if st.button("Generate Box Plot"):
                x = None if x_col == "None" else x_col
                color = None if color_col == "None" else color_col
                title = f"Box Plot of {y_col}"
                fig = px.box(df, y=y_col, x=x, color=color, title=title)

        elif chart_type == "Heatmap":
            if len(numeric_cols) > 1:
                cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
                if st.button("Generate Heatmap"):
                    corr = df[cols].corr()
                    title = "Correlation Matrix"
                    fig = px.imshow(corr, text_auto=True, title=title)
            else:
                st.warning("Need at least 2 numerical columns.")

        elif chart_type in ["Pie", "Donut"]:
            with col1: names = st.selectbox("Labels (Categorical)", cat_cols if cat_cols else all_cols)
            with col2: values = st.selectbox("Values (Numerical)", numeric_cols)

            if st.button(f"Generate {chart_type} Chart"):
                title = f"{values} distribution by {names}"
                if chart_type == "Pie":
                    fig = px.pie(df, names=names, values=values, title=title)
                else:
                    fig = px.pie(df, names=names, values=values, title=title, hole=0.4)

        elif chart_type == "Sunburst":
            path_cols = st.multiselect("Hierarchy Path (Select in order)", cat_cols if cat_cols else all_cols)
            values = st.selectbox("Values", numeric_cols)

            if path_cols and st.button("Generate Sunburst"):
                title = f"Sunburst of {values}"
                fig = px.sunburst(df, path=path_cols, values=values, title=title)

        elif chart_type == "Treemap":
            path_cols = st.multiselect("Hierarchy Path", cat_cols if cat_cols else all_cols)
            values = st.selectbox("Values", numeric_cols)

            if path_cols and st.button("Generate Treemap"):
                title = f"Treemap of {values}"
                fig = px.treemap(df, path=path_cols, values=values, title=title)

        elif chart_type == "Funnel":
            with col1: x_col = st.selectbox("Values", numeric_cols)
            with col2: y_col = st.selectbox("Stages", cat_cols if cat_cols else all_cols)

            if st.button("Generate Funnel"):
                title = f"Funnel of {x_col} by {y_col}"
                fig = px.funnel(df, x=x_col, y=y_col, title=title)

        elif chart_type == "Radar":
            with col1: r_col = st.selectbox("Radius (Numerical)", numeric_cols)
            with col2: theta_col = st.selectbox("Angle (Categorical)", cat_cols if cat_cols else all_cols)
            color_col = st.selectbox("Color (Group)", ["None"] + cat_cols)

            if st.button("Generate Radar Chart"):
                color = None if color_col == "None" else color_col
                title = f"Radar Chart: {r_col} by {theta_col}"
                fig = px.line_polar(df, r=r_col, theta=theta_col, color=color, line_close=True, title=title)

        elif chart_type == "Area":
            with col1: x_col = st.selectbox("X Axis", all_cols)
            with col2: y_col = st.selectbox("Y Axis", numeric_cols)
            color_col = st.selectbox("Color (Stack)", ["None"] + cat_cols)

            if st.button("Generate Area Chart"):
                color = None if color_col == "None" else color_col
                title = f"Area Chart: {y_col} over {x_col}"
                fig = px.area(df, x=x_col, y=y_col, color=color, title=title)

        elif chart_type == "Violin":
            with col1: y_col = st.selectbox("Numerical Data", numeric_cols)
            with col2: x_col = st.selectbox("Category (Optional)", ["None"] + cat_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            if st.button("Generate Violin Plot"):
                x = None if x_col == "None" else x_col
                color = None if color_col == "None" else color_col
                title = f"Violin Plot of {y_col}"
                fig = px.violin(df, y=y_col, x=x, color=color, box=True, points="all", title=title)

        # Display and Save
        if fig:
            st.plotly_chart(fig, use_container_width=True)

            if st.button("📌 Add to Dashboard", type="primary"):
                add_to_dashboard(fig, title, chart_type)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
