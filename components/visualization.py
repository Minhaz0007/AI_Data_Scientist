"""
Enhanced Visualization Generator Component
Includes intelligent auto-visualization, chart recommendations, and one-click dashboard population.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze_data_for_visualization(df):
    """Analyze data and suggest the best visualizations."""
    suggestions = []

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()

    # 1. Distribution analysis for all numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5
        skewness = df[col].skew()
        if abs(skewness) > 1:
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {col} (Skewed)',
                'columns': [col],
                'priority': 'high',
                'reason': f'Highly skewed distribution (skew={skewness:.2f})'
            })
        else:
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'columns': [col],
                'priority': 'medium',
                'reason': 'Normal distribution analysis'
            })

    # 2. Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'columns': numeric_cols,
                'priority': 'high',
                'reason': f'Found {len(high_corr_pairs)} highly correlated column pairs'
            })

            # Add scatter plots for top correlated pairs
            for col1, col2, corr in high_corr_pairs[:3]:
                suggestions.append({
                    'type': 'scatter',
                    'title': f'{col1} vs {col2} (r={corr:.2f})',
                    'columns': [col1, col2],
                    'priority': 'high',
                    'reason': f'High correlation ({corr:.2f})'
                })

    # 3. Categorical analysis
    for cat_col in categorical_cols[:3]:
        n_unique = df[cat_col].nunique()
        if 2 <= n_unique <= 15:
            suggestions.append({
                'type': 'bar',
                'title': f'Distribution of {cat_col}',
                'columns': [cat_col],
                'priority': 'medium',
                'reason': f'{n_unique} categories - good for bar chart'
            })

            # Combination with numeric
            if numeric_cols:
                suggestions.append({
                    'type': 'box',
                    'title': f'{numeric_cols[0]} by {cat_col}',
                    'columns': [numeric_cols[0], cat_col],
                    'priority': 'medium',
                    'reason': 'Compare numeric distribution across categories'
                })

        elif n_unique <= 8:
            suggestions.append({
                'type': 'pie',
                'title': f'Proportion of {cat_col}',
                'columns': [cat_col],
                'priority': 'low',
                'reason': f'{n_unique} categories - suitable for pie chart'
            })

    # 4. Time series analysis
    for dt_col in datetime_cols:
        if numeric_cols:
            suggestions.append({
                'type': 'line',
                'title': f'{numeric_cols[0]} over Time',
                'columns': [dt_col, numeric_cols[0]],
                'priority': 'high',
                'reason': 'Time series visualization'
            })

    # 5. Outlier visualization
    for col in numeric_cols[:3]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > len(df) * 0.05:
            suggestions.append({
                'type': 'box',
                'title': f'Outlier Analysis: {col}',
                'columns': [col],
                'priority': 'high',
                'reason': f'{outliers} potential outliers ({outliers/len(df)*100:.1f}%)'
            })

    return suggestions


def generate_auto_chart(df, suggestion):
    """Generate a chart based on a suggestion."""
    chart_type = suggestion['type']
    columns = suggestion['columns']
    title = suggestion['title']

    try:
        if chart_type == 'histogram':
            fig = px.histogram(df, x=columns[0], title=title, marginal='box')

        elif chart_type == 'scatter':
            fig = px.scatter(df, x=columns[0], y=columns[1], title=title, trendline='ols')

        elif chart_type == 'heatmap':
            corr = df[columns].corr()
            fig = px.imshow(corr, text_auto='.2f', title=title, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

        elif chart_type == 'bar':
            counts = df[columns[0]].value_counts().head(15)
            fig = px.bar(x=counts.index, y=counts.values, title=title, labels={'x': columns[0], 'y': 'Count'})

        elif chart_type == 'box':
            if len(columns) == 1:
                fig = px.box(df, y=columns[0], title=title, points='outliers')
            else:
                fig = px.box(df, y=columns[0], x=columns[1], title=title, points='outliers')

        elif chart_type == 'pie':
            counts = df[columns[0]].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, title=title)

        elif chart_type == 'line':
            df_sorted = df.sort_values(columns[0])
            fig = px.line(df_sorted, x=columns[0], y=columns[1], title=title)

        elif chart_type == 'violin':
            fig = px.violin(df, y=columns[0], title=title, box=True)

        else:
            return None

        return fig

    except Exception as e:
        return None


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

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Auto-Generate",
        "Custom Builder",
        "Quick Charts"
    ])

    with tab1:
        render_auto_generate(df)

    with tab2:
        render_custom_builder(df)

    with tab3:
        render_quick_charts(df)


def render_auto_generate(df):
    """Render the auto-generate visualization tab."""
    st.subheader("Intelligent Auto-Visualization")
    st.markdown("Automatically generate the most relevant visualizations for your data.")

    # Analyze data
    suggestions = analyze_data_for_visualization(df)

    if not suggestions:
        st.info("No visualization suggestions available for this dataset.")
        return

    # Group by priority
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    low_priority = [s for s in suggestions if s['priority'] == 'low']

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("High Priority", len(high_priority))
    col2.metric("Medium Priority", len(medium_priority))
    col3.metric("Low Priority", len(low_priority))

    st.markdown("---")

    # One-click generate all
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Generate Visualizations")
    with col2:
        if st.button("Add All to Dashboard", type="primary"):
            for suggestion in high_priority + medium_priority[:3]:
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    add_to_dashboard(fig, suggestion['title'], suggestion['type'])
            st.success(f"Added {len(high_priority) + min(3, len(medium_priority))} charts to dashboard!")
            st.rerun()

    # High priority visualizations
    if high_priority:
        st.markdown("#### High Priority Visualizations")
        for i, suggestion in enumerate(high_priority):
            with st.expander(f"{suggestion['title']} ({suggestion['type'].title()})", expanded=i < 2):
                st.caption(f"Reason: {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"Add to Dashboard", key=f"auto_high_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])

    # Medium priority
    if medium_priority:
        st.markdown("#### Medium Priority Visualizations")
        for i, suggestion in enumerate(medium_priority[:5]):
            with st.expander(f"{suggestion['title']} ({suggestion['type'].title()})"):
                st.caption(f"Reason: {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"Add to Dashboard", key=f"auto_med_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])

    # Low priority (collapsed)
    if low_priority:
        with st.expander(f"Low Priority ({len(low_priority)} charts)"):
            for i, suggestion in enumerate(low_priority[:3]):
                st.markdown(f"**{suggestion['title']}** - {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button(f"Add to Dashboard", key=f"auto_low_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])


def render_quick_charts(df):
    """Render quick chart generation options."""
    st.subheader("Quick Chart Generator")
    st.markdown("Generate common charts with one click.")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Numeric Analysis")

        if numeric_cols:
            selected_num = st.selectbox("Select Numeric Column", numeric_cols, key="quick_num")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Histogram"):
                    fig = px.histogram(df, x=selected_num, title=f'Distribution of {selected_num}', marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'Distribution of {selected_num}', 'Histogram')

            with col_b:
                if st.button("Box Plot"):
                    fig = px.box(df, y=selected_num, title=f'Box Plot of {selected_num}', points='outliers')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'Box Plot of {selected_num}', 'Box')

            if len(numeric_cols) > 1:
                second_num = st.selectbox("Second Numeric Column", [c for c in numeric_cols if c != selected_num], key="quick_num2")
                if st.button("Scatter Plot"):
                    fig = px.scatter(df, x=selected_num, y=second_num, title=f'{second_num} vs {selected_num}', trendline='ols')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'{second_num} vs {selected_num}', 'Scatter')

    with col2:
        st.markdown("### Categorical Analysis")

        if categorical_cols:
            selected_cat = st.selectbox("Select Categorical Column", categorical_cols, key="quick_cat")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Bar Chart"):
                    counts = df[selected_cat].value_counts().head(15)
                    fig = px.bar(x=counts.index, y=counts.values, title=f'Distribution of {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'Distribution of {selected_cat}', 'Bar')

            with col_b:
                if st.button("Pie Chart"):
                    counts = df[selected_cat].value_counts().head(10)
                    fig = px.pie(values=counts.values, names=counts.index, title=f'Proportion of {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'Proportion of {selected_cat}', 'Pie')

            if numeric_cols:
                if st.button("Category Comparison (Box)"):
                    fig = px.box(df, x=selected_cat, y=numeric_cols[0], title=f'{numeric_cols[0]} by {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_quick_chart'] = (fig, f'{numeric_cols[0]} by {selected_cat}', 'Box')

    # Add last chart to dashboard
    if 'last_quick_chart' in st.session_state:
        st.markdown("---")
        if st.button("Add Last Chart to Dashboard", type="primary"):
            fig, title, chart_type = st.session_state['last_quick_chart']
            add_to_dashboard(fig, title, chart_type)

    # All distributions at once
    st.markdown("---")
    st.markdown("### Bulk Generation")

    if st.button("Generate All Numeric Distributions"):
        cols = st.columns(2)
        for i, col in enumerate(numeric_cols[:6]):
            with cols[i % 2]:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        if st.button("Generate Correlation Heatmap"):
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Matrix', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            st.session_state['last_quick_chart'] = (fig, 'Correlation Matrix', 'Heatmap')


def render_custom_builder(df):
    """Render the custom visualization builder."""
    st.subheader("Custom Visualization Builder")

    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

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
        with st.form("chart_config_form"):
            col1, col2 = st.columns(2)

            # Form Inputs based on chart type
            if chart_type == "Scatter":
                with col1: x_col = st.selectbox("X Axis", numeric_cols)
                with col2: y_col = st.selectbox("Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                color_col = st.selectbox("Color (Optional)", ["None"] + all_cols)
                size_col = st.selectbox("Size (Optional)", ["None"] + numeric_cols)
                trendline = st.checkbox("Add Trendline", value=True)

            elif chart_type == "Line":
                with col1: x_col = st.selectbox("X Axis (Time/Sequence)", all_cols)
                with col2: y_col = st.selectbox("Y Axis", numeric_cols)
                color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            elif chart_type == "Bar":
                with col1: x_col = st.selectbox("X Axis (Categorical)", all_cols)
                with col2: y_col = st.selectbox("Y Axis (Numerical)", numeric_cols)
                color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)
                barmode = st.selectbox("Bar Mode", ["group", "stack", "overlay", "relative"])

            elif chart_type == "Histogram":
                x_col = st.selectbox("Column", numeric_cols)
                bins = st.slider("Number of Bins", 5, 100, 20)
                color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)
                marginal = st.selectbox("Marginal Plot", ["None", "box", "violin", "rug"])

            elif chart_type == "Box":
                with col1: y_col = st.selectbox("Numerical Column", numeric_cols)
                with col2: x_col = st.selectbox("Categorical Column (Optional)", ["None"] + all_cols)
                color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)
                show_points = st.checkbox("Show All Points", value=False)

            elif chart_type == "Heatmap":
                if len(numeric_cols) > 1:
                    cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
                else:
                    st.warning("Need at least 2 numerical columns.")
                    cols = []

            elif chart_type in ["Pie", "Donut"]:
                with col1: names = st.selectbox("Labels (Categorical)", cat_cols if cat_cols else all_cols)
                with col2: values = st.selectbox("Values (Numerical)", numeric_cols)

            elif chart_type == "Sunburst":
                path_cols = st.multiselect("Hierarchy Path (Select in order)", cat_cols if cat_cols else all_cols)
                values = st.selectbox("Values", numeric_cols)

            elif chart_type == "Treemap":
                path_cols = st.multiselect("Hierarchy Path", cat_cols if cat_cols else all_cols)
                values = st.selectbox("Values", numeric_cols)

            elif chart_type == "Funnel":
                with col1: x_col = st.selectbox("Values", numeric_cols)
                with col2: y_col = st.selectbox("Stages", cat_cols if cat_cols else all_cols)

            elif chart_type == "Radar":
                with col1: r_col = st.selectbox("Radius (Numerical)", numeric_cols)
                with col2: theta_col = st.selectbox("Angle (Categorical)", cat_cols if cat_cols else all_cols)
                color_col = st.selectbox("Color (Group)", ["None"] + cat_cols)

            elif chart_type == "Area":
                with col1: x_col = st.selectbox("X Axis", all_cols)
                with col2: y_col = st.selectbox("Y Axis", numeric_cols)
                color_col = st.selectbox("Color (Stack)", ["None"] + cat_cols)

            elif chart_type == "Violin":
                with col1: y_col = st.selectbox("Numerical Data", numeric_cols)
                with col2: x_col = st.selectbox("Category (Optional)", ["None"] + cat_cols)
                color_col = st.selectbox("Color (Optional)", ["None"] + cat_cols)

            submitted = st.form_submit_button("Generate Chart", type="primary")

        # Chart Generation
        if submitted:
            if chart_type == "Scatter":
                color = None if color_col == "None" else color_col
                size = None if size_col == "None" else size_col
                title = f"{y_col} vs {x_col}"
                trend = 'ols' if trendline else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color, size=size, title=title, trendline=trend)

            elif chart_type == "Line":
                color = None if color_col == "None" else color_col
                title = f"{y_col} over {x_col}"
                fig = px.line(df, x=x_col, y=y_col, color=color, title=title)

            elif chart_type == "Bar":
                color = None if color_col == "None" else color_col
                title = f"{y_col} by {x_col}"
                fig = px.bar(df, x=x_col, y=y_col, color=color, barmode=barmode, title=title)

            elif chart_type == "Histogram":
                color = None if color_col == "None" else color_col
                marg = None if marginal == "None" else marginal
                title = f"Distribution of {x_col}"
                fig = px.histogram(df, x=x_col, nbins=bins, color=color, title=title, marginal=marg)

            elif chart_type == "Box":
                x = None if x_col == "None" else x_col
                color = None if color_col == "None" else color_col
                points = "all" if show_points else "outliers"
                title = f"Box Plot of {y_col}"
                fig = px.box(df, y=y_col, x=x, color=color, title=title, points=points)

            elif chart_type == "Heatmap":
                if len(cols) > 1:
                    corr = df[cols].corr()
                    title = "Correlation Matrix"
                    fig = px.imshow(corr, text_auto='.2f', title=title, color_continuous_scale='RdBu_r')
                else:
                    st.error("Please select at least 2 columns.")

            elif chart_type in ["Pie", "Donut"]:
                title = f"{values} distribution by {names}"
                if chart_type == "Pie":
                    fig = px.pie(df, names=names, values=values, title=title)
                else:
                    fig = px.pie(df, names=names, values=values, title=title, hole=0.4)

            elif chart_type == "Sunburst":
                if path_cols:
                    title = f"Sunburst of {values}"
                    fig = px.sunburst(df, path=path_cols, values=values, title=title)
                else:
                    st.error("Select hierarchy path.")

            elif chart_type == "Treemap":
                if path_cols:
                    title = f"Treemap of {values}"
                    fig = px.treemap(df, path=path_cols, values=values, title=title)
                else:
                    st.error("Select hierarchy path.")

            elif chart_type == "Funnel":
                title = f"Funnel of {x_col} by {y_col}"
                fig = px.funnel(df, x=x_col, y=y_col, title=title)

            elif chart_type == "Radar":
                color = None if color_col == "None" else color_col
                title = f"Radar Chart: {r_col} by {theta_col}"
                fig = px.line_polar(df, r=r_col, theta=theta_col, color=color, line_close=True, title=title)

            elif chart_type == "Area":
                color = None if color_col == "None" else color_col
                title = f"Area Chart: {y_col} over {x_col}"
                fig = px.area(df, x=x_col, y=y_col, color=color, title=title)

            elif chart_type == "Violin":
                x = None if x_col == "None" else x_col
                color = None if color_col == "None" else color_col
                title = f"Violin Plot of {y_col}"
                fig = px.violin(df, y=y_col, x=x, color=color, box=True, points="all", title=title)

            if fig:
                st.session_state['last_chart'] = {'fig': fig, 'title': title, 'type': chart_type}

        # Restore last chart if available
        if not fig and 'last_chart' in st.session_state and st.session_state['last_chart']:
            last = st.session_state['last_chart']
            if last['type'] == chart_type:
                fig = last['fig']
                title = last['title']

        if fig:
            st.plotly_chart(fig, use_container_width=True)
            if st.button("Add to Dashboard", type="primary"):
                add_to_dashboard(fig, title, chart_type)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
