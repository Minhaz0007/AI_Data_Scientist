"""
Enhanced Dashboard Component
Features: Auto-generated KPI cards, data overview dashboard, pinned charts, multiple views.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def _render_kpi_card(label, value, delta=None, delta_type="positive"):
    """Render a styled KPI card using HTML."""
    delta_html = ""
    if delta is not None:
        delta_class = f"kpi-delta-{delta_type}"
        arrow = "+" if delta_type == "positive" else ""
        delta_html = f'<div class="kpi-delta {delta_class}">{arrow}{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def _render_auto_dashboard(df):
    """Auto-generate a comprehensive data overview dashboard."""

    # ─── KPI Row ───
    st.markdown("### Key Metrics")
    kpi_cols = st.columns(5)

    with kpi_cols[0]:
        _render_kpi_card("Total Rows", f"{len(df):,}")
    with kpi_cols[1]:
        _render_kpi_card("Total Columns", f"{len(df.columns)}")
    with kpi_cols[2]:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        _render_kpi_card("Missing Data", f"{missing_pct:.1f}%",
                         "Clean" if missing_pct < 5 else "Needs attention",
                         "positive" if missing_pct < 5 else "negative")
    with kpi_cols[3]:
        dup_count = df.duplicated().sum()
        _render_kpi_card("Duplicates", f"{dup_count:,}",
                         "None found" if dup_count == 0 else f"{dup_count/len(df)*100:.1f}% of rows",
                         "positive" if dup_count == 0 else "negative")
    with kpi_cols[4]:
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        _render_kpi_card("Memory", f"{mem_mb:.1f} MB")

    st.markdown("")

    # ─── Column Type Distribution ───
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Column Types")
        type_counts = df.dtypes.astype(str).value_counts()
        type_map = {
            'int64': 'Integer', 'float64': 'Float', 'object': 'Text',
            'bool': 'Boolean', 'datetime64[ns]': 'DateTime', 'category': 'Category',
            'int32': 'Integer', 'float32': 'Float', 'int16': 'Integer', 'int8': 'Integer',
            'float16': 'Float'
        }
        labels = [type_map.get(str(t), str(t)) for t in type_counts.index]
        fig_types = px.pie(
            values=type_counts.values, names=labels,
            color_discrete_sequence=px.colors.sequential.Purp,
            hole=0.4
        )
        fig_types.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0b8'),
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        st.markdown("### Missing Values Overview")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if len(missing) > 0:
            fig_missing = px.bar(
                x=missing.values, y=missing.index,
                orientation='h',
                labels={'x': 'Missing Count', 'y': 'Column'},
                color=missing.values,
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0a0b8'),
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values found! Your data is complete.")

    # ─── Numeric Column Distributions ───
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.markdown("### Numeric Distributions")
        display_cols = numeric_cols[:6]
        chart_cols = st.columns(min(len(display_cols), 3))

        for i, col in enumerate(display_cols):
            with chart_cols[i % 3]:
                fig = px.histogram(
                    df, x=col, nbins=30,
                    color_discrete_sequence=['#6366f1'],
                    opacity=0.8
                )
                fig.update_layout(
                    title=dict(text=col, font=dict(size=13)),
                    margin=dict(t=35, b=20, l=20, r=20),
                    height=220,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0b8', size=10),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True)

    # ─── Correlation Heatmap ───
    if len(numeric_cols) >= 2:
        st.markdown("### Correlation Matrix")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto='.2f', aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0b8')
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ─── Categorical Columns ───
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.markdown("### Top Categories")
        display_cat = cat_cols[:4]
        cat_chart_cols = st.columns(min(len(display_cat), 2))

        for i, col in enumerate(display_cat):
            with cat_chart_cols[i % 2]:
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index.astype(str), y=value_counts.values,
                    labels={'x': col, 'y': 'Count'},
                    color_discrete_sequence=['#8b5cf6']
                )
                fig.update_layout(
                    title=dict(text=f"{col} (Top 10)", font=dict(size=13)),
                    margin=dict(t=35, b=20, l=20, r=20),
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0b8', size=10),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True)


def render():
    """Render the Dashboard page."""

    if st.session_state.get('data') is None:
        st.markdown("""
        <div class="help-tip">
            <strong>💡 No data loaded yet</strong><br>
            Upload a dataset in <strong>Data Ingestion</strong> to automatically generate a dashboard with KPIs, charts, and insights.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Data Ingestion", type="primary"):
            st.session_state['current_page'] = "Data Ingestion"
            st.rerun()
        return

    df = st.session_state['data']

    # Dashboard view selector
    tab1, tab2, tab3 = st.tabs(["Overview Dashboard", "Pinned Charts", "Data Explorer"])

    # ─── Tab 1: Auto-Generated Overview ───
    with tab1:
        _render_auto_dashboard(df)

    # ─── Tab 2: Pinned Charts ───
    with tab2:
        if 'dashboard_charts' not in st.session_state or not st.session_state['dashboard_charts']:
            st.markdown("""
            <div class="help-tip">
                <strong>📌 No pinned charts yet</strong><br>
                Go to the <strong>Visualization</strong> page to create charts and pin them to your dashboard.
                Pinned charts will appear here so you can monitor them all in one place.
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Visualization", type="primary", key="goto_viz"):
                st.session_state['current_page'] = "Visualization"
                st.rerun()
        else:
            # Dashboard Controls
            ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
            with ctrl1:
                st.markdown(f"**{len(st.session_state['dashboard_charts'])} Pinned Charts**")
            with ctrl2:
                layout = st.selectbox("Layout", ["1 Column", "2 Columns", "3 Columns"], index=1, key="pin_layout")
            with ctrl3:
                if st.button("Clear All", key="clear_pinned"):
                    st.session_state['dashboard_charts'] = []
                    st.rerun()

            # Persistence
            with st.expander("💾 Save / Load Dashboard"):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Export Configuration"):
                        import json
                        try:
                            export_data = []
                            for chart in st.session_state['dashboard_charts']:
                                fig_json = chart['figure'].to_json()
                                export_data.append({
                                    'title': chart['title'],
                                    'type': chart['type'],
                                    'fig_json': fig_json
                                })
                            st.download_button(
                                "Download Dashboard JSON",
                                data=json.dumps(export_data),
                                file_name="dashboard_config.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Error exporting: {e}")

                with c2:
                    uploaded_config = st.file_uploader("Load Configuration", type=['json'], key="load_dash")
                    if uploaded_config and st.button("Load Dashboard"):
                        try:
                            import json
                            import plotly.io as pio
                            data = json.load(uploaded_config)
                            st.session_state['dashboard_charts'] = []
                            for item in data:
                                fig = pio.from_json(item['fig_json'])
                                st.session_state['dashboard_charts'].append({
                                    'figure': fig,
                                    'title': item['title'],
                                    'type': item['type']
                                })
                            st.success("Dashboard loaded!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading: {e}")

            cols_count = int(layout.split()[0])
            charts = st.session_state['dashboard_charts']

            for i in range(0, len(charts), cols_count):
                cols = st.columns(cols_count)
                for j in range(cols_count):
                    if i + j < len(charts):
                        chart_data = charts[i + j]
                        with cols[j]:
                            with st.container():
                                st.markdown(f"#### {chart_data.get('title', 'Untitled Chart')}")
                                st.plotly_chart(chart_data['figure'], use_container_width=True)
                                c1, c2 = st.columns([1, 4])
                                with c1:
                                    if st.button("Remove", key=f"del_{i+j}", help="Remove from dashboard"):
                                        st.session_state['dashboard_charts'].pop(i + j)
                                        st.rerun()
                                with c2:
                                    st.caption(f"Type: {chart_data.get('type', 'Unknown')}")

    # ─── Tab 3: Data Explorer ───
    with tab3:
        st.markdown("### Quick Data Explorer")
        st.markdown("""
        <div class="help-tip">
            <strong>🔍 Explore your data interactively</strong><br>
            Select columns and chart types below to quickly visualize any aspect of your dataset.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()

        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            chart_type = st.selectbox("Chart Type", [
                "Histogram", "Scatter Plot", "Box Plot", "Bar Chart", "Line Chart"
            ], key="explorer_chart")
        with ex_col2:
            x_col = st.selectbox("X-Axis", all_cols, key="explorer_x")
        with ex_col3:
            y_options = ["(count)"] + all_cols
            y_col = st.selectbox("Y-Axis", y_options, key="explorer_y")

        color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="explorer_color")
        color_param = color_col if color_col != "None" else None

        if st.button("Generate Chart", type="primary", key="explorer_generate"):
            try:
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, color=color_param,
                                       color_discrete_sequence=px.colors.qualitative.Set2,
                                       opacity=0.85)
                elif chart_type == "Scatter Plot":
                    y = y_col if y_col != "(count)" else (numeric_cols[0] if numeric_cols else all_cols[0])
                    fig = px.scatter(df, x=x_col, y=y, color=color_param,
                                     color_discrete_sequence=px.colors.qualitative.Set2,
                                     opacity=0.7)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=color_param, y=x_col,
                                  color=color_param,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                elif chart_type == "Bar Chart":
                    if y_col == "(count)":
                        vc = df[x_col].value_counts().head(20)
                        fig = px.bar(x=vc.index.astype(str), y=vc.values,
                                     labels={'x': x_col, 'y': 'Count'},
                                     color_discrete_sequence=['#6366f1'])
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, color=color_param,
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                elif chart_type == "Line Chart":
                    y = y_col if y_col != "(count)" else (numeric_cols[0] if numeric_cols else all_cols[0])
                    fig = px.line(df, x=x_col, y=y, color=color_param,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                else:
                    fig = px.histogram(df, x=x_col)

                fig.update_layout(
                    margin=dict(t=30, b=30, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0b8'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True)

                # Pin to dashboard option
                if st.button("📌 Pin to Dashboard", key="pin_explorer_chart"):
                    title = f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col != "(count)" else "")
                    st.session_state['dashboard_charts'].append({
                        'figure': fig,
                        'title': title,
                        'type': chart_type
                    })
                    st.success("Chart pinned to dashboard!")

            except Exception as e:
                st.error(f"Error generating chart: {e}")
