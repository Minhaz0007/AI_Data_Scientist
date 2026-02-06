"""
Enhanced Visualization Generator Component
Includes intelligent auto-visualization, chart recommendations, one-click dashboard population,
and Apache ECharts integration for highly customizable JavaScript-powered visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False


# ─── ECharts Theme Presets ───────────────────────────────────────────
ECHARTS_THEMES = {
    "Default": None,
    "Vintage": ["#d87c7c", "#919e8b", "#d7ab82", "#6e7074", "#61a0a8", "#efa18d", "#787464", "#cc7e63", "#724e58", "#4b565b"],
    "Macarons": ["#2ec7c9", "#b6a2de", "#5ab1ef", "#ffb980", "#d87a80", "#8d98b3", "#e5cf0d", "#97b552", "#95706d", "#dc69aa"],
    "Infographic": ["#c1232b", "#27727b", "#fcce10", "#e87c25", "#b5c334", "#fe8463", "#9bca63", "#fad860", "#f3a43b", "#60c0dd"],
    "Roma": ["#e01f54", "#001852", "#f5e8c8", "#b8d2c7", "#c6b38e", "#a4d8c2", "#f3d999", "#d3758f", "#dcc392", "#2e4783"],
    "Shine": ["#c12e34", "#e6b600", "#0098d9", "#2b821d", "#005eaa", "#339ca8", "#cda819", "#32a487"],
    "Walden": ["#3fb1e3", "#6be6c1", "#626c91", "#a0a7e6", "#c4ebad", "#96dee8"],
    "Westeros": ["#516b91", "#59c4e6", "#edafda", "#93b7e3", "#a5e7f0", "#cbb0e3"],
    "Chalk": ["#fc97af", "#87f7cf", "#f7f494", "#72ccff", "#f7c5a0", "#d4a4eb", "#d2f5a6", "#76f2f2"],
}

ECHARTS_COLOR_SCHEMES = {
    "Indigo": ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#4f46e5", "#4338ca"],
    "Sunset": ["#f97316", "#fb923c", "#fdba74", "#ef4444", "#f59e0b", "#eab308"],
    "Ocean": ["#0ea5e9", "#38bdf8", "#7dd3fc", "#06b6d4", "#22d3ee", "#67e8f9"],
    "Forest": ["#22c55e", "#4ade80", "#86efac", "#16a34a", "#15803d", "#14532d"],
    "Berry": ["#d946ef", "#e879f9", "#f0abfc", "#a855f7", "#c084fc", "#9333ea"],
    "Slate": ["#64748b", "#94a3b8", "#cbd5e1", "#475569", "#334155", "#1e293b"],
}


def _build_echarts_base(title="", theme_colors=None, animation=True, height=400):
    """Build a base ECharts option dict with common settings."""
    option = {
        "title": {"text": title, "left": "center", "textStyle": {"fontFamily": "Inter, sans-serif", "fontSize": 16, "fontWeight": 600}},
        "tooltip": {"trigger": "axis", "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": "#e5e7eb", "textStyle": {"color": "#1a1a2e"}},
        "animation": animation,
        "animationDuration": 800,
        "animationEasing": "cubicInOut",
    }
    if theme_colors:
        option["color"] = theme_colors
    return option


def _build_echarts_bar(df, x_col, y_col, title="", color_scheme="Indigo", horizontal=False, stacked=False, show_labels=True, gradient=False, rounded=True):
    """Build an ECharts bar chart with extensive customization."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    x_data = df[x_col].astype(str).tolist()
    y_data = df[y_col].tolist()

    series_item = {
        "type": "bar",
        "data": y_data,
        "barWidth": "50%",
        "itemStyle": {"borderRadius": [4, 4, 0, 0] if rounded and not horizontal else ([0, 4, 4, 0] if rounded else [0, 0, 0, 0])},
    }

    if gradient:
        series_item["itemStyle"]["color"] = {
            "type": "linear", "x": 0, "y": 0, "x2": 0 if not horizontal else 1, "y2": 1 if not horizontal else 0,
            "colorStops": [{"offset": 0, "color": colors[0]}, {"offset": 1, "color": colors[1] if len(colors) > 1 else colors[0]}]
        }

    if show_labels:
        series_item["label"] = {"show": True, "position": "top" if not horizontal else "right", "fontSize": 10}

    if horizontal:
        option["xAxis"] = {"type": "value"}
        option["yAxis"] = {"type": "category", "data": x_data, "axisLabel": {"fontSize": 11}}
        option["tooltip"]["trigger"] = "axis"
    else:
        option["xAxis"] = {"type": "category", "data": x_data, "axisLabel": {"rotate": 30 if len(x_data) > 8 else 0, "fontSize": 11}}
        option["yAxis"] = {"type": "value"}

    option["series"] = [series_item]
    option["grid"] = {"containLabel": True, "left": "3%", "right": "4%", "bottom": "10%", "top": "15%"}
    return option


def _build_echarts_line(df, x_col, y_cols, title="", color_scheme="Indigo", smooth=True, area=False, show_points=True, animation_type="expansion"):
    """Build an ECharts line/area chart."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    x_data = df[x_col].astype(str).tolist()

    series = []
    for i, y_col in enumerate(y_cols):
        s = {
            "name": y_col,
            "type": "line",
            "data": df[y_col].tolist(),
            "smooth": smooth,
            "symbol": "circle" if show_points else "none",
            "symbolSize": 6,
            "lineStyle": {"width": 2.5},
        }
        if area:
            color = colors[i % len(colors)]
            s["areaStyle"] = {"opacity": 0.15, "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1, "colorStops": [{"offset": 0, "color": color}, {"offset": 1, "color": "rgba(255,255,255,0)"}]}}
        series.append(s)

    option["xAxis"] = {"type": "category", "data": x_data, "boundaryGap": False, "axisLabel": {"fontSize": 11}}
    option["yAxis"] = {"type": "value", "splitLine": {"lineStyle": {"type": "dashed", "color": "#e5e7eb"}}}
    option["series"] = series
    option["grid"] = {"containLabel": True, "left": "3%", "right": "4%", "bottom": "3%", "top": "15%"}

    if len(y_cols) > 1:
        option["legend"] = {"data": y_cols, "bottom": 0}
        option["grid"]["bottom"] = "12%"

    return option


def _build_echarts_pie(df, name_col, value_col=None, title="", color_scheme="Indigo", donut=False, rose=False, show_labels=True):
    """Build an ECharts pie/donut/rose chart."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    if value_col:
        agg = df.groupby(name_col)[value_col].sum().reset_index()
        data = [{"value": float(row[value_col]), "name": str(row[name_col])} for _, row in agg.iterrows()]
    else:
        counts = df[name_col].value_counts().head(12)
        data = [{"value": int(v), "name": str(k)} for k, v in counts.items()]

    series_item = {
        "type": "pie",
        "radius": ["40%", "70%"] if donut else "65%",
        "center": ["50%", "55%"],
        "data": data,
        "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0, "shadowColor": "rgba(0, 0, 0, 0.2)"}},
        "itemStyle": {"borderRadius": 6, "borderColor": "#fff", "borderWidth": 2},
        "animationType": "scale",
        "animationEasing": "elasticOut",
    }

    if rose:
        series_item["roseType"] = "area"

    if show_labels:
        series_item["label"] = {"show": True, "formatter": "{b}: {d}%"}
    else:
        series_item["label"] = {"show": False}

    option["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    option["legend"] = {"orient": "horizontal", "bottom": 0, "type": "scroll"}
    option["series"] = [series_item]
    return option


def _build_echarts_scatter(df, x_col, y_col, title="", color_scheme="Indigo", size_col=None, bubble=False):
    """Build an ECharts scatter/bubble chart."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    if bubble and size_col and size_col in df.columns:
        size_data = df[size_col].fillna(0)
        max_size = size_data.max() if size_data.max() > 0 else 1
        data = [[float(row[x_col]), float(row[y_col]), float(row[size_col])] for _, row in df.head(500).iterrows() if pd.notna(row[x_col]) and pd.notna(row[y_col])]
        series_item = {
            "type": "scatter",
            "data": data,
            "symbolSize": {"type": "function_placeholder"},
            "itemStyle": {"opacity": 0.7},
        }
    else:
        data = [[float(row[x_col]), float(row[y_col])] for _, row in df.head(1000).iterrows() if pd.notna(row[x_col]) and pd.notna(row[y_col])]
        series_item = {
            "type": "scatter",
            "data": data,
            "symbolSize": 8,
            "itemStyle": {"opacity": 0.7},
        }

    option["xAxis"] = {"type": "value", "name": x_col, "nameLocation": "center", "nameGap": 30, "splitLine": {"lineStyle": {"type": "dashed"}}}
    option["yAxis"] = {"type": "value", "name": y_col, "nameLocation": "center", "nameGap": 40, "splitLine": {"lineStyle": {"type": "dashed"}}}
    option["series"] = [series_item]
    option["grid"] = {"containLabel": True, "left": "3%", "right": "8%", "bottom": "10%", "top": "15%"}

    # Visual map for density
    option["visualMap"] = {"show": False, "dimension": 1, "min": float(df[y_col].min()) if not df[y_col].isna().all() else 0, "max": float(df[y_col].max()) if not df[y_col].isna().all() else 1, "inRange": {"color": colors[:3] if len(colors) >= 3 else colors}}

    return option


def _build_echarts_heatmap(df, columns, title="", color_scheme="Indigo"):
    """Build an ECharts heatmap for correlation matrix."""
    corr = df[columns].corr()
    cols_list = corr.columns.tolist()

    data = []
    for i, row_name in enumerate(cols_list):
        for j, col_name in enumerate(cols_list):
            val = round(float(corr.iloc[i, j]), 2)
            data.append([j, i, val])

    option = _build_echarts_base(title)
    option["tooltip"] = {"position": "top", "formatter": {"type": "function_placeholder"}}
    option["xAxis"] = {"type": "category", "data": cols_list, "splitArea": {"show": True}, "axisLabel": {"rotate": 30, "fontSize": 10}}
    option["yAxis"] = {"type": "category", "data": cols_list, "splitArea": {"show": True}, "axisLabel": {"fontSize": 10}}
    option["visualMap"] = {"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0, "inRange": {"color": ["#3b82f6", "#f8f9fc", "#ef4444"]}}
    option["series"] = [{"type": "heatmap", "data": data, "label": {"show": True, "fontSize": 10}, "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}}}]
    option["grid"] = {"containLabel": True, "left": "3%", "right": "4%", "bottom": "18%", "top": "12%"}
    return option


def _build_echarts_radar(df, numeric_cols, title="", color_scheme="Indigo", group_col=None):
    """Build an ECharts radar chart."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    cols_to_use = numeric_cols[:8]
    indicators = []
    for col in cols_to_use:
        max_val = float(df[col].max()) if not df[col].isna().all() else 1
        indicators.append({"name": col, "max": max_val * 1.1})

    option["radar"] = {"indicator": indicators, "shape": "polygon", "splitNumber": 5, "axisName": {"fontSize": 11}}

    if group_col and group_col in df.columns and df[group_col].nunique() <= 6:
        series_data = []
        for i, group in enumerate(df[group_col].unique()[:6]):
            group_df = df[df[group_col] == group]
            values = [round(float(group_df[col].mean()), 2) for col in cols_to_use]
            series_data.append({"value": values, "name": str(group), "areaStyle": {"opacity": 0.1}})
        option["legend"] = {"data": [str(g) for g in df[group_col].unique()[:6]], "bottom": 0}
    else:
        values = [round(float(df[col].mean()), 2) for col in cols_to_use]
        series_data = [{"value": values, "name": "Average", "areaStyle": {"opacity": 0.15}}]

    option["series"] = [{"type": "radar", "data": series_data, "emphasis": {"lineStyle": {"width": 3}}}]
    return option


def _build_echarts_gauge(value, title="", max_val=100, color_scheme="Indigo"):
    """Build an ECharts gauge chart."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = {
        "title": {"text": title, "left": "center", "textStyle": {"fontFamily": "Inter, sans-serif", "fontSize": 16}},
        "series": [{
            "type": "gauge",
            "startAngle": 200,
            "endAngle": -20,
            "min": 0,
            "max": max_val,
            "progress": {"show": True, "width": 18, "itemStyle": {"color": colors[0]}},
            "pointer": {"show": True, "length": "60%", "width": 6},
            "axisLine": {"lineStyle": {"width": 18, "color": [[1, "#e5e7eb"]]}},
            "axisTick": {"show": False},
            "splitLine": {"distance": -18, "length": 12, "lineStyle": {"width": 2, "color": "#999"}},
            "axisLabel": {"distance": 25, "fontSize": 10},
            "detail": {"valueAnimation": True, "fontSize": 28, "fontWeight": 700, "formatter": "{value}", "offsetCenter": [0, "70%"], "color": colors[0]},
            "data": [{"value": round(value, 1)}],
            "animationDuration": 1500,
        }],
    }
    return option


def _build_echarts_parallel(df, numeric_cols, title="", color_scheme="Indigo"):
    """Build an ECharts parallel coordinates chart for multivariate analysis."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    cols_to_use = numeric_cols[:8]

    dims = []
    for i, col in enumerate(cols_to_use):
        dims.append({"dim": i, "name": col, "min": float(df[col].min()), "max": float(df[col].max())})

    data = df[cols_to_use].head(200).values.tolist()

    option = _build_echarts_base(title, colors)
    option["parallelAxis"] = dims
    option["parallel"] = {"left": "5%", "right": "13%", "bottom": "10%", "top": "15%"}
    option["series"] = [{"type": "parallel", "lineStyle": {"width": 1.5, "opacity": 0.4, "color": colors[0]}, "data": data, "smooth": True}]
    return option


def _build_echarts_boxplot(df, numeric_col, group_col=None, title="", color_scheme="Indigo"):
    """Build an ECharts boxplot."""
    colors = ECHARTS_COLOR_SCHEMES.get(color_scheme, ECHARTS_COLOR_SCHEMES["Indigo"])
    option = _build_echarts_base(title, colors)

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()[:10]
        categories = [str(g) for g in groups]
        box_data = []
        for g in groups:
            vals = df[df[group_col] == g][numeric_col].dropna().values
            if len(vals) > 0:
                q1, median, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3 - q1
                lower = max(vals.min(), q1 - 1.5 * iqr)
                upper = min(vals.max(), q3 + 1.5 * iqr)
                box_data.append([round(float(lower), 2), round(float(q1), 2), round(float(median), 2), round(float(q3), 2), round(float(upper), 2)])
            else:
                box_data.append([0, 0, 0, 0, 0])
    else:
        categories = [numeric_col]
        vals = df[numeric_col].dropna().values
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lower = max(vals.min(), q1 - 1.5 * iqr)
        upper = min(vals.max(), q3 + 1.5 * iqr)
        box_data = [[round(float(lower), 2), round(float(q1), 2), round(float(median), 2), round(float(q3), 2), round(float(upper), 2)]]

    option["xAxis"] = {"type": "category", "data": categories, "axisLabel": {"fontSize": 11}}
    option["yAxis"] = {"type": "value", "splitLine": {"lineStyle": {"type": "dashed"}}}
    option["series"] = [{"type": "boxplot", "data": box_data, "itemStyle": {"color": colors[0], "borderColor": colors[1] if len(colors) > 1 else colors[0]}}]
    option["grid"] = {"containLabel": True, "left": "3%", "right": "4%", "bottom": "3%", "top": "15%"}
    return option


def suggest_charts(df):
    """Analyze data and suggest the best visualizations."""
    suggestions = {}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, title="Correlation Heatmap", text_auto=True, color_continuous_scale='RdBu_r')
        suggestions['Correlation Heatmap'] = fig

    if len(numeric_cols) >= 3:
        fig = px.scatter_matrix(df, dimensions=numeric_cols[:3], title="Scatter Matrix (Top 3 Numeric)")
        suggestions['Scatter Matrix'] = fig

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        val_col = numeric_cols[0]
        df_agg = df.groupby(date_col)[val_col].mean().reset_index()
        fig = px.line(df_agg, x=date_col, y=val_col, title=f"Trend of {val_col} over Time")
        suggestions[f"Trend: {val_col}"] = fig

    if numeric_cols:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
        suggestions[f"Dist: {col}"] = fig

    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        if df[cat].nunique() < 20:
            agg_df = df.groupby(cat)[num].mean().reset_index().sort_values(num, ascending=False)
            fig = px.bar(agg_df, x=cat, y=num, title=f"Avg {num} by {cat}")
            suggestions[f"Bar: {num} by {cat}"] = fig

    if categorical_cols and len(numeric_cols) > 0:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        if df[cat].nunique() < 10:
            fig = px.box(df, x=cat, y=num, title=f"{num} Distribution by {cat}")
            suggestions[f"Box: {num} by {cat}"] = fig

    return suggestions


def analyze_visualization_suggestions(df):
    """Analyze data and return prioritized visualization suggestions."""
    suggestions = []

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)

    for col in numeric_cols[:3]:
        suggestions.append({'type': 'histogram', 'columns': [col], 'title': f'Distribution of {col}', 'reason': 'Understanding data distribution is fundamental', 'priority': 'high'})

    if len(numeric_cols) > 3:
        suggestions.append({'type': 'heatmap', 'columns': numeric_cols, 'title': 'Correlation Matrix', 'reason': 'Identify relationships between numeric variables', 'priority': 'high'})

    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                suggestions.append({'type': 'scatter', 'columns': [col1, col2], 'title': f'{col2} vs {col1}', 'reason': 'Explore potential relationships', 'priority': 'medium'})

    if categorical_cols and numeric_cols:
        for cat in categorical_cols[:2]:
            if df[cat].nunique() <= 10:
                for num in numeric_cols[:2]:
                    suggestions.append({'type': 'box', 'columns': [num, cat], 'title': f'{num} by {cat}', 'reason': 'Compare distributions across categories', 'priority': 'medium'})

    for cat in categorical_cols[:2]:
        if df[cat].nunique() <= 20:
            suggestions.append({'type': 'bar', 'columns': [cat], 'title': f'Distribution of {cat}', 'reason': 'Understand category frequencies', 'priority': 'medium'})

    if datetime_cols and numeric_cols:
        for date_col in datetime_cols[:1]:
            for num_col in numeric_cols[:2]:
                suggestions.append({'type': 'line', 'columns': [date_col, num_col], 'title': f'{num_col} over Time', 'reason': 'Analyze temporal patterns', 'priority': 'low'})

    for col in numeric_cols[:2]:
        suggestions.append({'type': 'violin', 'columns': [col], 'title': f'Violin Plot of {col}', 'reason': 'Detailed distribution visualization', 'priority': 'low'})

    for cat in categorical_cols[:1]:
        if df[cat].nunique() <= 8:
            suggestions.append({'type': 'pie', 'columns': [cat], 'title': f'Proportion of {cat}', 'reason': 'Show category proportions', 'priority': 'low'})

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
    except Exception:
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
    st.header("Visualization Studio")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    if 'dashboard_charts' not in st.session_state:
        st.session_state['dashboard_charts'] = []

    # Main tabs for different visualization modes
    tab_labels = ["AI Suggestions", "Custom Builder (Plotly)", "Quick Charts"]
    if ECHARTS_AVAILABLE:
        tab_labels.insert(2, "ECharts Studio (JS)")

    tabs = st.tabs(tab_labels)

    # ─── Tab 1: AI Suggestions ───────────────────────────────────────
    with tabs[0]:
        _render_ai_suggestions(df)

    # ─── Tab 2: Custom Builder (Plotly) ──────────────────────────────
    with tabs[1]:
        render_custom_builder(df)

    # ─── Tab 3: ECharts Studio (if available) ────────────────────────
    if ECHARTS_AVAILABLE:
        with tabs[2]:
            _render_echarts_studio(df)

    # ─── Tab: Quick Charts ───────────────────────────────────────────
    quick_tab_idx = 3 if ECHARTS_AVAILABLE else 2
    with tabs[quick_tab_idx]:
        render_quick_charts(df)


def _render_ai_suggestions(df):
    """Render AI-suggested visualizations tab."""
    with st.expander("AI-Suggested Visualizations", expanded=True):
        suggestions = suggest_charts(df)
        if suggestions:
            cols = st.columns(2)
            for i, (name, fig) in enumerate(suggestions.items()):
                with cols[i % 2]:
                    st.write(f"**{name}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"ai_sugg_chart_{i}")
                    if st.button(f"Add to Dashboard", key=f"sugg_{name}_{i}"):
                        add_to_dashboard(fig, name, "AI Suggestion")

            if st.button("Auto-Generate Dashboard (Add All)", type="primary"):
                for name, fig in suggestions.items():
                    add_to_dashboard(fig, name, "AI Suggestion")
                st.success(f"Added {len(suggestions)} charts to dashboard!")
        else:
            st.info("Not enough data patterns for suggestions.")

    st.markdown("---")

    viz_suggestions = analyze_visualization_suggestions(df)
    high_priority = [s for s in viz_suggestions if s['priority'] == 'high']
    medium_priority = [s for s in viz_suggestions if s['priority'] == 'medium']
    low_priority = [s for s in viz_suggestions if s['priority'] == 'low']

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Generate Visualizations")
    with col2:
        if st.button("Add All to Dashboard", type="primary", key="btn_add_all_viz"):
            for suggestion in high_priority + medium_priority[:3]:
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    add_to_dashboard(fig, suggestion['title'], suggestion['type'])
            st.success(f"Added {len(high_priority) + min(3, len(medium_priority))} charts to dashboard!")
            st.rerun()

    if high_priority:
        st.markdown("#### High Priority Visualizations")
        for i, suggestion in enumerate(high_priority):
            with st.expander(f"{suggestion['title']} ({suggestion['type'].title()})", expanded=i < 2):
                st.caption(f"Reason: {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_high_{i}")
                    if st.button(f"Add to Dashboard", key=f"auto_high_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])

    if medium_priority:
        st.markdown("#### Medium Priority Visualizations")
        for i, suggestion in enumerate(medium_priority[:5]):
            with st.expander(f"{suggestion['title']} ({suggestion['type'].title()})"):
                st.caption(f"Reason: {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_med_{i}")
                    if st.button(f"Add to Dashboard", key=f"auto_med_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])

    if low_priority:
        with st.expander(f"Low Priority ({len(low_priority)} charts)"):
            for i, suggestion in enumerate(low_priority[:3]):
                st.markdown(f"**{suggestion['title']}** - {suggestion['reason']}")
                fig = generate_auto_chart(df, suggestion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_low_{i}")
                    if st.button(f"Add to Dashboard", key=f"auto_low_{i}"):
                        add_to_dashboard(fig, suggestion['title'], suggestion['type'])


def _render_echarts_studio(df):
    """Render the ECharts-powered visualization studio."""
    st.markdown("### ECharts Visualization Studio")
    st.caption("Powered by Apache ECharts (JavaScript) - highly customizable interactive charts with animations, themes, and advanced chart types.")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for visualization.")
        return

    # Chart type selection
    echarts_types = [
        "Bar Chart", "Line / Area Chart", "Pie / Donut / Rose",
        "Scatter / Bubble", "Heatmap (Correlation)", "Radar Chart",
        "Gauge", "Box Plot", "Parallel Coordinates"
    ]

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_type = st.selectbox("Chart Type", echarts_types, key="ec_chart_type")
    with c2:
        color_scheme = st.selectbox("Color Scheme", list(ECHARTS_COLOR_SCHEMES.keys()), key="ec_color_scheme")

    # Customization sidebar
    with st.expander("Customization Options", expanded=False):
        cust_c1, cust_c2, cust_c3 = st.columns(3)
        with cust_c1:
            animation = st.checkbox("Enable Animation", value=True, key="ec_animation")
            show_labels = st.checkbox("Show Labels", value=True, key="ec_labels")
        with cust_c2:
            chart_height = st.slider("Chart Height (px)", 300, 800, 450, 50, key="ec_height")
            gradient_fill = st.checkbox("Gradient Fill", value=True, key="ec_gradient")
        with cust_c3:
            rounded_corners = st.checkbox("Rounded Corners", value=True, key="ec_rounded")
            smooth_lines = st.checkbox("Smooth Lines", value=True, key="ec_smooth")

    # Chart-specific configuration and rendering
    try:
        if chart_type == "Bar Chart":
            bc1, bc2 = st.columns(2)
            with bc1:
                x_col = st.selectbox("Category (X Axis)", cat_cols + all_cols, key="ec_bar_x")
            with bc2:
                y_col = st.selectbox("Value (Y Axis)", numeric_cols, key="ec_bar_y")
            horizontal = st.checkbox("Horizontal", value=False, key="ec_bar_horiz")

            if st.button("Generate EChart", type="primary", key="ec_bar_gen"):
                agg_df = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(20)
                option = _build_echarts_bar(agg_df, x_col, y_col, title=f"Avg {y_col} by {x_col}", color_scheme=color_scheme, horizontal=horizontal, show_labels=show_labels, gradient=gradient_fill, rounded=rounded_corners)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_bar_chart")

        elif chart_type == "Line / Area Chart":
            x_col = st.selectbox("X Axis (Sequence/Time)", all_cols, key="ec_line_x")
            y_cols = st.multiselect("Y Axis (Numeric)", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1], key="ec_line_y")
            area_fill = st.checkbox("Area Fill", value=False, key="ec_area_fill")
            show_points = st.checkbox("Show Data Points", value=True, key="ec_line_points")

            if y_cols and st.button("Generate EChart", type="primary", key="ec_line_gen"):
                plot_df = df.sort_values(x_col).head(500)
                option = _build_echarts_line(plot_df, x_col, y_cols, title=f"{'Area' if area_fill else 'Line'}: {', '.join(y_cols)}", color_scheme=color_scheme, smooth=smooth_lines, area=area_fill, show_points=show_points)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_line_chart")

        elif chart_type == "Pie / Donut / Rose":
            pc1, pc2 = st.columns(2)
            with pc1:
                name_col = st.selectbox("Category", cat_cols if cat_cols else all_cols, key="ec_pie_name")
            with pc2:
                value_col = st.selectbox("Value (Optional)", ["Auto (Count)"] + numeric_cols, key="ec_pie_val")
            pie_style = st.radio("Style", ["Pie", "Donut", "Rose (Nightingale)"], horizontal=True, key="ec_pie_style")

            if st.button("Generate EChart", type="primary", key="ec_pie_gen"):
                val = None if value_col == "Auto (Count)" else value_col
                option = _build_echarts_pie(df, name_col, val, title=f"{'Donut' if pie_style == 'Donut' else ('Rose' if 'Rose' in pie_style else 'Pie')}: {name_col}", color_scheme=color_scheme, donut=(pie_style == "Donut"), rose=("Rose" in pie_style), show_labels=show_labels)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_pie_chart")

        elif chart_type == "Scatter / Bubble":
            sc1, sc2 = st.columns(2)
            with sc1:
                x_col = st.selectbox("X Axis", numeric_cols, key="ec_scat_x")
            with sc2:
                y_col = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="ec_scat_y")
            bubble = st.checkbox("Bubble Mode (size by column)", value=False, key="ec_bubble")
            size_col = None
            if bubble and len(numeric_cols) > 2:
                size_col = st.selectbox("Size Column", [c for c in numeric_cols if c not in [x_col, y_col]], key="ec_scat_size")

            if st.button("Generate EChart", type="primary", key="ec_scat_gen"):
                option = _build_echarts_scatter(df, x_col, y_col, title=f"{y_col} vs {x_col}", color_scheme=color_scheme, size_col=size_col, bubble=bubble)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_scat_chart")

        elif chart_type == "Heatmap (Correlation)":
            selected_cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:6], key="ec_heat_cols")
            if len(selected_cols) >= 2 and st.button("Generate EChart", type="primary", key="ec_heat_gen"):
                option = _build_echarts_heatmap(df, selected_cols, title="Correlation Heatmap", color_scheme=color_scheme)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_heat_chart")
            elif len(selected_cols) < 2:
                st.info("Select at least 2 numeric columns.")

        elif chart_type == "Radar Chart":
            radar_cols = st.multiselect("Numeric Dimensions", numeric_cols, default=numeric_cols[:5], key="ec_radar_cols")
            group_col = st.selectbox("Group By (Optional)", ["None"] + cat_cols, key="ec_radar_group")

            if len(radar_cols) >= 3 and st.button("Generate EChart", type="primary", key="ec_radar_gen"):
                grp = None if group_col == "None" else group_col
                option = _build_echarts_radar(df, radar_cols, title="Radar Analysis", color_scheme=color_scheme, group_col=grp)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_radar_chart")
            elif len(radar_cols) < 3:
                st.info("Select at least 3 numeric columns.")

        elif chart_type == "Gauge":
            gc1, gc2 = st.columns(2)
            with gc1:
                gauge_col = st.selectbox("Metric Column", numeric_cols, key="ec_gauge_col")
            with gc2:
                gauge_agg = st.selectbox("Aggregation", ["Mean", "Median", "Sum", "Min", "Max"], key="ec_gauge_agg")

            if st.button("Generate EChart", type="primary", key="ec_gauge_gen"):
                agg_map = {"Mean": "mean", "Median": "median", "Sum": "sum", "Min": "min", "Max": "max"}
                val = float(getattr(df[gauge_col], agg_map[gauge_agg])())
                max_val = float(df[gauge_col].max()) * 1.2
                option = _build_echarts_gauge(val, title=f"{gauge_agg} of {gauge_col}", max_val=max_val, color_scheme=color_scheme)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_gauge_chart")

        elif chart_type == "Box Plot":
            bpc1, bpc2 = st.columns(2)
            with bpc1:
                bp_col = st.selectbox("Numeric Column", numeric_cols, key="ec_box_col")
            with bpc2:
                bp_group = st.selectbox("Group By (Optional)", ["None"] + cat_cols, key="ec_box_group")

            if st.button("Generate EChart", type="primary", key="ec_box_gen"):
                grp = None if bp_group == "None" else bp_group
                option = _build_echarts_boxplot(df, bp_col, group_col=grp, title=f"Box Plot: {bp_col}", color_scheme=color_scheme)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_box_chart")

        elif chart_type == "Parallel Coordinates":
            par_cols = st.multiselect("Numeric Dimensions", numeric_cols, default=numeric_cols[:5], key="ec_par_cols")

            if len(par_cols) >= 2 and st.button("Generate EChart", type="primary", key="ec_par_gen"):
                option = _build_echarts_parallel(df, par_cols, title="Parallel Coordinates", color_scheme=color_scheme)
                st_echarts(options=option, height=f"{chart_height}px", key="ec_par_chart")
            elif len(par_cols) < 2:
                st.info("Select at least 2 numeric columns.")

    except Exception as e:
        st.error(f"Error generating EChart: {e}")


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
                if st.button("Histogram", key="btn_histogram"):
                    fig = px.histogram(df, x=selected_num, title=f'Distribution of {selected_num}', marginal='box')
                    st.plotly_chart(fig, use_container_width=True, key="quick_histogram")
                    st.session_state['last_quick_chart'] = (fig, f'Distribution of {selected_num}', 'Histogram')

            with col_b:
                if st.button("Box Plot", key="btn_boxplot"):
                    fig = px.box(df, y=selected_num, title=f'Box Plot of {selected_num}', points='outliers')
                    st.plotly_chart(fig, use_container_width=True, key="quick_boxplot")
                    st.session_state['last_quick_chart'] = (fig, f'Box Plot of {selected_num}', 'Box')

            if len(numeric_cols) > 1:
                second_num = st.selectbox("Second Numeric Column", [c for c in numeric_cols if c != selected_num], key="quick_num2")
                if st.button("Scatter Plot", key="btn_scatter"):
                    fig = px.scatter(df, x=selected_num, y=second_num, title=f'{second_num} vs {selected_num}', trendline='ols')
                    st.plotly_chart(fig, use_container_width=True, key="quick_scatter")
                    st.session_state['last_quick_chart'] = (fig, f'{second_num} vs {selected_num}', 'Scatter')

    with col2:
        st.markdown("### Categorical Analysis")

        if categorical_cols:
            selected_cat = st.selectbox("Select Categorical Column", categorical_cols, key="quick_cat")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Bar Chart", key="btn_bar"):
                    counts = df[selected_cat].value_counts().head(15)
                    fig = px.bar(x=counts.index, y=counts.values, title=f'Distribution of {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True, key="quick_bar")
                    st.session_state['last_quick_chart'] = (fig, f'Distribution of {selected_cat}', 'Bar')

            with col_b:
                if st.button("Pie Chart", key="btn_pie"):
                    counts = df[selected_cat].value_counts().head(10)
                    fig = px.pie(values=counts.values, names=counts.index, title=f'Proportion of {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True, key="quick_pie")
                    st.session_state['last_quick_chart'] = (fig, f'Proportion of {selected_cat}', 'Pie')

            if numeric_cols:
                if st.button("Category Comparison (Box)", key="btn_cat_box"):
                    fig = px.box(df, x=selected_cat, y=numeric_cols[0], title=f'{numeric_cols[0]} by {selected_cat}')
                    st.plotly_chart(fig, use_container_width=True, key="quick_cat_box")
                    st.session_state['last_quick_chart'] = (fig, f'{numeric_cols[0]} by {selected_cat}', 'Box')

    if 'last_quick_chart' in st.session_state:
        st.markdown("---")
        if st.button("Add Last Chart to Dashboard", type="primary", key="btn_add_last"):
            fig, title, chart_type = st.session_state['last_quick_chart']
            add_to_dashboard(fig, title, chart_type)

    st.markdown("---")
    st.markdown("### Bulk Generation")

    if st.button("Generate All Numeric Distributions", key="btn_all_dist"):
        cols = st.columns(2)
        for i, col in enumerate(numeric_cols[:6]):
            with cols[i % 2]:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True, key=f"bulk_dist_{i}")

    if len(numeric_cols) > 1:
        if st.button("Generate Correlation Heatmap", key="btn_corr_heatmap"):
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Matrix', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True, key="quick_corr_heatmap")
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

    chart_type = st.selectbox("Select Chart Type", chart_types, key="custom_chart_type")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    fig = None
    title = ""

    try:
        with st.form("chart_config_form"):
            col1, col2 = st.columns(2)

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

        if not fig and 'last_chart' in st.session_state and st.session_state['last_chart']:
            last = st.session_state['last_chart']
            if last['type'] == chart_type:
                fig = last['fig']
                title = last['title']

        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_builder_chart")
            if st.button("Add to Dashboard", type="primary", key="btn_add_custom_to_dashboard"):
                add_to_dashboard(fig, title, chart_type)

    except Exception as e:
        st.error(f"Error generating chart: {e}")
