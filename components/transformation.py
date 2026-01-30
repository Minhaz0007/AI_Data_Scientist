"""
Enhanced Data Transformation Component
Includes auto-suggestions, smart transformations, and comprehensive data reshaping tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import filter_data, group_and_aggregate, pivot_table, unpivot_table


def analyze_transformation_opportunities(df):
    """Analyze data and suggest useful transformations."""
    suggestions = []

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime').columns.tolist()

    # Suggest log transformation for skewed distributions
    for col in numeric_cols:
        if df[col].min() > 0:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                suggestions.append({
                    'type': 'log_transform',
                    'column': col,
                    'reason': f"Column '{col}' is highly skewed (skewness={skewness:.2f}). Log transform may help normalize.",
                    'priority': 'medium'
                })

    # Suggest scaling for columns with very different ranges
    if len(numeric_cols) > 1:
        ranges = {col: df[col].max() - df[col].min() for col in numeric_cols if df[col].notna().any()}
        if ranges:
            max_range = max(ranges.values())
            min_range = min(ranges.values())
            if max_range > 0 and min_range > 0 and max_range / min_range > 100:
                suggestions.append({
                    'type': 'scaling',
                    'columns': numeric_cols,
                    'reason': "Numeric columns have very different scales. Consider standardization or normalization.",
                    'priority': 'high'
                })

    # Suggest aggregation if there are categorical columns with many rows per category
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.1 and df[col].nunique() > 1:
            suggestions.append({
                'type': 'aggregation',
                'column': col,
                'reason': f"Column '{col}' has {df[col].nunique()} unique values. Consider grouping and aggregating.",
                'priority': 'low'
            })

    # Suggest time-based features if datetime columns exist
    for col in datetime_cols:
        suggestions.append({
            'type': 'datetime_features',
            'column': col,
            'reason': f"Column '{col}' is datetime. Extract year, month, day, weekday features.",
            'priority': 'medium'
        })

    # Suggest binning for continuous variables with wide ranges
    for col in numeric_cols:
        if df[col].nunique() > 50:
            suggestions.append({
                'type': 'binning',
                'column': col,
                'reason': f"Column '{col}' has many unique values ({df[col].nunique()}). Consider binning for analysis.",
                'priority': 'low'
            })

    return suggestions


def apply_quick_transformation(df, transform_type, column, params=None):
    """Apply a quick transformation to the data."""
    df_new = df.copy()

    if transform_type == 'log_transform':
        df_new[f'{column}_log'] = np.log1p(df_new[column])

    elif transform_type == 'standardize':
        mean = df_new[column].mean()
        std = df_new[column].std()
        df_new[f'{column}_standardized'] = (df_new[column] - mean) / std

    elif transform_type == 'normalize':
        min_val = df_new[column].min()
        max_val = df_new[column].max()
        df_new[f'{column}_normalized'] = (df_new[column] - min_val) / (max_val - min_val)

    elif transform_type == 'sqrt_transform':
        df_new[f'{column}_sqrt'] = np.sqrt(df_new[column].clip(lower=0))

    elif transform_type == 'square_transform':
        df_new[f'{column}_squared'] = df_new[column] ** 2

    elif transform_type == 'reciprocal':
        df_new[f'{column}_reciprocal'] = 1 / df_new[column].replace(0, np.nan)

    return df_new


def render():
    st.header("Data Transformation")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Auto-Suggest",
        "Filter",
        "Group & Aggregate",
        "Pivot/Unpivot",
        "Calculated Columns",
        "Merge Datasets"
    ])

    with tab1:
        render_auto_suggest(df)

    with tab2:
        render_filter(df)

    with tab3:
        render_aggregation(df)

    with tab4:
        render_pivot(df)

    with tab5:
        render_calculated_columns(df)

    with tab6:
        render_merge(df)

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Automated Suggestions")
    from utils.data_processor import get_transformation_suggestions
    suggestions = get_transformation_suggestions(df)

    if suggestions:
        for i, sugg in enumerate(suggestions):
            with st.expander(f"Suggestion: {sugg['type']} for {sugg['column']}"):
                st.write(f"**Reason:** {sugg['reason']}")
                if st.button(f"Apply {sugg['type']}", key=f"apply_sugg_{i}"):
                    if sugg['action'] == 'log':
                         import numpy as np
                         # Handle zeros/negative
                         min_val = df[sugg['column']].min()
                         shift = 0
                         if min_val <= 0:
                             shift = abs(min_val) + 1
                         st.session_state['data'][sugg['column']] = np.log(df[sugg['column']] + shift)
                         st.success(f"Applied Log Transform (shift={shift})")
                         st.rerun()
                    elif sugg['action'] == 'to_datetime':
                        st.session_state['data'][sugg['column']] = pd.to_datetime(df[sugg['column']], errors='coerce')
                        st.success("Converted to DateTime")
                        st.rerun()
    else:
        st.info("No transformation suggestions at this time.")

def render_filter(df):
    st.subheader("Filter Data")

    col_to_filter = st.selectbox("Select Column to Filter", df.columns)
    condition = st.selectbox("Condition", [
        'equals', 'not_equals', 'greater_than', 'less_than',
        'contains', 'between', 'is_null', 'is_not_null'
    ])

    if condition == 'between':
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.text_input("Min Value")
        with col2:
            max_val = st.text_input("Max Value")
        value = f"{min_val},{max_val}" if min_val and max_val else ""
    elif condition in ['is_null', 'is_not_null']:
        value = None
    else:
        # Show unique values for easier selection
        if df[col_to_filter].nunique() <= 50:
            unique_vals = df[col_to_filter].dropna().unique().tolist()
            value = st.selectbox("Select Value", unique_vals)
        else:
            value = st.text_input("Value")

    if st.button("Preview Filter"):
        if condition in ['is_null', 'is_not_null']:
            if condition == 'is_null':
                filtered_df = df[df[col_to_filter].isnull()]
            else:
                filtered_df = df[df[col_to_filter].notna()]
        elif not value and condition not in ['is_null', 'is_not_null']:
            st.error("Please enter a value.")
            return
        else:
            try:
                filtered_df = filter_data(df, col_to_filter, condition, value)
            except Exception as e:
                st.error(f"Error filtering: {e}")
                return

        st.write(f"**Filtered Data:** {len(filtered_df)} rows (from {len(df)} original)")
        st.dataframe(filtered_df.head(20))

        st.session_state['filtered_preview'] = filtered_df

    if 'filtered_preview' in st.session_state:
        if st.button("Apply Filter to Dataset"):
            st.session_state['data'] = st.session_state['filtered_preview']
            del st.session_state['filtered_preview']
            st.success("Dataset updated with filtered data.")
            st.rerun()


def render_aggregation(df):
    st.subheader("Group by and Aggregate")

    group_cols = st.multiselect("Group By Column(s)", df.columns.tolist())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for aggregation.")
        return

    agg_col = st.selectbox("Aggregate Column", numeric_cols)
    method = st.selectbox("Aggregation Method", ['mean', 'sum', 'count', 'min', 'max', 'std', 'median', 'first', 'last'])

    # Multi-aggregation option
    multi_agg = st.checkbox("Multiple aggregations")
    if multi_agg:
        methods = st.multiselect("Select Methods", ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'], default=['mean', 'sum'])

    if st.button("Aggregate"):
        if not group_cols:
            st.error("Please select at least one column to group by.")
        else:
            try:
                if multi_agg and methods:
                    result = df.groupby(group_cols)[agg_col].agg(methods).reset_index()
                elif len(group_cols) == 1:
                    result = group_and_aggregate(df, group_cols[0], agg_col, method)
                else:
                    result = df.groupby(group_cols)[agg_col].agg(method).reset_index()

                st.write("**Aggregated Result:**")
                st.dataframe(result)

                st.session_state['agg_result'] = result

            except Exception as e:
                st.error(f"Error aggregating: {e}")

    if 'agg_result' in st.session_state:
        if st.button("Save Aggregation as New Dataset"):
            st.session_state['data'] = st.session_state['agg_result']
            del st.session_state['agg_result']
            st.success("Dataset replaced with aggregated data.")
            st.rerun()


def render_pivot(df):
    st.subheader("Pivot / Unpivot Operations")

    operation = st.radio("Operation", ["Pivot Table", "Unpivot (Melt)"])

    if operation == "Pivot Table":
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            index_col = st.selectbox("Row Index", all_cols, key="pivot_index")
            values_col = st.selectbox("Values", numeric_cols, key="pivot_values") if numeric_cols else None
        with col2:
            columns_col = st.selectbox("Column Headers", [c for c in all_cols if c != index_col], key="pivot_cols")
            aggfunc = st.selectbox("Aggregation", ['mean', 'sum', 'count', 'min', 'max'])

        if st.button("Create Pivot Table"):
            if not numeric_cols:
                st.error("Need numeric columns for pivot values.")
            else:
                try:
                    pivot_df = pivot_table(df, index_col, columns_col, values_col, aggfunc)
                    st.write("**Pivot Table Result:**")
                    st.dataframe(pivot_df)

                    st.session_state['pivot_result'] = pivot_df.reset_index()

                except Exception as e:
                    st.error(f"Error creating pivot table: {e}")

        if 'pivot_result' in st.session_state:
            if st.button("Save Pivot as New Dataset"):
                st.session_state['data'] = st.session_state['pivot_result']
                del st.session_state['pivot_result']
                st.success("Dataset replaced with pivot table.")
                st.rerun()

    else:  # Unpivot
        all_cols = df.columns.tolist()

        id_vars = st.multiselect("ID Columns (keep fixed)", all_cols, key="melt_id")
        value_vars = st.multiselect("Value Columns (to unpivot)",
                                     [c for c in all_cols if c not in id_vars],
                                     key="melt_values")

        var_name = st.text_input("Variable Column Name", "variable")
        value_name = st.text_input("Value Column Name", "value")

        if st.button("Unpivot Data"):
            if not id_vars:
                st.error("Please select at least one ID column.")
            else:
                try:
                    melted_df = unpivot_table(df, id_vars, value_vars if value_vars else None, var_name, value_name)
                    st.write("**Unpivoted Data:**")
                    st.dataframe(melted_df.head(20))
                    st.write(f"Shape: {melted_df.shape}")

                    st.session_state['melt_result'] = melted_df

                except Exception as e:
                    st.error(f"Error unpivoting: {e}")

        if 'melt_result' in st.session_state:
            if st.button("Save Unpivoted as New Dataset"):
                st.session_state['data'] = st.session_state['melt_result']
                del st.session_state['melt_result']
                st.success("Dataset replaced with unpivoted data.")
                st.rerun()


def render_calculated_columns(df):
    st.subheader("Create Calculated Columns")

    st.info("Create new columns using Python expressions. Use column names directly (e.g., `price * quantity`).")

    # Quick calculations
    st.markdown("### Quick Calculations")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)

        with col1:
            col_a = st.selectbox("Column A", numeric_cols, key="calc_col_a")
        with col2:
            operation = st.selectbox("Operation", ['+', '-', '*', '/', '%', '**'])
        with col3:
            col_b = st.selectbox("Column B", [c for c in numeric_cols if c != col_a], key="calc_col_b")

        new_col_name_quick = st.text_input("New Column Name", f"{col_a}_{operation}_{col_b}")

        if st.button("Create Quick Calculation"):
            df_new = df.copy()
            if operation == '+':
                df_new[new_col_name_quick] = df_new[col_a] + df_new[col_b]
            elif operation == '-':
                df_new[new_col_name_quick] = df_new[col_a] - df_new[col_b]
            elif operation == '*':
                df_new[new_col_name_quick] = df_new[col_a] * df_new[col_b]
            elif operation == '/':
                df_new[new_col_name_quick] = df_new[col_a] / df_new[col_b].replace(0, np.nan)
            elif operation == '%':
                df_new[new_col_name_quick] = df_new[col_a] % df_new[col_b]
            elif operation == '**':
                df_new[new_col_name_quick] = df_new[col_a] ** df_new[col_b]

            st.session_state['data'] = df_new
            st.success(f"Created column '{new_col_name_quick}'!")
            st.rerun()

    st.markdown("---")

    # Custom expression
    st.markdown("### Custom Expression")

    new_col_name = st.text_input("New Column Name", key="custom_col_name")

    with st.expander("Available Columns"):
        st.write(df.columns.tolist())

    expression = st.text_area(
        "Expression",
        placeholder="Examples:\n- price * quantity\n- (column_a + column_b) / 2\n- column_a.str.upper()\n- pd.to_datetime(date_col)",
        height=100
    )

    st.warning("Note: Use `df['column_name']` syntax for columns with spaces or special characters.")

    if st.button("Preview Calculation"):
        if not new_col_name or not expression:
            st.error("Please provide both column name and expression.")
        else:
            try:
                df_copy = df.copy()
                local_vars = {col: df_copy[col] for col in df_copy.columns}
                local_vars['df'] = df_copy
                local_vars['pd'] = pd
                local_vars['np'] = np

                result = eval(expression, {"__builtins__": {}}, local_vars)
                df_copy[new_col_name] = result

                st.write("**Preview (first 10 rows):**")
                st.dataframe(df_copy[[new_col_name] + df.columns.tolist()[:5]].head(10))

                st.session_state['calc_preview'] = df_copy
            except Exception as e:
                st.error(f"Error in expression: {e}")

    if 'calc_preview' in st.session_state:
        if st.button("Apply Calculated Column"):
            st.session_state['data'] = st.session_state['calc_preview']
            del st.session_state['calc_preview']
            st.success(f"Column '{new_col_name}' added to dataset.")
            st.rerun()


def render_merge(df):
    st.subheader("Merge with Another Dataset")

    st.info("Upload a second dataset to merge with the current one.")

    uploaded_file = st.file_uploader("Upload second dataset", type=['csv', 'xlsx', 'json'], key="merge_upload")

    if uploaded_file:
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'csv':
                df2 = pd.read_csv(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                df2 = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                df2 = pd.read_json(uploaded_file)

            st.write(f"**Second Dataset:** {df2.shape[0]} rows, {df2.shape[1]} columns")
            st.dataframe(df2.head())

            # Find common columns
            common_cols = list(set(df.columns) & set(df2.columns))

            if not common_cols:
                st.warning("No common columns found. Please select columns to merge on.")
                col1, col2 = st.columns(2)
                with col1:
                    left_on = st.selectbox("Left Dataset Column", df.columns.tolist())
                with col2:
                    right_on = st.selectbox("Right Dataset Column", df2.columns.tolist())
                on = None
            else:
                on = st.multiselect("Merge On (common columns)", common_cols, default=common_cols[:1])
                left_on = None
                right_on = None

            how = st.selectbox("Merge Type", ['inner', 'left', 'right', 'outer'])

            # Show merge info
            st.markdown("**Merge Types:**")
            st.caption("- **inner**: Only matching rows from both tables")
            st.caption("- **left**: All rows from left table + matching from right")
            st.caption("- **right**: All rows from right table + matching from left")
            st.caption("- **outer**: All rows from both tables")

            if st.button("Preview Merge"):
                try:
                    if on:
                        merged = pd.merge(df, df2, on=on, how=how)
                    else:
                        merged = pd.merge(df, df2, left_on=left_on, right_on=right_on, how=how)

                    st.write(f"**Merged Result:** {merged.shape[0]} rows, {merged.shape[1]} columns")
                    st.dataframe(merged.head(20))

                    st.session_state['merged_preview'] = merged
                except Exception as e:
                    st.error(f"Error merging: {e}")

            if 'merged_preview' in st.session_state:
                if st.button("Apply Merge"):
                    st.session_state['data'] = st.session_state['merged_preview']
                    del st.session_state['merged_preview']
                    st.success("Dataset replaced with merged data.")
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading second dataset: {e}")
