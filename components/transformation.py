import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import filter_data, group_and_aggregate, pivot_table, unpivot_table

def render():
    st.header("Data Transformation")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Filter",
        "Group & Aggregate",
        "Pivot/Unpivot",
        "Calculated Columns",
        "Merge Datasets"
    ])

    with tab1:
        render_filter(df)

    with tab2:
        render_aggregation(df)

    with tab3:
        render_pivot(df)

    with tab4:
        render_calculated_columns(df)

    with tab5:
        render_merge(df)

def render_filter(df):
    st.subheader("Filter Data")

    col_to_filter = st.selectbox("Select Column to Filter", df.columns)
    condition = st.selectbox("Condition", [
        'equals', 'not_equals', 'greater_than', 'less_than',
        'contains', 'between'
    ])

    if condition == 'between':
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.text_input("Min Value")
        with col2:
            max_val = st.text_input("Max Value")
        value = f"{min_val},{max_val}" if min_val and max_val else ""
    else:
        value = st.text_input("Value")

    if st.button("Preview Filter"):
        if not value:
            st.error("Please enter a value.")
        else:
            try:
                filtered_df = filter_data(df, col_to_filter, condition, value)
                st.write(f"**Filtered Data:** {len(filtered_df)} rows (from {len(df)} original)")
                st.dataframe(filtered_df.head(20))

                st.session_state['filtered_preview'] = filtered_df
            except Exception as e:
                st.error(f"Error filtering: {e}")

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
    method = st.selectbox("Aggregation Method", ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'])

    if st.button("Aggregate"):
        if not group_cols:
            st.error("Please select at least one column to group by.")
        else:
            try:
                if len(group_cols) == 1:
                    result = group_and_aggregate(df, group_cols[0], agg_col, method)
                else:
                    # Multi-column groupby
                    result = df.groupby(group_cols)[agg_col].agg(method).reset_index()

                st.write("**Aggregated Result:**")
                st.dataframe(result)

                # Option to save
                if st.button("Save Aggregation as New Dataset"):
                    st.session_state['data'] = result
                    st.success("Dataset replaced with aggregated data.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error aggregating: {e}")

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

                    if st.button("Save Pivot as New Dataset"):
                        st.session_state['data'] = pivot_df.reset_index()
                        st.success("Dataset replaced with pivot table.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error creating pivot table: {e}")

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

                    if st.button("Save Unpivoted as New Dataset"):
                        st.session_state['data'] = melted_df
                        st.success("Dataset replaced with unpivoted data.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error unpivoting: {e}")

def render_calculated_columns(df):
    st.subheader("Create Calculated Columns")

    st.info("Create new columns using Python expressions. Use column names directly (e.g., `price * quantity`).")

    new_col_name = st.text_input("New Column Name")

    # Show available columns
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
                # Create a copy and evaluate
                df_copy = df.copy()
                # Make columns accessible as variables
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
