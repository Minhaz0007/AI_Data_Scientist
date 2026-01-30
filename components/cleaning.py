import streamlit as st
import pandas as pd
from utils.data_processor import (
    remove_duplicates, impute_missing, normalize_column_names, get_missing_summary,
    convert_column_type, get_column_types, clean_string_column, map_values,
    replace_values, drop_columns, rename_columns, reorder_columns, split_column,
    remove_duplicates_subset, get_duplicate_rows, remove_rows_by_condition,
    label_encode, one_hot_encode
)

def render():
    st.header("Data Cleaning")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    st.subheader("Current Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    # Create tabs for different cleaning operations
    tabs = st.tabs([
        "Missing Values",
        "Duplicates",
        "Data Types",
        "String Cleaning",
        "Value Mapping",
        "Column Management",
        "Encoding"
    ])

    with tabs[0]:
        render_missing_values(df)

    with tabs[1]:
        render_duplicates(df)

    with tabs[2]:
        render_data_types(df)

    with tabs[3]:
        render_string_cleaning(df)

    with tabs[4]:
        render_value_mapping(df)

    with tabs[5]:
        render_column_management(df)

    with tabs[6]:
        render_encoding(df)

    st.markdown("---")

    # Show Data Preview
    st.subheader("Cleaned Data Preview")
    st.dataframe(st.session_state['data'].head())


def render_missing_values(df):
    st.subheader("Handle Missing Values")
    missing_summary = get_missing_summary(df)

    if missing_summary:
        st.write("Columns with missing values:")
        st.dataframe(pd.DataFrame(list(missing_summary.items()), columns=['Column', 'Missing Count']))

        col_to_clean = st.selectbox("Select Column", list(missing_summary.keys()), key="missing_col")
        strategy = st.selectbox("Imputation Strategy", ['mean', 'median', 'mode', 'constant', 'drop'], key="missing_strategy")

        fill_value = None
        if strategy == 'constant':
            fill_value = st.text_input("Enter Constant Value", key="missing_constant")

        if st.button("Apply Imputation", key="apply_imputation"):
            try:
                st.session_state['data'] = impute_missing(
                    st.session_state['data'],
                    [col_to_clean],
                    strategy,
                    fill_value
                )
                st.success(f"Imputed column '{col_to_clean}' using {strategy}.")
                st.rerun()
            except Exception as e:
                st.error(f"Error imputing: {e}")
    else:
        st.success("No missing values found.")


def render_duplicates(df):
    st.subheader("Duplicate Removal")

    # Basic duplicate info
    duplicates_count = df.duplicated().sum()
    st.write(f"**Total Duplicate Rows (all columns):** {duplicates_count}")

    # Subset-based deduplication
    st.markdown("#### Remove Duplicates by Columns")
    subset_cols = st.multiselect(
        "Select columns to check for duplicates (leave empty for all columns)",
        df.columns.tolist(),
        key="dup_subset"
    )

    keep_option = st.selectbox(
        "Which duplicate to keep?",
        ['first', 'last', 'none (remove all)'],
        key="dup_keep"
    )
    keep = keep_option if keep_option != 'none (remove all)' else False

    # Show duplicate count for selected subset
    if subset_cols:
        subset_dup_count = df.duplicated(subset=subset_cols).sum()
        st.write(f"Duplicates based on selected columns: {subset_dup_count}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Preview Duplicates", key="preview_dups"):
            subset = subset_cols if subset_cols else None
            dups = get_duplicate_rows(df, subset)
            if len(dups) > 0:
                st.write(f"Found {len(dups)} duplicate rows:")
                st.dataframe(dups.head(50))
            else:
                st.info("No duplicates found.")

    with col2:
        if st.button("Remove Duplicates", key="remove_dups"):
            subset = subset_cols if subset_cols else None
            original_len = len(df)
            st.session_state['data'] = remove_duplicates_subset(df, subset, keep)
            removed = original_len - len(st.session_state['data'])
            st.success(f"Removed {removed} duplicate rows.")
            st.rerun()


def render_data_types(df):
    st.subheader("Data Type Conversion")

    # Show current types
    with st.expander("Current Column Types"):
        type_df = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(df[col].dtype) for col in df.columns],
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Sample Value': [str(df[col].dropna().iloc[0]) if df[col].notna().any() else 'N/A' for col in df.columns]
        })
        st.dataframe(type_df)

    st.markdown("#### Convert Column Type")
    col_to_convert = st.selectbox("Select Column", df.columns.tolist(), key="type_col")
    target_type = st.selectbox(
        "Target Type",
        ['int', 'float', 'string', 'datetime', 'boolean', 'category'],
        key="target_type"
    )

    datetime_format = None
    if target_type == 'datetime':
        datetime_format = st.text_input(
            "DateTime Format (optional)",
            placeholder="%Y-%m-%d, %d/%m/%Y, etc.",
            key="dt_format"
        )
        st.caption("Leave empty for automatic detection")

    if st.button("Convert Type", key="convert_type"):
        try:
            st.session_state['data'] = convert_column_type(
                st.session_state['data'],
                col_to_convert,
                target_type,
                datetime_format if datetime_format else None
            )
            st.success(f"Converted '{col_to_convert}' to {target_type}.")
            st.rerun()
        except Exception as e:
            st.error(f"Error converting: {e}")


def render_string_cleaning(df):
    st.subheader("String Cleaning")

    # Filter to string/object columns
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if not string_cols:
        st.info("No string columns found in the dataset.")
        return

    col_to_clean = st.selectbox("Select Column", string_cols, key="str_col")

    # Show sample values
    st.write("**Sample values:**")
    st.write(df[col_to_clean].head(5).tolist())

    operations = st.multiselect(
        "Select Operations",
        [
            'trim',
            'lower',
            'upper',
            'title',
            'remove_special',
            'remove_digits',
            'remove_whitespace',
            'normalize_whitespace'
        ],
        key="str_ops"
    )

    with st.expander("Operation Descriptions"):
        st.markdown("""
        - **trim**: Remove leading/trailing whitespace
        - **lower**: Convert to lowercase
        - **upper**: Convert to uppercase
        - **title**: Convert to Title Case
        - **remove_special**: Remove special characters (keep letters, numbers, spaces)
        - **remove_digits**: Remove all digits
        - **remove_whitespace**: Remove all whitespace
        - **normalize_whitespace**: Replace multiple spaces with single space
        """)

    if operations and st.button("Apply String Cleaning", key="apply_str_clean"):
        try:
            st.session_state['data'] = clean_string_column(
                st.session_state['data'],
                col_to_clean,
                operations
            )
            st.success(f"Applied {', '.join(operations)} to '{col_to_clean}'.")
            st.rerun()
        except Exception as e:
            st.error(f"Error cleaning: {e}")

    st.markdown("---")

    # Column name normalization
    st.markdown("#### Standardize Column Names")
    st.write("Convert all column names to snake_case format.")
    if st.button("Normalize Column Names", key="normalize_cols"):
        st.session_state['data'] = normalize_column_names(df)
        st.success("Column names normalized to snake_case.")
        st.rerun()


def render_value_mapping(df):
    st.subheader("Value Mapping & Replacement")

    tab1, tab2 = st.tabs(["Find & Replace", "Value Mapping"])

    with tab1:
        st.markdown("#### Find and Replace")
        col_to_replace = st.selectbox("Select Column", df.columns.tolist(), key="replace_col")

        # Show unique values
        unique_count = df[col_to_replace].nunique()
        if unique_count <= 50:
            with st.expander(f"Unique Values ({unique_count})"):
                st.write(df[col_to_replace].unique().tolist())

        find_value = st.text_input("Find Value", key="find_val")
        replace_value = st.text_input("Replace With", key="replace_val")
        match_type = st.selectbox(
            "Match Type",
            ['exact', 'contains', 'startswith', 'endswith', 'regex'],
            key="match_type"
        )

        if st.button("Replace Values", key="apply_replace"):
            if not find_value:
                st.error("Please enter a value to find.")
            else:
                try:
                    st.session_state['data'] = replace_values(
                        st.session_state['data'],
                        col_to_replace,
                        find_value,
                        replace_value,
                        match_type
                    )
                    st.success(f"Replaced values in '{col_to_replace}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error replacing: {e}")

    with tab2:
        st.markdown("#### Map Multiple Values")
        map_col = st.selectbox("Select Column", df.columns.tolist(), key="map_col")

        # Show unique values for reference
        unique_vals = df[map_col].dropna().unique()
        if len(unique_vals) <= 20:
            st.write("**Current unique values:**", unique_vals.tolist())

        st.write("Enter mapping as JSON (e.g., `{\"old1\": \"new1\", \"old2\": \"new2\"}`):")
        mapping_str = st.text_area("Mapping", key="mapping_str", height=100)

        if st.button("Apply Mapping", key="apply_mapping"):
            if not mapping_str:
                st.error("Please enter a mapping.")
            else:
                try:
                    import json
                    mapping = json.loads(mapping_str)
                    st.session_state['data'] = map_values(
                        st.session_state['data'],
                        map_col,
                        mapping
                    )
                    st.success(f"Applied mapping to '{map_col}'.")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Use format: {\"old\": \"new\"}")
                except Exception as e:
                    st.error(f"Error mapping: {e}")


def render_column_management(df):
    st.subheader("Column Management")

    tab1, tab2, tab3, tab4 = st.tabs(["Drop Columns", "Rename Columns", "Reorder Columns", "Split Column"])

    with tab1:
        st.markdown("#### Drop Columns")
        cols_to_drop = st.multiselect(
            "Select columns to drop",
            df.columns.tolist(),
            key="drop_cols"
        )

        if cols_to_drop:
            st.warning(f"Will drop: {', '.join(cols_to_drop)}")

            if st.button("Drop Selected Columns", key="apply_drop"):
                st.session_state['data'] = drop_columns(st.session_state['data'], cols_to_drop)
                st.success(f"Dropped {len(cols_to_drop)} column(s).")
                st.rerun()

    with tab2:
        st.markdown("#### Rename Columns")
        col_to_rename = st.selectbox("Select Column to Rename", df.columns.tolist(), key="rename_col")
        new_name = st.text_input("New Name", key="new_col_name")

        if st.button("Rename Column", key="apply_rename"):
            if not new_name:
                st.error("Please enter a new name.")
            elif new_name in df.columns and new_name != col_to_rename:
                st.error(f"Column '{new_name}' already exists.")
            else:
                st.session_state['data'] = rename_columns(
                    st.session_state['data'],
                    {col_to_rename: new_name}
                )
                st.success(f"Renamed '{col_to_rename}' to '{new_name}'.")
                st.rerun()

    with tab3:
        st.markdown("#### Reorder Columns")
        st.write("Drag columns to reorder (first selected = first column):")

        new_order = st.multiselect(
            "Column Order",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key="col_order"
        )

        if len(new_order) == len(df.columns):
            if st.button("Apply New Order", key="apply_reorder"):
                st.session_state['data'] = reorder_columns(st.session_state['data'], new_order)
                st.success("Column order updated.")
                st.rerun()
        else:
            st.info("Select all columns to set the order.")

    with tab4:
        st.markdown("#### Split Column")
        col_to_split = st.selectbox("Select Column to Split", df.columns.tolist(), key="split_col")

        # Show sample
        st.write("**Sample values:**")
        st.write(df[col_to_split].head(3).tolist())

        delimiter = st.text_input("Delimiter", placeholder="e.g., , or - or space", key="split_delim")
        new_names = st.text_input(
            "New Column Names (comma-separated, optional)",
            placeholder="e.g., first_name, last_name",
            key="split_names"
        )

        if st.button("Split Column", key="apply_split"):
            if not delimiter:
                st.error("Please enter a delimiter.")
            else:
                try:
                    names = [n.strip() for n in new_names.split(',')] if new_names else None
                    st.session_state['data'] = split_column(
                        st.session_state['data'],
                        col_to_split,
                        delimiter,
                        names
                    )
                    st.success(f"Split column '{col_to_split}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error splitting: {e}")


def render_encoding(df):
    st.subheader("Categorical Encoding")

    # Filter to categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    if not cat_cols:
        st.info("No categorical columns found.")
        return

    tab1, tab2 = st.tabs(["Label Encoding", "One-Hot Encoding"])

    with tab1:
        st.markdown("#### Label Encoding")
        st.write("Convert categorical values to numeric codes (0, 1, 2, ...)")

        label_col = st.selectbox("Select Column", cat_cols, key="label_col")

        # Show unique values and proposed mapping
        unique_vals = sorted(df[label_col].dropna().unique(), key=str)
        if len(unique_vals) <= 20:
            proposed_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            st.write("**Proposed Mapping:**")
            st.json(proposed_mapping)

        if st.button("Apply Label Encoding", key="apply_label"):
            try:
                st.session_state['data'], mapping = label_encode(
                    st.session_state['data'],
                    label_col
                )
                st.success(f"Created '{label_col}_encoded' column.")
                st.write("**Mapping used:**")
                st.json(mapping)
                st.rerun()
            except Exception as e:
                st.error(f"Error encoding: {e}")

    with tab2:
        st.markdown("#### One-Hot Encoding")
        st.write("Create binary columns for each category value.")

        onehot_cols = st.multiselect("Select Column(s)", cat_cols, key="onehot_cols")

        if onehot_cols:
            # Show category counts
            for col in onehot_cols:
                n_unique = df[col].nunique()
                st.write(f"- **{col}**: {n_unique} unique values (will create {n_unique} new columns)")

        drop_first = st.checkbox(
            "Drop first category (avoid multicollinearity)",
            key="drop_first"
        )

        if onehot_cols and st.button("Apply One-Hot Encoding", key="apply_onehot"):
            try:
                original_cols = len(df.columns)
                st.session_state['data'] = one_hot_encode(
                    st.session_state['data'],
                    onehot_cols,
                    drop_first
                )
                new_cols = len(st.session_state['data'].columns)
                st.success(f"One-hot encoded {len(onehot_cols)} column(s). Created {new_cols - original_cols + len(onehot_cols)} new columns.")
                st.rerun()
            except Exception as e:
                st.error(f"Error encoding: {e}")
