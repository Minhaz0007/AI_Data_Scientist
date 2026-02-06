"""
Enhanced Data Cleaning Component
Includes auto-clean suggestions, one-click cleaning, and comprehensive data cleaning tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pyarrow as pa
from utils.data_processor import (
    remove_duplicates, impute_missing, normalize_column_names, get_missing_summary,
    convert_column_type, get_column_types, clean_string_column, map_values,
    replace_values, drop_columns, rename_columns, reorder_columns, split_column,
    remove_duplicates_subset, get_duplicate_rows, remove_rows_by_condition,
    label_encode, one_hot_encode,
    get_data_health_report, apply_selected_recommendations
)


def render_data_insights(df):
    """
    Render a comprehensive data health insights dashboard.
    Shown immediately when the user navigates to the Data Cleaning page.
    Analyzes the entire file and suggests what cleaning actions to take.
    """
    report = get_data_health_report(df)

    # --- Quality Score Header ---
    score = report['quality_score']
    if score >= 90:
        score_color = "green"
        score_label = "Excellent"
    elif score >= 70:
        score_color = "orange"
        score_label = "Good"
    elif score >= 50:
        score_color = "orange"
        score_label = "Fair"
    else:
        score_color = "red"
        score_label = "Needs Attention"

    st.subheader("Data Health Report")
    st.markdown(f"**Overall Quality Score: {score}/100** ({score_label})")
    st.progress(min(score / 100, 1.0))

    # Quality breakdown
    breakdown = report.get('quality_breakdown', {})
    qcol1, qcol2, qcol3 = st.columns(3)
    qcol1.metric("Completeness", f"{breakdown.get('missing_score', 0)}/100",
                  help="Based on missing value ratio")
    qcol2.metric("Uniqueness", f"{breakdown.get('duplicate_score', 0)}/100",
                  help="Based on duplicate row ratio")
    qcol3.metric("Consistency", f"{breakdown.get('outlier_score', 0)}/100",
                  help="Based on outlier ratio in numeric columns")

    # --- Dataset Overview ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{report['total_rows']:,}")
    col2.metric("Columns", report['total_columns'])
    col3.metric("Missing Values", f"{report['total_missing']:,}")
    col4.metric("Duplicate Rows", f"{report['total_duplicates']:,}")

    # --- Issues Found ---
    if report['issues']:
        st.markdown("---")
        st.markdown("### Issues Found")

        high_issues = [i for i in report['issues'] if i['severity'] == 'high']
        med_issues = [i for i in report['issues'] if i['severity'] == 'medium']
        low_issues = [i for i in report['issues'] if i['severity'] == 'low']

        for issue in high_issues:
            st.error(f"**HIGH:** {issue['message']}")
        for issue in med_issues:
            st.warning(f"**MEDIUM:** {issue['message']}")
        if low_issues:
            with st.expander(f"Low severity issues ({len(low_issues)})", expanded=False):
                for issue in low_issues:
                    st.info(issue['message'])
    else:
        st.success("No data quality issues detected. Your data looks clean!")
        return

    # --- Column Health Summary ---
    st.markdown("---")
    with st.expander("Column-by-Column Health", expanded=False):
        col_rows = []
        for col, info in report['column_health'].items():
            status_map = {
                'good': 'Good',
                'fair': 'Fair',
                'poor': 'Poor',
                'critical': 'Critical',
                'constant': 'Constant',
            }
            status = status_map.get(info['health'], info['health'])
            flags = []
            if info['missing'] > 0:
                flags.append(f"{info['missing_pct']}% missing")
            if info.get('type_mismatch'):
                flags.append("type mismatch")
            if info.get('has_whitespace'):
                flags.append("whitespace")
            if info.get('outliers', 0) > 0:
                flags.append(f"{info['outlier_pct']}% outliers")
            if info['health'] == 'constant':
                flags.append("constant value")

            col_rows.append({
                'Column': col,
                'Type': info['dtype'],
                'Health': status,
                'Missing': f"{info['missing']:,} ({info['missing_pct']}%)" if info['missing'] > 0 else "0",
                'Issues': ', '.join(flags) if flags else 'None',
            })

        st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    # --- Recommended Actions ---
    if report['recommendations']:
        st.markdown("---")
        st.markdown("### Recommended Cleaning Actions")
        st.markdown("Select the actions you want to apply. Each shows the estimated impact on your data.")

        # Store report in session state for use by the apply button
        st.session_state['_health_report'] = report

        selected = []
        for rec in report['recommendations']:
            # Build impact label
            if rec['impact_type'] == 'rows_removed':
                impact = f"**Impact:** {rec['impact_rows']:,} rows will be removed"
                pct = round(rec['impact_rows'] / report['total_rows'] * 100, 1) if report['total_rows'] > 0 else 0
                impact += f" ({pct}% of data)"
            elif rec['impact_type'] == 'cols_removed':
                impact = f"**Impact:** {rec['impact_cols']} column(s) will be removed"
            elif rec['impact_type'] == 'values_filled':
                impact = f"**Impact:** Missing values filled (no rows removed)"
            elif rec['impact_type'] == 'types_fixed':
                impact = f"**Impact:** Column types corrected (no rows removed)"
            elif rec['impact_type'] == 'values_cleaned':
                impact = f"**Impact:** Values cleaned in place (no rows removed)"
            else:
                impact = ""

            checked = st.checkbox(
                rec['label'],
                value=rec.get('safe', False) and rec['impact_type'] != 'rows_removed',
                key=f"rec_{rec['id']}",
                help=rec.get('description', '')
            )
            st.caption(f"  {impact}")

            if checked:
                selected.append(rec['id'])

        st.markdown("---")

        # Estimate total impact
        total_rows_removed = sum(
            r['impact_rows'] for r in report['recommendations']
            if r['id'] in selected and r.get('impact_rows', 0) > 0
        )
        total_cols_removed = sum(
            r['impact_cols'] for r in report['recommendations']
            if r['id'] in selected and r.get('impact_cols', 0) > 0
        )

        est_col1, est_col2, est_col3 = st.columns(3)
        est_col1.metric("Actions Selected", len(selected))
        est_col2.metric("Est. Rows Removed", f"{total_rows_removed:,}")
        est_col3.metric("Est. Columns Removed", total_cols_removed)

        if report['total_rows'] > 0:
            remaining_pct = round((report['total_rows'] - total_rows_removed) / report['total_rows'] * 100, 1)
            st.markdown(f"**Estimated data retained: {remaining_pct}%** ({report['total_rows'] - total_rows_removed:,} rows)")

        # Apply button
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("Apply Selected Cleaning Actions", type="primary", use_container_width=True, key="apply_insights_clean"):
                if not selected:
                    st.warning("No actions selected. Please check at least one action above.")
                else:
                    with st.spinner("Applying selected cleaning actions..."):
                        df_clean, log = apply_selected_recommendations(df, selected, report)

                        st.success("Cleaning complete!")
                        st.markdown("**What was done:**")
                        for entry in log:
                            st.write(f"- {entry}")

                        # Before/after comparison
                        cmp1, cmp2 = st.columns(2)
                        with cmp1:
                            st.markdown("**Before:**")
                            st.write(f"Rows: {len(df):,}")
                            st.write(f"Columns: {len(df.columns)}")
                            st.write(f"Missing: {df.isnull().sum().sum():,}")
                        with cmp2:
                            st.markdown("**After:**")
                            st.write(f"Rows: {len(df_clean):,}")
                            st.write(f"Columns: {len(df_clean.columns)}")
                            st.write(f"Missing: {df_clean.isnull().sum().sum():,}")

                        rows_removed = len(df) - len(df_clean)
                        if rows_removed > 0:
                            retained = round(len(df_clean) / len(df) * 100, 1)
                            st.info(f"Removed {rows_removed:,} rows. {retained}% of data retained.")

                        # Preview
                        st.markdown("**Preview of cleaned data:**")
                        try:
                            st.dataframe(df_clean.head(10), use_container_width=True)
                        except pa.ArrowInvalid:
                            st.dataframe(df_clean.head(10).astype(str), use_container_width=True)

                        if st.button("Confirm and Apply", key="confirm_insights_clean"):
                            st.session_state['data'] = df_clean
                            st.success("Changes saved!")
                            st.rerun()


def analyze_cleaning_needs(df):
    """Analyze the dataset and return cleaning recommendations."""
    issues = []
    actions = []

    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            'type': 'duplicates',
            'severity': 'high' if dup_count > len(df) * 0.1 else 'medium',
            'message': f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)",
            'action': 'Remove duplicate rows'
        })
        actions.append({'type': 'remove_duplicates'})

    # Check for missing values
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            pct = missing / len(df) * 100
            if pct > 50:
                issues.append({
                    'type': 'missing_high',
                    'severity': 'high',
                    'column': col,
                    'message': f"Column '{col}' has {missing} missing values ({pct:.1f}%)",
                    'action': f"Consider dropping column '{col}'"
                })
                actions.append({'type': 'drop_column', 'column': col})
            else:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    strategy = 'median'
                else:
                    strategy = 'mode'
                issues.append({
                    'type': 'missing',
                    'severity': 'medium',
                    'column': col,
                    'message': f"Column '{col}' has {missing} missing values ({pct:.1f}%)",
                    'action': f"Impute with {strategy}"
                })
                actions.append({'type': 'impute', 'column': col, 'strategy': strategy})

    # Check for potential type issues
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Check if should be numeric
                try:
                    pd.to_numeric(non_null)
                    issues.append({
                        'type': 'type_mismatch',
                        'severity': 'low',
                        'column': col,
                        'message': f"Column '{col}' appears numeric but stored as text",
                        'action': 'Convert to numeric'
                    })
                    actions.append({'type': 'convert_numeric', 'column': col})
                except:
                    pass

    # Check for whitespace issues in string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().any():
            sample = df[col].dropna().head(100)
            has_whitespace = sample.apply(lambda x: str(x) != str(x).strip()).any()
            if has_whitespace:
                issues.append({
                    'type': 'whitespace',
                    'severity': 'low',
                    'column': col,
                    'message': f"Column '{col}' has leading/trailing whitespace",
                    'action': 'Trim whitespace'
                })
                actions.append({'type': 'trim', 'column': col})

    # Check for outliers in numeric columns
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > len(df) * 0.05:
            issues.append({
                'type': 'outliers',
                'severity': 'medium',
                'column': col,
                'message': f"Column '{col}' has {outliers} potential outliers ({outliers/len(df)*100:.1f}%)",
                'action': 'Review and handle outliers'
            })

    return issues, actions


def auto_clean_data(df, actions):
    """
    Apply automatic cleaning based on recommended actions.
    Order: duplicates first, then column drops, then imputation, then type conversion, then trim.
    This avoids cascade issues where column removal creates false duplicates.
    """
    df_clean = df.copy()
    log = []

    # Sort actions: duplicates first, then drops, then impute, then convert, then trim
    action_order = {
        'remove_duplicates': 0,
        'drop_column': 1,
        'impute': 2,
        'convert_numeric': 3,
        'trim': 4,
    }
    sorted_actions = sorted(actions, key=lambda a: action_order.get(a['type'], 99))

    for action in sorted_actions:
        try:
            if action['type'] == 'remove_duplicates':
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                removed = before - len(df_clean)
                log.append(f"Removed {removed:,} duplicate rows")

            elif action['type'] == 'drop_column':
                col = action['column']
                if col in df_clean.columns:
                    df_clean = df_clean.drop(columns=[col])
                    log.append(f"Dropped column '{col}'")

            elif action['type'] == 'impute':
                col = action['column']
                if col in df_clean.columns and df_clean[col].isnull().any():
                    n_missing = int(df_clean[col].isnull().sum())
                    if action['strategy'] == 'median':
                        value = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(value)
                        log.append(f"Filled {n_missing:,} missing in '{col}' with median ({value:.2f})")
                    elif action['strategy'] == 'mode':
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                            log.append(f"Filled {n_missing:,} missing in '{col}' with mode ({mode_val.iloc[0]})")

            elif action['type'] == 'convert_numeric':
                col = action['column']
                if col in df_clean.columns:
                    before_missing = int(df_clean[col].isnull().sum())
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    after_missing = int(df_clean[col].isnull().sum())
                    new_missing = after_missing - before_missing
                    msg = f"Converted '{col}' to numeric"
                    if new_missing > 0:
                        msg += f" ({new_missing} non-numeric values became missing)"
                    log.append(msg)

            elif action['type'] == 'trim':
                col = action['column']
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
                    log.append(f"Trimmed whitespace in '{col}'")

        except Exception as e:
            log.append(f"Error in {action['type']}: {str(e)}")

    return df_clean, log


def render_auto_clean(df):
    """Render the auto-clean tab with intelligent suggestions and one-click cleaning."""
    st.subheader("Intelligent Auto-Clean")
    st.markdown("Automatically detect and fix common data quality issues.")

    # Analyze data
    issues, actions = analyze_cleaning_needs(df)

    if not issues:
        st.success("No data quality issues detected! Your data is clean.")
        return

    # Display issues summary
    st.markdown("### Detected Issues")

    high_issues = [i for i in issues if i['severity'] == 'high']
    medium_issues = [i for i in issues if i['severity'] == 'medium']
    low_issues = [i for i in issues if i['severity'] == 'low']

    col1, col2, col3 = st.columns(3)
    col1.metric("High Severity", len(high_issues), help="Critical issues that should be addressed")
    col2.metric("Medium Severity", len(medium_issues), help="Important issues to review")
    col3.metric("Low Severity", len(low_issues), help="Minor issues for optimization")

    # Display issues by severity
    if high_issues:
        st.markdown("#### High Severity Issues")
        for issue in high_issues:
            st.error(f"{issue['message']} | **Suggested:** {issue['action']}")

    if medium_issues:
        st.markdown("#### Medium Severity Issues")
        for issue in medium_issues:
            st.warning(f"{issue['message']} | **Suggested:** {issue['action']}")

    if low_issues:
        with st.expander("Low Severity Issues", expanded=False):
            for issue in low_issues:
                st.info(f"{issue['message']} | **Suggested:** {issue['action']}")

    st.markdown("---")

    # Auto-clean options
    st.markdown("### Auto-Clean Options")

    col1, col2 = st.columns(2)

    with col1:
        # Select which actions to apply
        st.markdown("**Select actions to apply:**")

        dup_count = df.duplicated().sum()
        dup_label = f"Remove duplicates ({dup_count:,} rows)" if dup_count > 0 else "Remove duplicates (none found)"
        apply_duplicates = st.checkbox(dup_label, value=False, key="ac_dup",
                                       help="Removes exact duplicate rows. Disabled by default to prevent accidental data loss.")
        apply_imputation = st.checkbox("Impute missing values (median/mode)", value=True, key="ac_imp",
                                       help="Fills missing values without removing any rows.")
        apply_type_conversion = st.checkbox("Convert mismatched types", value=False, key="ac_type",
                                            help="Converts text columns that look numeric. Non-numeric values become missing.")
        apply_trim = st.checkbox("Trim whitespace", value=True, key="ac_trim",
                                  help="Removes leading/trailing spaces from text columns.")
        drop_high_missing = st.checkbox("Drop columns with >50% missing", value=False, key="ac_drop",
                                         help="Removes entire columns that are mostly empty.")

    with col2:
        st.markdown("**Preview changes:**")

        # Build filtered actions based on selections
        filtered_actions = []
        for action in actions:
            if action['type'] == 'remove_duplicates' and apply_duplicates:
                filtered_actions.append(action)
            elif action['type'] == 'impute' and apply_imputation:
                filtered_actions.append(action)
            elif action['type'] == 'drop_column' and drop_high_missing:
                filtered_actions.append(action)
            elif action['type'] == 'convert_numeric' and apply_type_conversion:
                filtered_actions.append(action)
            elif action['type'] == 'trim' and apply_trim:
                filtered_actions.append(action)

        st.write(f"**{len(filtered_actions)} actions selected**")

        for action in filtered_actions[:10]:
            if action['type'] == 'remove_duplicates':
                st.write(f"- Remove duplicates")
            elif action['type'] == 'impute':
                st.write(f"- Impute `{action['column']}` with {action['strategy']}")
            elif action['type'] == 'drop_column':
                st.write(f"- Drop column `{action['column']}`")
            elif action['type'] == 'convert_numeric':
                st.write(f"- Convert `{action['column']}` to numeric")
            elif action['type'] == 'trim':
                st.write(f"- Trim whitespace in `{action['column']}`")

        if len(filtered_actions) > 10:
            st.write(f"- ... and {len(filtered_actions) - 10} more actions")

    st.markdown("---")

    # Apply button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Apply Auto-Clean", type="primary", use_container_width=True):
            if not filtered_actions:
                st.warning("No actions selected. Please select at least one cleaning option.")
            else:
                with st.spinner("Cleaning data..."):
                    df_clean, log = auto_clean_data(df, filtered_actions)

                    # Show results
                    st.success("Auto-clean completed!")

                    st.markdown("**Cleaning Log:**")
                    for entry in log:
                        st.write(f"- {entry}")

                    # Compare before/after
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before:**")
                        st.write(f"Rows: {len(df):,}")
                        st.write(f"Columns: {len(df.columns)}")
                        st.write(f"Missing: {df.isnull().sum().sum():,}")

                    with col2:
                        st.markdown("**After:**")
                        st.write(f"Rows: {len(df_clean):,}")
                        st.write(f"Columns: {len(df_clean.columns)}")
                        st.write(f"Missing: {df_clean.isnull().sum().sum():,}")

                    # Preview cleaned data
                    st.markdown("**Preview of cleaned data:**")
                    try:
                        st.dataframe(df_clean.head(10), use_container_width=True)
                    except pa.ArrowInvalid:
                        st.warning("Displaying data as string due to mixed types.")
                        st.dataframe(df_clean.head(10).astype(str), use_container_width=True)

                    # Confirm button
                    if st.button("Confirm and Apply Changes", key="confirm_auto_clean"):
                        st.session_state['data'] = df_clean
                        st.success("Changes applied successfully!")
                        st.rerun()


def render():
    st.header("Data Cleaning")

    if st.session_state['data'] is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['data']

    # --- Data Health Insights (shown immediately) ---
    render_data_insights(df)

    st.markdown("---")
    st.markdown("---")

    # --- Detailed Cleaning Tools ---
    st.subheader("Detailed Cleaning Tools")
    st.markdown("Use the tabs below for granular control over specific cleaning operations.")

    # Create tabs for different cleaning operations
    tabs = st.tabs([
        "Auto-Clean",
        "Missing Values",
        "Duplicates",
        "Data Types",
        "String Cleaning",
        "Value Mapping",
        "Column Management",
        "Encoding"
    ])

    with tabs[0]:
        render_auto_clean(df)

    with tabs[1]:
        render_missing_values(df)

    with tabs[2]:
        render_duplicates(df)

    with tabs[3]:
        render_data_types(df)

    with tabs[4]:
        render_string_cleaning(df)

    with tabs[5]:
        render_value_mapping(df)

    with tabs[6]:
        render_column_management(df)

    with tabs[7]:
        render_encoding(df)

    st.markdown("---")

    # Show Data Preview
    st.subheader("Current Data Preview")
    try:
        st.dataframe(st.session_state['data'].head())
    except pa.ArrowInvalid:
        st.warning("Displaying data as string due to mixed types.")
        st.dataframe(st.session_state['data'].head().astype(str))


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
