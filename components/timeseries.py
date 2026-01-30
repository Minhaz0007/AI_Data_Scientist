"""
Time Series Analysis & Forecasting Component
Provides comprehensive time series analysis, decomposition, and forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.ml_engine import MLEngine


def render():
    """Render the Time Series Analysis page."""
    if st.session_state['data'] is None:
        st.warning("Please load data first in the Data Ingestion page.")
        return

    df = st.session_state['data']
    ml = MLEngine()

    st.markdown("""
    Analyze time series data with decomposition, stationarity testing, and forecasting.
    Supports **ARIMA** and **Exponential Smoothing** models.
    """)

    # Column Selection
    col1, col2 = st.columns(2)

    with col1:
        # Find datetime columns
        datetime_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                else:
                    # Try to parse
                    pd.to_datetime(df[col].head(10), errors='raise')
                    datetime_cols.append(col)
            except:
                pass

        if not datetime_cols:
            datetime_cols = df.columns.tolist()

        date_col = st.selectbox(
            "Select Date/Time Column",
            options=datetime_cols,
            help="Column containing date/time values"
        )

    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox(
            "Select Value Column",
            options=numeric_cols,
            help="Column containing the values to analyze"
        )

    if not date_col or not value_col:
        st.info("Please select both date and value columns.")
        return

    # Frequency Selection
    freq = st.selectbox(
        "Data Frequency",
        options=['D', 'W', 'M', 'Q', 'Y', 'H'],
        format_func=lambda x: {
            'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly',
            'Q': 'Quarterly', 'Y': 'Yearly', 'H': 'Hourly'
        }.get(x, x),
        index=0
    )

    st.markdown("---")

    # Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["Time Series Overview", "Decomposition", "Forecasting"])

    with tab1:
        st.subheader("Time Series Overview")

        with st.spinner("Analyzing time series..."):
            try:
                results = ml.analyze_time_series(df, date_col, value_col, freq=freq)

                # Basic Stats
                stats = results['basic_stats']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{stats['mean']:.2f}")
                col2.metric("Std Dev", f"{stats['std']:.2f}")
                col3.metric("Min", f"{stats['min']:.2f}")
                col4.metric("Max", f"{stats['max']:.2f}")

                st.metric("Trend Direction", stats['trend'].capitalize())

                # Time Series Plot
                series = results['series']
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name=value_col,
                    line=dict(color='blue')
                ))

                # Add rolling mean
                window = min(7, len(series) // 4)
                if window > 1:
                    rolling_mean = series.rolling(window=window).mean()
                    fig.add_trace(go.Scatter(
                        x=rolling_mean.index,
                        y=rolling_mean.values,
                        mode='lines',
                        name=f'{window}-Period Moving Average',
                        line=dict(color='red', dash='dash')
                    ))

                fig.update_layout(
                    title='Time Series Plot',
                    xaxis_title='Date',
                    yaxis_title=value_col,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stationarity Test
                st.subheader("Stationarity Analysis")
                if results.get('stationarity'):
                    stat_result = results['stationarity']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ADF Statistic", stat_result['adf_statistic'])
                    col2.metric("P-Value", stat_result['p_value'])
                    col3.metric("Is Stationary?", "Yes" if stat_result['is_stationary'] else "No")

                    if stat_result['is_stationary']:
                        st.success("The time series is stationary (p < 0.05)")
                    else:
                        st.warning("The time series is non-stationary. Consider differencing for forecasting.")
                else:
                    st.info("Unable to perform stationarity test.")

                # ACF/PACF Plots
                if 'acf' in results and 'pacf' in results:
                    st.subheader("Autocorrelation Analysis")

                    fig_acf = make_subplots(rows=1, cols=2,
                                           subplot_titles=['Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)'])

                    acf_vals = results['acf']
                    pacf_vals = results['pacf']
                    lags = list(range(len(acf_vals)))

                    fig_acf.add_trace(
                        go.Bar(x=lags, y=acf_vals, name='ACF', marker_color='blue'),
                        row=1, col=1
                    )
                    fig_acf.add_trace(
                        go.Bar(x=lags, y=pacf_vals, name='PACF', marker_color='green'),
                        row=1, col=2
                    )

                    # Confidence bands
                    conf = 1.96 / np.sqrt(len(series))
                    for col_idx in [1, 2]:
                        fig_acf.add_hline(y=conf, line_dash='dash', line_color='red',
                                         row=1, col=col_idx)
                        fig_acf.add_hline(y=-conf, line_dash='dash', line_color='red',
                                         row=1, col=col_idx)

                    fig_acf.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_acf, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing time series: {str(e)}")

    with tab2:
        st.subheader("Seasonal Decomposition")

        try:
            results = ml.analyze_time_series(df, date_col, value_col, freq=freq)

            if results.get('decomposition'):
                decomp = results['decomposition']

                # Create subplots
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                    shared_xaxes=True,
                    vertical_spacing=0.05
                )

                series = results['series']

                fig.add_trace(
                    go.Scatter(x=series.index, y=series.values, name='Original', line=dict(color='blue')),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=decomp['trend'].index, y=decomp['trend'].values,
                              name='Trend', line=dict(color='green')),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=decomp['seasonal'].index, y=decomp['seasonal'].values,
                              name='Seasonal', line=dict(color='orange')),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(x=decomp['residual'].index, y=decomp['residual'].values,
                              name='Residual', line=dict(color='red')),
                    row=4, col=1
                )

                fig.update_layout(height=800, showlegend=False, title='Seasonal Decomposition')
                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                st.markdown("""
                **Interpretation:**
                - **Trend**: Long-term increase or decrease in the data
                - **Seasonal**: Repeating patterns at fixed intervals
                - **Residual**: Random variation after removing trend and seasonality
                """)
            else:
                st.warning("Unable to perform seasonal decomposition. Need more data points.")

        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")

    with tab3:
        st.subheader("Time Series Forecasting")

        col1, col2 = st.columns(2)

        with col1:
            forecast_periods = st.slider(
                "Forecast Periods",
                min_value=7,
                max_value=365,
                value=30,
                help="Number of future periods to forecast"
            )

        with col2:
            forecast_method = st.selectbox(
                "Forecasting Method",
                options=['auto', 'arima', 'exponential_smoothing'],
                format_func=lambda x: {
                    'auto': 'Auto (Best Model + Auto-ARIMA Tuning)',
                    'arima': 'ARIMA (Standard)',
                    'exponential_smoothing': 'Exponential Smoothing'
                }.get(x, x)
            )

        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                try:
                    results = ml.forecast_time_series(
                        df, date_col, value_col,
                        periods=forecast_periods,
                        method=forecast_method
                    )

                    if 'error' in results:
                        st.error(f"Forecasting error: {results['error']}")
                    else:
                        # Show metrics
                        st.subheader("Model Performance")

                        if 'arima' in results:
                            col1, col2 = st.columns(2)
                            col1.metric("ARIMA MAPE", f"{results['arima']['validation_mape']:.2f}%")
                            col1.metric("ARIMA AIC", results['arima']['aic'])

                        if 'exp_smoothing' in results:
                            if 'arima' in results:
                                col2.metric("Exp. Smoothing MAPE", f"{results['exp_smoothing']['validation_mape']:.2f}%")
                            else:
                                st.metric("Exp. Smoothing MAPE", f"{results['exp_smoothing']['validation_mape']:.2f}%")

                        if 'best_method' in results:
                            st.success(f"Best Method: **{results['best_method'].replace('_', ' ').title()}**")

                        # Forecast Plot
                        st.subheader("Forecast Visualization")

                        # Get original series
                        ts_results = ml.analyze_time_series(df, date_col, value_col, freq=freq)
                        series = ts_results['series']

                        fig = go.Figure()

                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))

                        # Forecast
                        if 'best_forecast' in results:
                            forecast = results['best_forecast']
                            fig.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast.values,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red', dash='dash')
                            ))

                            # Confidence interval (approximate)
                            std = series.std()
                            fig.add_trace(go.Scatter(
                                x=list(forecast.index) + list(forecast.index)[::-1],
                                y=list(forecast.values + 1.96*std) + list(forecast.values - 1.96*std)[::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.1)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='95% Confidence Interval'
                            ))

                        fig.update_layout(
                            title='Time Series Forecast',
                            xaxis_title='Date',
                            yaxis_title=value_col,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Forecast Table
                        st.subheader("Forecast Values")
                        if 'best_forecast' in results:
                            forecast_df = pd.DataFrame({
                                'Date': results['best_forecast'].index,
                                'Forecast': results['best_forecast'].values.round(2)
                            })
                            st.dataframe(forecast_df, use_container_width=True)

                            # Download option
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                "Download Forecast CSV",
                                csv,
                                "forecast.csv",
                                "text/csv"
                            )

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

    # Tips Section
    with st.expander("Time Series Analysis Tips"):
        st.markdown("""
        **Best Practices:**
        1. **Data Preparation**: Ensure your date column is properly formatted
        2. **Frequency**: Choose the correct frequency for your data (daily, weekly, etc.)
        3. **Stationarity**: Non-stationary data may need differencing before modeling
        4. **Seasonality**: Look for repeating patterns in the decomposition

        **Model Selection:**
        - **ARIMA**: Good for data with trends and autocorrelation
        - **Exponential Smoothing**: Good for data with trends and seasonality
        - **Auto**: Tries both and picks the best based on validation error

        **Interpreting Results:**
        - **MAPE** (Mean Absolute Percentage Error): Lower is better, <10% is excellent
        - **AIC** (Akaike Information Criterion): Lower indicates better model fit
        """)
