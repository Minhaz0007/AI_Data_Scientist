import streamlit as st
import plotly.graph_objects as go

def render():
    st.header("Dashboard")

    if 'dashboard_charts' not in st.session_state or not st.session_state['dashboard_charts']:
        st.info("Your dashboard is empty. Go to the **Visualization** page to create and pin charts here.")
        if st.button("Go to Visualization"):
            st.session_state['current_page'] = "Visualization"
            st.rerun()
        return

    # Dashboard Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**{len(st.session_state['dashboard_charts'])} Saved Charts**")
    with col2:
        layout = st.selectbox("Layout", ["1 Column", "2 Columns", "3 Columns"], index=1)
    with col3:
        if st.checkbox("Auto-Refresh (30s)"):
            import time
            time.sleep(30)
            st.rerun()

    # Persistence
    with st.expander("💾 Save / Load Dashboard"):
        c1, c2 = st.columns(2)
        with c1:
            # Export
            if st.button("Export Configuration"):
                import json
                try:
                    export_data = []
                    for chart in st.session_state['dashboard_charts']:
                         # Convert figure to JSON
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
            # Import
            uploaded_config = st.file_uploader("Load Configuration", type=['json'])
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

    # Display Charts
    charts = st.session_state['dashboard_charts']

    # Iterate in chunks based on columns
    for i in range(0, len(charts), cols_count):
        cols = st.columns(cols_count)
        for j in range(cols_count):
            if i + j < len(charts):
                chart_data = charts[i + j]
                with cols[j]:
                    with st.container():
                        st.markdown(f"### {chart_data.get('title', 'Untitled Chart')}")
                        st.plotly_chart(chart_data['figure'], use_container_width=True)

                        c1, c2 = st.columns([1, 4])
                        with c1:
                            if st.button("🗑️", key=f"del_{i+j}", help="Remove from dashboard"):
                                st.session_state['dashboard_charts'].pop(i + j)
                                st.rerun()
                        with c2:
                            st.caption(f"Type: {chart_data.get('type', 'Unknown')}")

                        st.markdown("---")
