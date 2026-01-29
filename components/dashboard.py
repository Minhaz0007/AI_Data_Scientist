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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{len(st.session_state['dashboard_charts'])} Saved Charts**")
    with col2:
        layout = st.selectbox("Layout", ["1 Column", "2 Columns", "3 Columns"], index=1)

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
