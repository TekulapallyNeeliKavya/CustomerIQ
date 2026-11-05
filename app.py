"""
CustomerIQ - AI-Powered Customer Segmentation and Analytics System
Main Application Entrypoint and Multi-Page Navigation Hub
"""

import streamlit as st
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
    reset_to_default_dataset,
)

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="CustomerIQ – AI Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application session state & custom UI styling
initialize_app_state()
apply_custom_css()

# Modern Streamlit Page Navigation Architecture
pages = [
    st.Page("pages/dashboard.py", title="Executive Dashboard", icon="📊", default=True),
    st.Page("pages/data_analysis.py", title="Customer Data Analysis", icon="🔍"),
    st.Page("pages/clustering_analysis.py", title="Clustering Analysis", icon="📈"),
    st.Page("pages/customer_segments.py", title="Segment Visualization", icon="🎯"),
    st.Page("pages/customer_insights.py", title="Customer Insights", icon="💡"),
    st.Page("pages/customer_prediction.py", title="Customer Prediction", icon="🔮"),
    st.Page("pages/reports.py", title="Report Generation", icon="📑"),
]

pg = st.navigation(pages)

# Persistent Sidebar Brand & System Specs
with st.sidebar:
    st.markdown("### 🎯 **CustomerIQ**")
    st.caption("AI-Powered Customer Segmentation & Analytics System v1.0.0")

    # Dataset status indicator
    if st.session_state.get("is_custom_dataset", False):
        st.success("🟢 Active: Proprietary Dataset")
        if st.button("Reset to Default Mall Data", use_container_width=True):
            reset_to_default_dataset()
            st.rerun()
    else:
        st.info("🔵 Active: Mall Customers Dataset")

    st.markdown("---")
    st.markdown("**Core Architecture:**")
    st.markdown("- **Algorithm**: K-Means (`k-means++`)")
    st.markdown("- **Normalization**: `StandardScaler` (Z-Score)")
    st.markdown(f"- **Active Clusters**: K = {st.session_state.get('optimal_k', 5)}")
    st.markdown("- **Projection**: Principal Component Analysis (PCA)")
    st.markdown("---")
    st.caption("Enterprise AI/ML Analytics Suite")

# Run active routed page
pg.run()
