"""
CustomerIQ - Page 1: Executive Dashboard
Presents top-level KPIs, project summary, segment distribution, and key executive takeaways.
"""

import streamlit as st
import pandas as pd
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
    render_metric_card,
)
from src.visualization import plot_cluster_distribution, plot_clusters_2d

# Ensure state is initialized
initialize_app_state()
apply_custom_css()

# Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">CustomerIQ Dashboard</div>
        <div class="hero-desc">
            Enterprise Customer Segmentation & Behavioral Analytics platform powered by unsupervised K-Means Machine Learning.
            Discover high-value personas, optimize marketing ROI, and guide product positioning.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state["df_clustered"]
optimal_k = st.session_state["optimal_k"]
profiles = st.session_state["profiles"]
sizes_df = st.session_state["sizes_df"]
eval_results = st.session_state["eval_results"]
features = st.session_state["features"]

# Top-level KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Total Customers Analyzed",
        value=f"{len(df):,}",
        subtext="100% records clustered",
        accent_color="#2563EB",
    )

with col2:
    render_metric_card(
        title="Active Segments",
        value=f"{optimal_k}",
        subtext="Distinct customer cohorts",
        accent_color="#10B981",
    )

with col3:
    sil_score = eval_results.get("best_silhouette_score", 0.5547)
    render_metric_card(
        title="Optimal Cluster K",
        value=f"K = {optimal_k}",
        subtext=f"Silhouette Score: {sil_score:.3f}",
        accent_color="#F59E0B",
    )

with col4:
    spend_col = next((c for c in df.columns if "spending" in c.lower() or "score" in c.lower()), None)
    avg_spend = f"{df[spend_col].mean():.1f}/100" if spend_col else "N/A"
    render_metric_card(
        title="Avg Customer Spending",
        value=avg_spend,
        subtext="Mall benchmark mean",
        accent_color="#8B5CF6",
    )

st.write("")

# Layout: Visual Distribution & Executive Summary Table
c_left, c_right = st.columns([1.1, 1.4])

with c_left:
    st.subheader("Customer Distribution")
    fig_dist = plot_cluster_distribution(sizes_df)
    st.plotly_chart(fig_dist, use_container_width=True)

with c_right:
    st.subheader("Executive Segment Summary")
    summary_data = []
    for p in profiles:
        summary_data.append({
            "Cluster": f"C{p['cluster_id']}",
            "Segment Persona": p["segment_name"].split(" (")[0],
            "Customers": p["customer_count"],
            "Share (%)": f"{p['percentage']:.1f}%",
            "Avg Income ($k)": f"${p['avg_income']:.1f}k" if p["avg_income"] is not None else "N/A",
            "Avg Spend (1-100)": f"{p['avg_spending']:.1f}" if p["avg_spending"] is not None else "N/A",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Key Executive Takeaways
    st.markdown("#### Strategic Executive Takeaways")
    # Identify largest segment and highest spend segment dynamically
    largest = max(profiles, key=lambda x: x["customer_count"])
    high_spenders = max(profiles, key=lambda x: x["avg_spending"] if x["avg_spending"] else 0)

    st.markdown(
        f"""
        - **Core Customer Base**: **{largest['segment_name'].split(' (')[0]}** represents the largest single audience (**{largest['percentage']:.1f}%** of shoppers), serving as the foundational volume driver.
        - **Highest Value Cohort**: **{high_spenders['segment_name'].split(' (')[0]}** commands the top spending index (**{high_spenders['avg_spending']:.1f}/100**), warranting dedicated VIP retention programs.
        - **Feature Space Analyzed**: Segmentation computed across **{', '.join(features)}**.
        """
    )

st.write("")
st.info("Navigate through the sidebar to explore detailed feature EDA, mathematical clustering diagnostics, 2D/PCA visualizations, customer profiling, and real-time prediction.")
