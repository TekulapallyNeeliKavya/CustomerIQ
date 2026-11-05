"""
CustomerIQ - Page 5: Customer Insights & Business Recommendations
Translates mathematical cluster centroids into actionable marketing personas
and data-backed strategic recommendations.
"""

import streamlit as st
from src.app_state import (
    initialize_app_state,
    apply_custom_css,
)

initialize_app_state()
apply_custom_css()

st.title("Customer Insights & Business Strategy")
st.markdown("Actionable behavioral archetypes and tailored marketing playbooks derived from empirical cluster centroids.")

profiles = st.session_state["profiles"]
optimal_k = st.session_state["optimal_k"]

# Distinct badge accent colors for clusters
CLUSTER_COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"]

for p in profiles:
    c_id = p["cluster_id"]
    color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]

    with st.container():
        st.markdown(
            f"""
            <div class="cluster-card" style="border-left-color: {color};">
                <div class="cluster-card-header">
                    <div>
                        <span class="cluster-badge" style="background-color: {color}20; color: {color};">
                            Cluster {c_id}
                        </span>
                        <span style="font-size: 1.25rem; font-weight: 700; margin-left: 10px; color: #1E293B;">
                            {p['segment_name']}
                        </span>
                    </div>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #64748B;">
                        {p['archetype_category']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics row for this cluster
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Segment Size", f"{p['customer_count']} customers", f"{p['percentage']:.1f}% of total")
        with m2:
            inc_val = f"${p['avg_income']:.1f}k" if p["avg_income"] is not None else "N/A"
            st.metric("Avg Annual Income", inc_val)
        with m3:
            sp_val = f"{p['avg_spending']:.1f}/100" if p["avg_spending"] is not None else "N/A"
            st.metric("Avg Spending Score", sp_val)
        with m4:
            st.metric("Avg Age", f"{p['avg_age']} yrs" if p["avg_age"] != "N/A" else "N/A")

        # Two columns: Characteristics on left, Recommendations on right
        c_left, c_right = st.columns([1, 1.4])

        with c_left:
            st.markdown("##### Empirical Characteristics")
            for char in p["important_characteristics"]:
                st.markdown(f"- {char}")

        with c_right:
            st.markdown("##### Actionable Business Playbook (2–3 Strategies)")
            for rec in p["recommendations"]:
                st.markdown(
                    f"""
                    <div class="rec-item" style="border-left-color: {color};">
                        <div class="rec-title">{rec['strategy']}</div>
                        <div class="rec-body"><strong>Action:</strong> {rec['action']}</div>
                        <div class="rec-body" style="color: #64748B;"><strong>Recommended Channel:</strong> {rec['channel']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

# Strategic Advisory Disclaimer
st.caption(
    "Disclaimer: Machine learning recommendations represent data-informed behavioral hypotheses "
    "based on empirical cluster distributions. Strategic outcomes should be validated via iterative A/B testing "
    "and cohort tracking rather than presumed as guaranteed commercial results."
)
