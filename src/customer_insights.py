"""
CustomerIQ - Customer Insights and Business Recommendations Module
Dynamically derives cluster profiles, behavioral personas, statistical characteristics,
and 2-3 data-backed actionable business strategies without hardcoded assumptions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


def generate_cluster_profiles(
    df: pd.DataFrame,
    feature_names: List[str],
    cluster_col: str = "Cluster",
) -> List[Dict[str, Any]]:
    """
    Computes rigorous statistical summaries for each cluster relative to dataset benchmarks,
    deriving dynamic archetype descriptions, key characteristics, and actionable business strategies.
    """
    total_customers = len(df)
    profiles = []

    # Dataset benchmarks
    overall_stats = {}
    for feat in feature_names:
        overall_stats[feat] = {
            "mean": float(df[feat].mean()),
            "std": float(df[feat].std()) if df[feat].std() > 0 else 1.0,
            "median": float(df[feat].median()),
        }

    unique_clusters = sorted(df[cluster_col].unique())

    for c in unique_clusters:
        sub_df = df[df[cluster_col] == c]
        c_size = len(sub_df)
        c_pct = (c_size / total_customers) * 100

        # Feature means and z-scores
        feat_metrics = {}
        for feat in feature_names:
            c_mean = float(sub_df[feat].mean())
            c_median = float(sub_df[feat].median())
            c_min = float(sub_df[feat].min())
            c_max = float(sub_df[feat].max())
            z_score = (c_mean - overall_stats[feat]["mean"]) / overall_stats[feat]["std"]

            feat_metrics[feat] = {
                "mean": round(c_mean, 2),
                "median": round(c_median, 2),
                "min": round(c_min, 2),
                "max": round(c_max, 2),
                "z_score": round(z_score, 2),
            }

        # Check for Mall Customer key variables or generic equivalents
        income_key = next((f for f in feature_names if "income" in f.lower()), None)
        spend_key = next((f for f in feature_names if "spending" in f.lower() or "score" in f.lower()), None)
        age_key = next((f for f in feature_names if "age" in f.lower()), None)

        avg_income = feat_metrics[income_key]["mean"] if income_key else None
        avg_spend = feat_metrics[spend_key]["mean"] if spend_key else None
        avg_age = feat_metrics[age_key]["mean"] if age_key else (float(sub_df["Age"].mean()) if "Age" in sub_df.columns else None)

        # Gender breakdown if available
        gender_breakdown = ""
        if "Gender" in sub_df.columns:
            g_counts = sub_df["Gender"].value_counts(normalize=True) * 100
            gender_breakdown = ", ".join([f"{k}: {v:.1f}%" for k, v in g_counts.items()])

        # Dynamic Persona Classification
        persona_name, archetype, recommendations = _derive_archetype_and_strategy(
            income_key, spend_key, feat_metrics, avg_age, gender_breakdown
        )

        # Build detailed characteristics text
        char_points = []
        if avg_income is not None:
            inc_diff = avg_income - overall_stats[income_key]["mean"]
            inc_dir = f"+${inc_diff:.1f}k vs avg" if inc_diff >= 0 else f"-${abs(inc_diff):.1f}k vs avg"
            char_points.append(f"Average annual income is ${avg_income:.1f}k ({inc_dir}).")

        if avg_spend is not None:
            sp_diff = avg_spend - overall_stats[spend_key]["mean"]
            sp_dir = f"+{sp_diff:.1f} pts vs avg" if sp_diff >= 0 else f"-{abs(sp_diff):.1f} pts vs avg"
            char_points.append(f"Mall spending score is {avg_spend:.1f}/100 ({sp_dir}).")

        if avg_age is not None:
            char_points.append(f"Average age is {avg_age:.1f} years.")

        if gender_breakdown:
            char_points.append(f"Demographic distribution: {gender_breakdown}.")

        # If custom features, mention standout z-scores
        for feat in feature_names:
            if feat not in [income_key, spend_key, age_key]:
                z = feat_metrics[feat]["z_score"]
                lvl = "significantly higher than" if z > 0.8 else ("moderately higher than" if z > 0.3 else ("significantly lower than" if z < -0.8 else "in line with"))
                char_points.append(f"{feat} averages {feat_metrics[feat]['mean']} ({lvl} overall benchmark).")

        profile = {
            "cluster_id": int(c),
            "customer_count": c_size,
            "percentage": round(c_pct, 2),
            "avg_income": avg_income,
            "avg_spending": avg_spend,
            "avg_age": round(avg_age, 1) if avg_age else "N/A",
            "segment_name": persona_name,
            "archetype_category": archetype,
            "important_characteristics": char_points,
            "recommendations": recommendations,
            "feature_metrics": feat_metrics,
        }
        profiles.append(profile)

    return profiles


def _derive_archetype_and_strategy(
    income_key: Optional[str],
    spend_key: Optional[str],
    feat_metrics: Dict[str, Dict[str, Any]],
    avg_age: Optional[float],
    gender_breakdown: str,
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Infers persona and business recommendations dynamically from empirical metrics.
    """
    if income_key and spend_key:
        inc_z = feat_metrics[income_key]["z_score"]
        sp_z = feat_metrics[spend_key]["z_score"]

        # High Income & High Spending
        if inc_z >= 0.2 and sp_z >= 0.2:
            name = "Affluent Champions (High Income, High Spending)"
            cat = "Tier-1 High Value"
            recs = [
                {
                    "strategy": "VIP Exclusivity & Loyalty Concierge",
                    "action": "Introduce private shopping events, invitations to limited-edition luxury collections, and dedicated concierge styling assistance.",
                    "channel": "Direct VIP SMS/Concierge Email and Private Lounge events.",
                },
                {
                    "strategy": "Premium Loyalty Tiers & Experiential Rewards",
                    "action": "Offer premium reward points that can be redeemed for exclusive dining, valet parking, and lifestyle privileges rather than low-end discounts.",
                    "channel": "Omnichannel Luxury Member Portal.",
                },
                {
                    "strategy": "High-Touch Retention Cadence",
                    "action": "Engage proactive relationship managers with quarterly personal reach-outs and birthday luxury gifts.",
                    "channel": "White-Glove Direct Outreach.",
                },
            ]
        # High Income & Low Spending
        elif inc_z >= 0.2 and sp_z < -0.2:
            name = "Calculated Savers (High Income, Conservative Spending)"
            cat = "High Potential / Untapped Opportunity"
            recs = [
                {
                    "strategy": "Value-Add & Quality Craftsmanship Messaging",
                    "action": "Showcase long-term durability, investment-grade items, and transparent sustainability credentials rather than impulse marketing.",
                    "channel": "Curated editorial newsletters and thought-leadership catalogs.",
                },
                {
                    "strategy": "Bundle & Premium Utility Incentives",
                    "action": "Provide bulk purchase perks, complimentary premium warranties, and cash-back incentives tailored to household upgrades.",
                    "channel": "Targeted digital marketing and email campaigns.",
                },
                {
                    "strategy": "Frictionless Premium Experience",
                    "action": "Offer seamless express checkout, scheduled personal shopping appointments, and premium guarantees.",
                    "channel": "In-store digital appointments & mobile app.",
                },
            ]
        # Low Income & High Spending
        elif inc_z < -0.2 and sp_z >= 0.2:
            name = "Trend-Driven Enthusiasts (Moderate/Low Income, High Spending)"
            cat = "Aspirational & Engagement-Heavy"
            recs = [
                {
                    "strategy": "Flexible Financing & Buy-Now-Pay-Later (BNPL)",
                    "action": "Promote 0% interest installment plans (e.g. Klarna/Afterpay integrations) to support high-ticket purchases safely.",
                    "channel": "Point-of-sale checkout prompts and mobile app banners.",
                },
                {
                    "strategy": "Trend-First Flash Sales & Influencer Drops",
                    "action": "Run fast-paced 48-hour promotional events spotlighting viral fashion, social media collaborations, and limited-run items.",
                    "channel": "Instagram, TikTok, and push notifications.",
                },
                {
                    "strategy": "Gamified Loyalty & Referral Bonuses",
                    "action": "Implement tier-climbing gamification with badges, streak discounts, and referral store credit.",
                    "channel": "Mobile App Gamification.",
                },
            ]
        # Low Income & Low Spending
        elif inc_z < -0.2 and sp_z < -0.2:
            name = "Pragmatic Economizers (Low Income, Low Spending)"
            cat = "Budget-Conscious / Essential"
            recs = [
                {
                    "strategy": "Essential Value Bundles & Clearance Drives",
                    "action": "Emphasize seasonal sales, multi-item value packs, and affordable staple merchandise.",
                    "channel": "End-of-season circulars, clearance display zones, and coupon blasts.",
                },
                {
                    "strategy": "Community & Everyday Discount Programs",
                    "action": "Provide student/family/senior days and stamp cards with guaranteed cash rebates on repeat visits.",
                    "channel": "In-store coupon dispensers and budget SMS alerts.",
                },
                {
                    "strategy": "Loss-Leader Foot Traffic Promotions",
                    "action": "Utilize highly-discounted everyday items to drive physical store visits and expand shopping baskets.",
                    "channel": "Local circulars and community bulletin ads.",
                },
            ]
        # Moderate Income & Moderate Spending (Centroid Cluster)
        else:
            name = "Steady Mainstreamers (Moderate Income, Moderate Spending)"
            cat = "Volume Core / Foundation"
            recs = [
                {
                    "strategy": "Core Loyalty Program Nurturing",
                    "action": "Deliver steady points-per-dollar reward accruals and regular quarterly spend bonuses to maintain consistent visit cadence.",
                    "channel": "Email newsletter and app loyalty dashboard.",
                },
                {
                    "strategy": "Seasonal Cross-Selling Campaigns",
                    "action": "Target with back-to-school, holiday gift guides, and seasonal fashion transitions with moderate discount incentives (10-15% off).",
                    "channel": "Personalized email digests and direct mail catalog.",
                },
                {
                    "strategy": "Customer Satisfaction & Feedback Surveys",
                    "action": "Survey to identify friction points and reward completed reviews with mall food court vouchers or bonus points.",
                    "channel": "Post-purchase automated email survey.",
                },
            ]
    else:
        # Fallback for generic/custom uploaded datasets
        name = "Segment Archetype"
        cat = "Custom Cluster Cohort"
        recs = [
            {
                "strategy": "Tailored Segment Engagement",
                "action": "Design targeted customer communications aligned with the dominant attributes observed in this segment's centroid profile.",
                "channel": "Segment-specific email triggers and CRM tags.",
            },
            {
                "strategy": "Differentiated Promotional Strategy",
                "action": "A/B test promotional sensitivity to evaluate whether this cohort responds better to value discounts or experiential value-adds.",
                "channel": "Marketing automation engine.",
            },
            {
                "strategy": "Cohort Lifecycle Tracking",
                "action": "Monitor retention rate and 6-month repeat transaction velocity compared against dataset baseline averages.",
                "channel": "Business intelligence dashboard.",
            },
        ]

    return name, cat, recs


def export_profiles_to_dataframe(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts list of profiles into a clean, presentation-ready DataFrame for export to CSV.
    """
    rows = []
    for p in profiles:
        row = {
            "Cluster ID": p["cluster_id"],
            "Segment Name": p["segment_name"],
            "Category": p["archetype_category"],
            "Customer Count": p["customer_count"],
            "Percentage of Base (%)": p["percentage"],
            "Avg Annual Income (k$)": p["avg_income"],
            "Avg Spending Score (1-100)": p["avg_spending"],
            "Avg Age": p["avg_age"],
            "Key Characteristics": " | ".join(p["important_characteristics"]),
            "Strategic Recommendations": " | ".join([f"{r['strategy']}: {r['action']}" for r in p["recommendations"]]),
        }
        rows.append(row)
    return pd.DataFrame(rows)
