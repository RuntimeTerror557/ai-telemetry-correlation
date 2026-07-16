"""
Streamlit dashboard for SOC / Fraud analysts.

Run with:
    streamlit run dashboard/app.py
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="AI Correlation Engine | SOC + Fraud Console", layout="wide")

PRIMARY = "#215868"
ACCENT = "#FFD449"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_data():
    scored = pd.read_csv(os.path.join(DATA_DIR, "all_scored_transactions.csv"))
    scored["timestamp"] = pd.to_datetime(scored["timestamp"])
    return scored


def main():
    st.markdown(
        f"<h1 style='color:{PRIMARY};margin-bottom:0;'>Unified Telemetry-Transaction Risk Console</h1>"
        f"<p style='color:#555;margin-top:4px;'>AI-Driven Correlation of Cybersecurity Telemetry &amp; Transactional Behaviour</p>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(os.path.join(DATA_DIR, "all_scored_transactions.csv")):
        st.warning("No scored data found. Run `python -m src.score_batch` first.")
        return

    df = load_data()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")
    bands = st.sidebar.multiselect("Risk band", options=sorted(df.risk_band.unique()),
                                    default=sorted(df.risk_band.unique()))
    channels = st.sidebar.multiselect("Channel", options=sorted(df.channel.unique()),
                                       default=sorted(df.channel.unique()))
    min_score = st.sidebar.slider("Minimum risk score", 0, 100, 0)

    filtered = df[
        df.risk_band.isin(bands) & df.channel.isin(channels) & (df.risk_score >= min_score)
    ]

    # ---- KPI row ----
    c1, c2, c3, c4 = st.columns(4)
    total_txn = len(df)
    total_alerts = int((df.risk_score >= 50).sum())
    critical = int((df.risk_band == "Critical").sum())
    avg_score = df.risk_score.mean()

    c1.metric("Transactions scored", f"{total_txn:,}")
    c2.metric("High/Critical alerts", f"{total_alerts:,}")
    c3.metric("Critical (correlated) cases", f"{critical:,}")
    c4.metric("Avg. risk score", f"{avg_score:.1f}")

    st.divider()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        band_counts = df.risk_band.value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig = px.bar(
            x=band_counts.index, y=band_counts.values,
            labels={"x": "Risk band", "y": "Transactions"},
            title="Alert Volume by Risk Band",
            color=band_counts.index,
            color_discrete_map={"Low": "#BFD8DC", "Medium": ACCENT, "High": "#1C7293", "Critical": PRIMARY},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        by_day = df.set_index("timestamp").resample("D")["risk_score"].apply(lambda s: (s >= 50).sum())
        fig2 = px.line(x=by_day.index, y=by_day.values, markers=True,
                        labels={"x": "Date", "y": "High/Critical alerts"},
                        title="Daily Alert Trend")
        fig2.update_traces(line_color=PRIMARY)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Alert Queue")
    st.caption("Ranked by correlated risk score. Select a transaction below to drill into its full evidence trail.")

    show_cols = ["txn_id", "customer_id", "timestamp", "amount", "channel", "rule_score", "ml_score", "risk_score", "risk_band"]
    st.dataframe(
        filtered.sort_values("risk_score", ascending=False)[show_cols].head(200),
        use_container_width=True, height=320,
    )

    st.divider()
    st.subheader("Case Drill-Down")
    alert_ids = filtered.sort_values("risk_score", ascending=False).txn_id.head(100).tolist()
    if alert_ids:
        selected = st.selectbox("Select a transaction ID", alert_ids)
        case = df[df.txn_id == selected].iloc[0]

        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown(f"**Customer:** {case.customer_id}")
            st.markdown(f"**Timestamp:** {case.timestamp}")
            st.markdown(f"**Amount:** ₹{case.amount:,.2f}")
            st.markdown(f"**Channel:** {case.channel}")
            st.markdown(f"**Risk score:** :red[{case.risk_score}] ({case.risk_band})")
            st.markdown(f"**Rule-based component:** {case.rule_score}")
            st.markdown(f"**ML component:** {case.ml_score}")
        with cc2:
            st.markdown("**Why this was flagged:**")
            reasons = case.reasons
            if isinstance(reasons, str):
                # stored as a python-list-repr string when round-tripped through CSV
                import ast
                try:
                    reasons = ast.literal_eval(reasons)
                except Exception:
                    reasons = [reasons]
            for r in reasons:
                st.markdown(f"- {r}")
    else:
        st.info("No transactions match the current filters.")


if __name__ == "__main__":
    main()
