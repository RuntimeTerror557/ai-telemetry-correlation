"""
Renders a static PNG snapshot of the risk console -- handy as a demo
screenshot / for the pitch deck when a live Streamlit session isn't
available. Run after `python -m src.score_batch`.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PRIMARY = "#215868"
ACCENT = "#FFD449"
LIGHT = "#EAF2F3"

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "all_scored_transactions.csv")
OUT = os.path.join(HERE, "risk_console_snapshot.png")


def main():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    fig.suptitle("Unified Telemetry-Transaction Risk Console", fontsize=18, fontweight="bold",
                 color=PRIMARY, x=0.03, ha="left")
    fig.text(0.03, 0.925, "AI-Driven Correlation of Cybersecurity Telemetry & Transactional Behaviour",
              fontsize=10, color="#555555")

    # KPI cards
    kpis = [
        ("Transactions scored", f"{len(df):,}"),
        ("High/Critical alerts", f"{(df.risk_score >= 50).sum():,}"),
        ("Critical cases", f"{(df.risk_band == 'Critical').sum():,}"),
        ("Avg. risk score", f"{df.risk_score.mean():.1f}"),
    ]
    for i, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        x0 = 0.02 + i * 0.245
        ax.add_patch(plt.Rectangle((x0, 0.05), 0.22, 0.85, transform=ax.transAxes,
                                    facecolor=LIGHT, edgecolor="none"))
        ax.text(x0 + 0.11, 0.62, value, transform=ax.transAxes, ha="center", va="center",
                fontsize=20, fontweight="bold", color=PRIMARY)
        ax.text(x0 + 0.11, 0.25, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=9.5, color="#444444")

    # Bar chart: alerts by band
    ax1 = fig.add_subplot(gs[1, 0])
    band_order = ["Low", "Medium", "High", "Critical"]
    counts = df.risk_band.value_counts().reindex(band_order).fillna(0)
    colors = ["#BFD8DC", ACCENT, "#1C7293", PRIMARY]
    ax1.bar(counts.index, counts.values, color=colors)
    ax1.set_title("Alert Volume by Risk Band", fontsize=11, fontweight="bold", color=PRIMARY)
    ax1.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + max(counts.values) * 0.02, f"{int(v)}", ha="center", fontsize=9)

    # Line chart: daily alerts
    ax2 = fig.add_subplot(gs[1, 1])
    by_day = df.set_index("timestamp").resample("D")["risk_score"].apply(lambda s: (s >= 50).sum())
    ax2.plot(by_day.index, by_day.values, color=PRIMARY, marker="o", markersize=3)
    ax2.set_title("Daily High/Critical Alert Trend", fontsize=11, fontweight="bold", color=PRIMARY)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", rotation=45, labelsize=7)

    # Channel breakdown
    ax3 = fig.add_subplot(gs[1, 2])
    alerts = df[df.risk_score >= 50]
    chan_counts = alerts.channel.value_counts()
    ax3.pie(chan_counts.values, labels=chan_counts.index, autopct="%1.0f%%",
            colors=[PRIMARY, "#1C7293", ACCENT, "#BFD8DC"], textprops={"fontsize": 8})
    ax3.set_title("Alerts by Channel", fontsize=11, fontweight="bold", color=PRIMARY)

    # Alert table (top 8)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    top = df.sort_values("risk_score", ascending=False).head(8)[
        ["txn_id", "customer_id", "amount", "channel", "risk_score", "risk_band"]
    ]
    top["amount"] = top["amount"].map(lambda v: f"₹{v:,.0f}")
    table = ax4.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(PRIMARY)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7FAFA" if r % 2 == 0 else "white")
    ax4.set_title("Top Correlated-Risk Alerts", fontsize=11, fontweight="bold", color=PRIMARY, loc="left", pad=14)

    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"Saved snapshot to {OUT}")


if __name__ == "__main__":
    main()
