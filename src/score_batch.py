"""
Runs the full pipeline end to end: load raw telemetry + transactions,
correlate into features, score every transaction, and write out an
analyst-ready alerts feed sorted by risk.

Usage:
    python -m src.score_batch
    python -m src.score_batch --threshold 50 --out data/alerts.csv
"""
import argparse

import pandas as pd

from src.correlation_engine import RiskScorer
from src.features import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default="data/transactions.csv")
    ap.add_argument("--telemetry", default="data/telemetry_events.csv")
    ap.add_argument("--model", default="models/risk_model.joblib")
    ap.add_argument("--threshold", type=float, default=25.0, help="min risk_score to keep as an alert")
    ap.add_argument("--out", default="data/alerts.csv")
    args = ap.parse_args()

    txn = pd.read_csv(args.transactions)
    tel = pd.read_csv(args.telemetry)

    print("Correlating telemetry with transactions...")
    feats = build_features(txn, tel)

    print("Scoring with RiskScorer (rules + ML blend)...")
    scorer = RiskScorer(model_path=args.model)
    scored = scorer.score_batch(feats)

    alerts = scored[scored.risk_score >= args.threshold].copy()
    alerts.to_csv(args.out, index=False)
    scored.to_csv("data/all_scored_transactions.csv", index=False)

    print(f"\nScored {len(scored)} transactions.")
    print(f"Alerts (risk_score >= {args.threshold}): {len(alerts)}")
    print(alerts.risk_band.value_counts())
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
