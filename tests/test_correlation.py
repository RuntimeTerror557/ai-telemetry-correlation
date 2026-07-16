import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.correlation_engine import RiskScorer
from src.features import build_features


def _toy_data():
    telemetry = pd.DataFrame([
        {
            "event_id": "E1", "customer_id": "C1", "timestamp": "2026-07-01 10:00:00",
            "event_type": "login", "device_id": "NEWDEV", "ip_country": "NG",
            "is_new_device": True, "is_new_geo": True, "failed_login_count": 3,
        },
        {
            "event_id": "E2", "customer_id": "C2", "timestamp": "2026-07-01 10:00:00",
            "event_type": "login", "device_id": "HOMEDEV", "ip_country": "IN",
            "is_new_device": False, "is_new_geo": False, "failed_login_count": 0,
        },
    ])
    transactions = pd.DataFrame([
        {
            "txn_id": "T1", "customer_id": "C1", "timestamp": "2026-07-01 10:10:00",
            "amount": 150000.0, "channel": "mobile", "merchant_category": "wire_transfer",
            "is_new_beneficiary": True, "label_fraud": 1,
        },
        {
            "txn_id": "T2", "customer_id": "C2", "timestamp": "2026-07-01 10:10:00",
            "amount": 500.0, "channel": "atm", "merchant_category": "retail",
            "is_new_beneficiary": False, "label_fraud": 0,
        },
    ])
    return transactions, telemetry


def test_correlated_event_flagged():
    txn, tel = _toy_data()
    feats = build_features(txn, tel)
    fraud_row = feats[feats.txn_id == "T1"].iloc[0]
    genuine_row = feats[feats.txn_id == "T2"].iloc[0]
    assert fraud_row.correlated_risk_event == 1
    assert genuine_row.correlated_risk_event == 0


def test_rule_based_scoring_ranks_fraud_higher():
    txn, tel = _toy_data()
    feats = build_features(txn, tel)
    scorer = RiskScorer(model_path="models/nonexistent.joblib")  # forces rule-only fallback
    scored = scorer.score_batch(feats)
    fraud_score = scored[scored.txn_id == "T1"].iloc[0].risk_score
    genuine_score = scored[scored.txn_id == "T2"].iloc[0].risk_score
    assert fraud_score > genuine_score
    assert fraud_score >= 75  # should land in Critical band


if __name__ == "__main__":
    test_correlated_event_flagged()
    test_rule_based_scoring_ranks_fraud_higher()
    print("All tests passed.")
