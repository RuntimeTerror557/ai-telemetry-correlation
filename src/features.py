"""
Core correlation logic: for every transaction, look back at that customer's
cybersecurity telemetry within a configurable window and derive features
that combine the two signal types. This is the "correlation" referenced
throughout the presentation -- a lone risky login or a lone large
transaction is much weaker signal than the two occurring together.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

LOOKBACK_MINUTES = 30
VELOCITY_WINDOW_MINUTES = 60


def build_features(transactions: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    txn = transactions.copy()
    tel = telemetry.copy()
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])

    tel_by_cust = {cid: g.sort_values("timestamp") for cid, g in tel.groupby("customer_id")}
    txn_by_cust = {cid: g.sort_values("timestamp") for cid, g in txn.groupby("customer_id")}

    feature_rows = []
    for cid, cust_txns in txn_by_cust.items():
        cust_tel = tel_by_cust.get(cid, pd.DataFrame(columns=tel.columns))
        cust_txn_ts = cust_txns["timestamp"].values

        for _, row in cust_txns.iterrows():
            window_start = row.timestamp - timedelta(minutes=LOOKBACK_MINUTES)
            window_events = cust_tel[(cust_tel.timestamp >= window_start) & (cust_tel.timestamp <= row.timestamp)]

            velocity_start = row.timestamp - timedelta(minutes=VELOCITY_WINDOW_MINUTES)
            txn_velocity = int(((cust_txns.timestamp >= velocity_start) & (cust_txns.timestamp <= row.timestamp)).sum())

            has_new_device = bool((window_events.is_new_device == True).any())  # noqa: E712
            has_new_geo = bool((window_events.is_new_geo == True).any())  # noqa: E712
            failed_logins = int(window_events.failed_login_count.sum()) if len(window_events) else 0
            n_telemetry_events = int(len(window_events))
            minutes_since_last_login = np.nan
            login_events = window_events[window_events.event_type == "login"]
            if len(login_events):
                minutes_since_last_login = (row.timestamp - login_events.timestamp.max()).total_seconds() / 60.0

            feature_rows.append({
                "txn_id": row.txn_id,
                "customer_id": cid,
                "timestamp": row.timestamp,
                "amount": row.amount,
                "channel": row.channel,
                "merchant_category": row.merchant_category,
                "is_new_beneficiary": int(row.is_new_beneficiary),
                "txn_velocity_1h": txn_velocity,
                "telemetry_events_in_window": n_telemetry_events,
                "has_new_device_login": int(has_new_device),
                "has_new_geo_login": int(has_new_geo),
                "failed_logins_in_window": failed_logins,
                "minutes_since_last_login": minutes_since_last_login if not np.isnan(minutes_since_last_login) else 999.0,
                "correlated_risk_event": int(has_new_device and has_new_geo),
                "label_fraud": int(row.label_fraud) if "label_fraud" in row else np.nan,
            })

    feats = pd.DataFrame(feature_rows)
    feats["log_amount"] = np.log1p(feats["amount"])
    return feats


FEATURE_COLUMNS = [
    "log_amount",
    "is_new_beneficiary",
    "txn_velocity_1h",
    "telemetry_events_in_window",
    "has_new_device_login",
    "has_new_geo_login",
    "failed_logins_in_window",
    "minutes_since_last_login",
    "correlated_risk_event",
]
