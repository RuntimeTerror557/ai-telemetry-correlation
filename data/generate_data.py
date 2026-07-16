"""
Synthetic data generator for the AI-Driven Correlation of Cybersecurity
Telemetry & Transactional Behaviour prototype.

Produces two datasets that share a customer_id key:
  data/telemetry_events.csv   -- SOC/SIEM-style security telemetry
  data/transactions.csv       -- core-banking transaction ledger

A configurable fraction of transactions are seeded as fraudulent and are
deliberately preceded by suspicious telemetry (new device, new geography,
failed logins) within a short time window -- this is the signal the
correlation engine is built to detect. Genuine transactions occasionally
carry the same individual signals in isolation (e.g. a new device with no
suspicious transaction) so that naive single-signal rules over-fire and a
correlated view is actually necessary.
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

CHANNELS = ["mobile", "web", "atm", "branch"]
MERCHANT_CATEGORIES = ["retail", "utilities", "travel", "electronics", "grocery", "wire_transfer", "jewellery"]
COUNTRIES = ["IN", "IN", "IN", "IN", "AE", "SG", "GB", "NG", "US"]  # IN heavily weighted (home market)
EVENT_TYPES = ["login", "password_reset", "session_start", "otp_request", "device_registration"]


def gen_customers(n_customers):
    customers = []
    for i in range(n_customers):
        cid = f"CUST{i:05d}"
        home_device = f"DEV{fake.uuid4()[:8]}"
        home_country = "IN"
        customers.append({"customer_id": cid, "home_device": home_device, "home_country": home_country})
    return pd.DataFrame(customers)


def random_timestamp(start, end):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def gen_telemetry(customers, start, end, base_events_per_customer=6):
    rows = []
    for _, cust in customers.iterrows():
        n_events = np.random.poisson(base_events_per_customer)
        for _ in range(n_events):
            ts = random_timestamp(start, end)
            rows.append({
                "event_id": f"EVT{fake.uuid4()[:10]}",
                "customer_id": cust.customer_id,
                "timestamp": ts,
                "event_type": random.choices(EVENT_TYPES, weights=[70, 5, 15, 8, 2])[0],
                "device_id": cust.home_device,
                "ip_country": cust.home_country,
                "is_new_device": False,
                "is_new_geo": False,
                "failed_login_count": np.random.choice([0, 0, 0, 1], p=[0.85, 0.05, 0.05, 0.05]),
            })
    return pd.DataFrame(rows)


def gen_transactions(customers, start, end, base_txn_per_customer=8):
    rows = []
    for _, cust in customers.iterrows():
        n_txn = np.random.poisson(base_txn_per_customer)
        for _ in range(n_txn):
            ts = random_timestamp(start, end)
            amount = float(np.round(np.random.lognormal(mean=7.5, sigma=1.1), 2))
            rows.append({
                "txn_id": f"TXN{fake.uuid4()[:10]}",
                "customer_id": cust.customer_id,
                "timestamp": ts,
                "amount": amount,
                "channel": random.choices(CHANNELS, weights=[45, 35, 12, 8])[0],
                "merchant_category": random.choice(MERCHANT_CATEGORIES),
                "is_new_beneficiary": random.random() < 0.12,
                "label_fraud": 0,
            })
    return pd.DataFrame(rows)


def inject_fraud_scenarios(telemetry, transactions, customers, fraud_rate=0.035):
    """
    Pick a subset of transactions to become fraudulent account-takeover events.
    For each, inject correlated suspicious telemetry just before the transaction:
    a login from a NEW device, from a NEW (foreign) geography, with 2-4 failed
    login attempts, followed by a high-value transaction to a new beneficiary.
    """
    txn = transactions.copy()
    tel = telemetry.copy()
    n_fraud = int(len(txn) * fraud_rate)
    fraud_idx = np.random.choice(txn.index, size=n_fraud, replace=False)

    new_telemetry_rows = []
    for idx in fraud_idx:
        row = txn.loc[idx]
        cust_id = row.customer_id
        attack_ts = row.timestamp - timedelta(minutes=random.randint(2, 20))

        # suspicious login telemetry immediately preceding the fraud transaction
        new_telemetry_rows.append({
            "event_id": f"EVT{fake.uuid4()[:10]}",
            "customer_id": cust_id,
            "timestamp": attack_ts,
            "event_type": "login",
            "device_id": f"DEV{fake.uuid4()[:8]}",         # new / unrecognized device
            "ip_country": random.choice(["NG", "RU", "VN", "BR", "UNKNOWN"]),  # new / foreign geo
            "is_new_device": True,
            "is_new_geo": True,
            "failed_login_count": random.randint(2, 4),
        })

        # bump the transaction to look like a fraud pattern: high amount, new beneficiary
        txn.loc[idx, "amount"] = float(np.round(np.random.uniform(35000, 250000), 2))
        txn.loc[idx, "is_new_beneficiary"] = True
        txn.loc[idx, "merchant_category"] = "wire_transfer"
        txn.loc[idx, "channel"] = random.choice(["mobile", "web"])
        txn.loc[idx, "label_fraud"] = 1

    tel = pd.concat([tel, pd.DataFrame(new_telemetry_rows)], ignore_index=True)

    # add a smaller number of "noisy" isolated new-device logins on GENUINE
    # transactions, so single-signal rules alone would over-fire (motivates
    # the correlated approach over naive rule chains).
    n_noise = int(len(txn) * 0.06)
    genuine_idx = txn[txn.label_fraud == 0].sample(n=n_noise, random_state=1).index
    noise_rows = []
    for idx in genuine_idx:
        row = txn.loc[idx]
        noise_rows.append({
            "event_id": f"EVT{fake.uuid4()[:10]}",
            "customer_id": row.customer_id,
            "timestamp": row.timestamp - timedelta(minutes=random.randint(1, 25)),
            "event_type": "login",
            "device_id": f"DEV{fake.uuid4()[:8]}",
            "ip_country": "IN",       # new device but still home country, no failed logins
            "is_new_device": True,
            "is_new_geo": False,
            "failed_login_count": 0,
        })
    tel = pd.concat([tel, pd.DataFrame(noise_rows)], ignore_index=True)

    return tel.sort_values("timestamp").reset_index(drop=True), txn.sort_values("timestamp").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=600)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--fraud-rate", type=float, default=0.035)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    end = datetime(2026, 7, 16, 12, 0, 0)
    start = end - timedelta(days=args.days)

    customers = gen_customers(args.customers)
    telemetry = gen_telemetry(customers, start, end)
    transactions = gen_transactions(customers, start, end)
    telemetry, transactions = inject_fraud_scenarios(telemetry, transactions, customers, args.fraud_rate)

    telemetry.to_csv(f"{args.outdir}/telemetry_events.csv", index=False)
    transactions.to_csv(f"{args.outdir}/transactions.csv", index=False)

    print(f"Customers:        {len(customers)}")
    print(f"Telemetry events: {len(telemetry)}")
    print(f"Transactions:     {len(transactions)}  (fraud: {int(transactions.label_fraud.sum())})")
    print(f"Written to {args.outdir}/telemetry_events.csv and {args.outdir}/transactions.csv")


if __name__ == "__main__":
    main()
