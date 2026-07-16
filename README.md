# AI-Driven Correlation of Cybersecurity Telemetry & Transactional Behaviour

**Team Sentinel — Finspark Hackathon 2026**

A working prototype that fuses SOC/SIEM-style cybersecurity telemetry (logins,
devices, geo-location, failed auth attempts) with core-banking transaction
data in real time, producing a single, explainable risk score per
transaction instead of two teams staring at two disconnected dashboards.

## Why this exists

Cyber-attacks and fraudulent transactions are usually monitored in silos:
the SOC watches logins and sessions, the Fraud team watches money movement.
A compromised login rarely triggers a transaction risk re-check, and a
large transaction rarely pulls up the session that authorized it. This
prototype closes that gap by correlating the two signal types in a shared
time window and scoring the *combination*, not just either signal alone.

## How it works

```
Cybersecurity Telemetry ─┐
                          ├─► Kafka-style ingestion (simulated here as CSV) ─► Correlation Engine
Transaction Data Feed  ──┘                                                    (rules + ML) ─► Risk Score ─► Analyst Dashboard
```

1. **`data/generate_data.py`** — synthesizes realistic telemetry and
   transaction datasets for ~600 customers over 30 days, and seeds a
   fraction of transactions as account-takeover fraud with correlated
   suspicious telemetry (new device + new geography + failed logins) just
   before the transaction — the exact pattern this system is built to catch.
2. **`src/features.py`** — for every transaction, looks back over a
   30-minute window of that customer's telemetry and engineers correlated
   features: new-device login, new-geography login, failed-login count,
   transaction velocity, new beneficiary, and a `correlated_risk_event`
   flag when device + geography anomalies co-occur.
3. **`src/train_model.py`** — trains a gradient-boosted classifier
   (`HistGradientBoostingClassifier`; swap in XGBoost by changing one import
   for production) on the engineered features to predict fraud probability.
4. **`src/correlation_engine.py`** — `RiskScorer` blends a transparent,
   weighted rule score (for compliance-ready explainability) with the ML
   model's probability, and returns a plain-language list of reasons for
   every score — this is what "explainable AI" means in the pitch deck.
5. **`src/score_batch.py`** — runs the full pipeline and writes an
   analyst-ready alert feed.
6. **`dashboard/app.py`** — a Streamlit console for SOC/Fraud analysts:
   KPIs, alert volume by risk band, daily trend, and a case drill-down
   that shows exactly why a transaction was flagged.
7. **`reports/generate_report.py`** — renders a static PNG snapshot of the
   console, useful for demos/screenshots when a live session isn't handy.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# 1. Generate synthetic telemetry + transaction data
python data/generate_data.py --customers 600 --days 30 --fraud-rate 0.035

# 2. Train the correlation model
python -m src.train_model

# 3. Score all transactions and produce the alert feed
python -m src.score_batch

# 4. Launch the analyst dashboard
streamlit run dashboard/app.py

# (optional) static snapshot for a demo screenshot
python reports/generate_report.py

# run tests
pytest tests/ -v
```

## Repository layout

```
ai-telemetry-correlation/
├── data/
│   ├── generate_data.py          # synthetic telemetry + transaction generator
│   ├── telemetry_events.csv      # generated
│   ├── transactions.csv          # generated
│   ├── engineered_features.csv   # generated (from training)
│   ├── all_scored_transactions.csv  # generated (from scoring)
│   └── alerts.csv                # generated (from scoring)
├── src/
│   ├── features.py               # telemetry <-> transaction correlation logic
│   ├── correlation_engine.py     # RiskScorer: rules + ML blend, explainability
│   ├── train_model.py            # model training + validation metrics
│   └── score_batch.py            # end-to-end batch scoring CLI
├── dashboard/
│   └── app.py                    # Streamlit SOC/Fraud analyst console
├── reports/
│   └── generate_report.py        # static PNG dashboard snapshot
├── tests/
│   └── test_correlation.py       # unit tests for correlation + scoring
├── models/
│   └── risk_model.joblib         # generated (from training)
└── requirements.txt
```

## Model performance (on synthetic validation data)

The correlated features separate fraud from genuine transactions cleanly —
ROC-AUC ≈ 0.9996, average precision ≈ 0.977 on held-out data — because the
fraud scenarios are specifically defined by the telemetry+transaction
correlation this system looks for. On real bank data, expect softer
separation; the rule-based component is there precisely so the system
degrades gracefully and stays explainable even if the ML signal is noisier.

## Design choices & production notes

- **Explainability first**: every score ships with plain-language reasons
  (`RiskScorer._rule_score`), so a compliance/audit team can see exactly
  why a case was raised — not just a black-box number.
- **Rules + ML blend**: rules alone over-fire on single weak signals (a
  genuine customer with a new phone); ML alone can be an opaque black box.
  Blending keeps recall high while keeping the score auditable.
- **Swap-in points for production**:
  - Replace the CSV data loaders with a Kafka/Kinesis consumer for true
    real-time streaming.
  - Replace `HistGradientBoostingClassifier` with XGBoost or a graph neural
    network (PyTorch Geometric) to also capture multi-account fraud rings.
  - Replace `models/risk_model.joblib` loading with a model registry
    (MLflow) and add scheduled retraining.
  - Add RBAC, audit logging, and PII pseudonymization before telemetry
    reaches the correlation engine (see Security Considerations in the
    accompanying presentation).

## License

Built for the Finspark Hackathon 2026. Sample/demo code — not production
security software.
