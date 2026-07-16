"""
RiskScorer: blends a transparent, rule-based score with a trained ML
model's probability, and produces a human-readable explanation for every
score. The blend keeps the system audit-friendly (banks' compliance teams
can see exactly why a score fired) while still benefiting from a model
that can pick up on subtler combinations than hand-written rules.
"""
from dataclasses import dataclass, field
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS

RULE_WEIGHTS = {
    "has_new_device_login": 25,
    "has_new_geo_login": 30,
    "correlated_risk_event": 20,   # extra weight when device+geo co-occur
    "failed_logins_in_window": 6,  # per failed attempt, capped
    "is_new_beneficiary": 8,
    "high_velocity": 10,           # txn_velocity_1h >= 3
    "large_amount": 10,            # log_amount in top decile-ish
}

LARGE_AMOUNT_THRESHOLD = 10.0   # log1p(amount) ~ amount > ~22,000
ML_BLEND_WEIGHT = 0.55           # weight given to the ML probability vs rule score


@dataclass
class ScoredAlert:
    txn_id: str
    customer_id: str
    timestamp: str
    amount: float
    channel: str
    rule_score: float
    ml_score: float
    risk_score: float
    risk_band: str
    reasons: List[str] = field(default_factory=list)


def _risk_band(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


class RiskScorer:
    def __init__(self, model_path: str = "models/risk_model.joblib"):
        self.model = None
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            self.model = None  # falls back to rule-only scoring

    def _rule_score(self, row: pd.Series):
        score = 0.0
        reasons = []
        if row.has_new_device_login:
            score += RULE_WEIGHTS["has_new_device_login"]
            reasons.append("Login from an unrecognized device shortly before the transaction")
        if row.has_new_geo_login:
            score += RULE_WEIGHTS["has_new_geo_login"]
            reasons.append("Login originated from an unfamiliar / foreign geography")
        if row.correlated_risk_event:
            score += RULE_WEIGHTS["correlated_risk_event"]
            reasons.append("New device AND new geography correlated in the same session")
        if row.failed_logins_in_window:
            add = min(row.failed_logins_in_window * RULE_WEIGHTS["failed_logins_in_window"], 24)
            score += add
            reasons.append(f"{int(row.failed_logins_in_window)} failed login attempt(s) just before the transaction")
        if row.is_new_beneficiary:
            score += RULE_WEIGHTS["is_new_beneficiary"]
            reasons.append("Funds sent to a new/unrecognized beneficiary")
        if row.txn_velocity_1h >= 3:
            score += RULE_WEIGHTS["high_velocity"]
            reasons.append(f"{int(row.txn_velocity_1h)} transactions by this customer in the last hour")
        if row.log_amount >= LARGE_AMOUNT_THRESHOLD:
            score += RULE_WEIGHTS["large_amount"]
            reasons.append("Transaction amount is unusually large for this profile")
        return min(score, 100.0), reasons

    def score_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        results = []
        if self.model is not None:
            ml_probs = self.model.predict_proba(features[FEATURE_COLUMNS])[:, 1] * 100
        else:
            ml_probs = np.zeros(len(features))

        for i, (_, row) in enumerate(features.iterrows()):
            rule_score, reasons = self._rule_score(row)
            ml_score = float(ml_probs[i])
            if self.model is not None:
                blended = ML_BLEND_WEIGHT * ml_score + (1 - ML_BLEND_WEIGHT) * rule_score
            else:
                blended = rule_score
            blended = float(min(blended, 100.0))

            if not reasons:
                reasons = ["No individually suspicious telemetry signals correlated with this transaction"]

            results.append(ScoredAlert(
                txn_id=row.txn_id,
                customer_id=row.customer_id,
                timestamp=str(row.timestamp),
                amount=float(row.amount),
                channel=row.channel,
                rule_score=round(rule_score, 1),
                ml_score=round(ml_score, 1),
                risk_score=round(blended, 1),
                risk_band=_risk_band(blended),
                reasons=reasons,
            ))

        out = pd.DataFrame([r.__dict__ for r in results])
        return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
