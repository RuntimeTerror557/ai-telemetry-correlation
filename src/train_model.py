"""
Trains a gradient-boosted classifier on the correlated telemetry+transaction
features to predict fraud probability. Swappable for XGBoost in production
(see README) -- scikit-learn's HistGradientBoostingClassifier is used here
for zero-friction local setup.
"""
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                              roc_auc_score)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default="data/transactions.csv")
    ap.add_argument("--telemetry", default="data/telemetry_events.csv")
    ap.add_argument("--out", default="models/risk_model.joblib")
    args = ap.parse_args()

    txn = pd.read_csv(args.transactions)
    tel = pd.read_csv(args.telemetry)
    feats = build_features(txn, tel)
    feats.to_csv("data/engineered_features.csv", index=False)

    X = feats[FEATURE_COLUMNS]
    y = feats["label_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.08,
        class_weight="balanced", random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("=== Validation performance ===")
    print(f"ROC-AUC:        {roc_auc_score(y_test, probs):.4f}")
    print(f"Avg Precision:  {average_precision_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds, digits=3))

    importances = pd.Series(
        getattr(model, "feature_importances_", [None] * len(FEATURE_COLUMNS)),
        index=FEATURE_COLUMNS,
    )
    if importances.notna().any():
        print("=== Feature importances ===")
        print(importances.sort_values(ascending=False))

    joblib.dump(model, args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()
