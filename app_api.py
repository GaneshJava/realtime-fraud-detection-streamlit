# app_api.py
# ---------------------------------
# FastAPI service using fraud_logic
# ---------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from fraud_logic import (
    score_transaction_ml,
    evaluate_rules,
    combine_final_risk,
    build_explanation,
    normalize_score
)

app = FastAPI(title="Fraud Detection API", version="1.0")

# -------------------------
# PAYLOAD SCHEMA
# -------------------------

class TransactionPayload(BaseModel):
    Amount: float
    Currency: str = "INR"
    TransactionType: str
    Channel: str
    Location: str | None = None
    DeviceID: str | None = None
    hour: int
    day_of_week: int
    month: int

    txn_city: str | None = None
    txn_country: str | None = None
    declared_country: str | None = None

    # optional rule engine fields
    monthly_avg: float | None = 0
    rolling_avg_7d: float | None = 0
    beneficiaries_added_24h: int | None = 0
    failed_login_attempts: int | None = 0
    suspicious_ip_flag: bool | None = False

# -------------------------
# API ENDPOINT
# -------------------------

@app.post("/score")
def score_transaction(payload: TransactionPayload):
    payload_dict = payload.dict()

    # --- ML ---
    fraud_prob, anomaly_raw, ml_label = score_transaction_ml(payload_dict)

    fraud_score = normalize_score(fraud_prob, 0.0, 0.02)
    anomaly_score = normalize_score(anomaly_raw, 0.0, 0.10)

    # --- RULES ---
    rules, rules_highest = evaluate_rules(payload_dict, payload.Currency)

    final_risk = combine_final_risk(ml_label, rules_highest)

    explanation = build_explanation(
        payload_dict,
        fraud_score,
        anomaly_score,
        ml_label,
        rules,
        final_risk
    )

    return {
        "fraud_probability": fraud_prob,
        "fraud_score": fraud_score,
        "anomaly_score": anomaly_score,
        "ml_label": ml_label,
        "rules_triggered": rules,
        "rules_highest": rules_highest,
        "final_risk": final_risk,
        "explanation": explanation,
        "payload_used": payload_dict
    }
