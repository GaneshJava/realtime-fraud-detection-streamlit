# app_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fraud_logic import (
    load_artifacts,
    score_transaction_ml,
    evaluate_rules,
    combine_final_risk,
)

app = FastAPI(title="Fraud Detection API")

# ===========================
# Load ML artifacts once at startup
# ===========================
supervised_pipeline, iforest_pipeline = load_artifacts()

# ===========================
# Pydantic model for incoming payload
# ===========================
class TransactionPayload(BaseModel):
    Amount: float
    TransactionType: str
    Location: Optional[str] = "Unknown"
    Channel: str
    hour: Optional[int] = 0
    day_of_week: Optional[int] = 0
    month: Optional[int] = 0
    DeviceID: Optional[str] = ""
    device_last_seen: Optional[str] = ""
    monthly_avg: Optional[float] = 0.0
    rolling_avg_7d: Optional[float] = 0.0
    txns_last_1h: Optional[int] = 0
    txns_last_24h: Optional[int] = 0
    txns_last_7d: Optional[int] = 0
    beneficiaries_added_24h: Optional[int] = 0
    beneficiary_added_minutes: Optional[int] = 9999
    failed_login_attempts: Optional[int] = 0
    ip_country: Optional[str] = ""
    declared_country: Optional[str] = ""
    suspicious_ip_flag: Optional[bool] = False
    last_known_lat: Optional[float] = None
    last_known_lon: Optional[float] = None
    txn_lat: Optional[float] = None
    txn_lon: Optional[float] = None
    atm_distance_km: Optional[float] = 0.0
    card_country: Optional[str] = ""
    cvv_provided: Optional[bool] = True
    card_small_attempts_in_5min: Optional[int] = 0
    pos_repeat_count: Optional[int] = 0
    selected_currency: Optional[str] = "INR"


# ===========================
# FastAPI endpoint
# ===========================
@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:
    """
    Accepts a transaction payload and returns:
    - ML fraud probability
    - Isolation Forest anomaly score
    - ML risk label
    - Triggered deterministic rules
    - Highest severity from rules
    - Final combined risk
    """
    # Convert Pydantic model to dict
    payload_dict = payload.dict()

    # --- ML scoring ---
    ml_prob, anomaly_score, ml_label = score_transaction_ml(
        supervised_pipeline, iforest_pipeline, payload_dict
    )

    # --- Rule evaluation ---
    rules_triggered, rules_highest = evaluate_rules(payload_dict, payload.selected_currency)

    # --- Final risk ---
    final_risk = combine_final_risk(ml_label, rules_highest)

    # Return structured result
    return {
        "ML": {
            "FraudProbability": round(ml_prob * 100, 2),  # 0-100%
            "AnomalyScore": round(anomaly_score * 100, 2), # scaled for clarity
            "MLRiskLabel": ml_label
        },
        "Rules": {
            "TriggeredRules": rules_triggered,
            "HighestSeverity": rules_highest
        },
        "FinalRisk": final_risk
    }
