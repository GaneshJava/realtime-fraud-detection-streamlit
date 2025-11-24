# app_api.py 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from fraud_logic import (
    load_artifacts,
    score_transaction_ml,
    evaluate_rules,
    combine_final_risk,
    normalize_score,
)

app = FastAPI(title="Fraud Detection API")

# ===========================================================
# Load ML artifacts once at startup
# ===========================================================
supervised_pipeline, iforest_pipeline = load_artifacts()


# ===========================================================
# Pydantic model for incoming payload — aligned with Streamlit
# ===========================================================
class TransactionPayload(BaseModel):
    Amount: float
    Currency: str = "INR"
    TransactionType: str
    Channel: str

    hour: Optional[int] = 0
    day_of_week: Optional[int] = 0
    month: Optional[int] = 0

    # Device info
    DeviceID: Optional[str] = ""
    device_last_seen: Optional[str] = ""

    # Telemetry
    monthly_avg: Optional[float] = 0.0
    rolling_avg_7d: Optional[float] = 0.0
    txns_last_1h: Optional[int] = 0
    txns_last_24h: Optional[int] = 0
    txns_last_7d: Optional[int] = 0
    beneficiaries_added_24h: Optional[int] = 0
    failed_login_attempts: Optional[int] = 0
    beneficiary_added_minutes: Optional[int] = 9999

    # IP / Geo — ip_country auto-derived in API
    client_ip: Optional[str] = ""
    txn_location_ip: Optional[str] = ""
    txn_city: Optional[str] = ""
    txn_country: Optional[str] = ""
    home_city: Optional[str] = ""
    home_country: Optional[str] = ""
    suspicious_ip_flag: Optional[bool] = False

    last_known_lat: Optional[float] = None
    last_known_lon: Optional[float] = None
    txn_lat: Optional[float] = None
    txn_lon: Optional[float] = None

    # Card / ATM
    card_country: Optional[str] = ""
    cvv_provided: Optional[bool] = True
    card_masked: Optional[str] = ""
    card_small_attempts_in_5min: Optional[int] = 0
    atm_distance_km: Optional[float] = 0.0
    pos_repeat_count: Optional[int] = 0

    # Merchant fields
    merchant_id: Optional[str] = ""
    payment_category: Optional[str] = ""
    shipping_address: Optional[str] = ""
    billing_address: Optional[str] = ""

    # NetBanking
    beneficiary: Optional[str] = ""
    new_beneficiary: Optional[bool] = False


# ===========================================================
# API Endpoint
# ===========================================================
@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:

    payload_dict = payload.dict()

    # -------------------------------
    # ip_country derived from txn_country
    # -------------------------------
    txn_country = payload_dict.get("txn_country", "")
    payload_dict["ip_country"] = txn_country.strip().lower()

    # -------------------------------
    # ML scoring (raw)
    # -------------------------------
    fraud_prob_raw, anomaly_raw, ml_label = score_transaction_ml(
        supervised_pipeline,
        iforest_pipeline,
        payload_dict,
    )

    # -------------------------------
    # Normalized 0–100 scores
    # -------------------------------
    fraud_score = normalize_score(fraud_prob_raw, min_val=0.0, max_val=0.02)
    anomaly_score = normalize_score(anomaly_raw, min_val=0.0, max_val=0.10)

    # -------------------------------
    # Deterministic Rules
    # -------------------------------
    rules_triggered, rules_highest = evaluate_rules(payload_dict, payload.Currency)

    # -------------------------------
    # Final Risk
    # -------------------------------
    final_risk = combine_final_risk(ml_label, rules_highest)

    # -------------------------------
    # Structured Response
    # -------------------------------
    return {
        "ML": {
            "FraudProbabilityRaw": fraud_prob_raw,
            "AnomalyScoreRaw": anomaly_raw,
            "FraudRiskScore": fraud_score,
            "AnomalyRiskScore": anomaly_score,
            "MLRiskLabel": ml_label,
        },
        "Rules": {
            "TriggeredRules": rules_triggered,
            "HighestSeverity": rules_highest,
        },
        "FinalRisk": final_risk,
    }
