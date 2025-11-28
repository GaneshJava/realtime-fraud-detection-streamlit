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
    Channel: str

    # Currency information
    selected_currency: Optional[str] = "INR"
    Currency: Optional[str] = "INR"

    # Temporal features
    hour: Optional[int] = 0
    day_of_week: Optional[int] = 0
    month: Optional[int] = 0

    # Device info
    DeviceID: Optional[str] = ""
    device_last_seen: Optional[str] = ""
    device_fingerprint: Optional[str] = ""
    app_version: Optional[str] = ""

    # Behavioural telemetry
    monthly_avg: Optional[float] = 0.0
    rolling_avg_7d: Optional[float] = 0.0
    txns_last_1h: Optional[int] = 0
    txns_last_24h: Optional[int] = 0
    txns_last_7d: Optional[int] = 0
    beneficiaries_added_24h: Optional[int] = 0
    beneficiary_added_minutes: Optional[int] = 9999
    failed_login_attempts: Optional[int] = 0

    # Beneficiary flags (mutually exclusive in UI; both optional here)
    existing_beneficiary: Optional[bool] = False
    new_beneficiary: Optional[bool] = False

    # IP / Geo
    client_ip: Optional[str] = ""
    ip_country: Optional[str] = ""
    txn_location_ip: Optional[str] = ""
    txn_city: Optional[str] = ""
    txn_country: Optional[str] = ""
    home_city: Optional[str] = ""
    home_country: Optional[str] = ""
    declared_country: Optional[str] = ""
    suspicious_ip_flag: Optional[bool] = False
    last_known_lat: Optional[float] = None
    last_known_lon: Optional[float] = None
    txn_lat: Optional[float] = None
    txn_lon: Optional[float] = None
    atm_distance_km: Optional[float] = 0.0

    # Card / POS / online
    card_country: Optional[str] = ""
    cvv_provided: Optional[bool] = True
    card_small_attempts_in_5min: Optional[int] = 0
    pos_repeat_count: Optional[int] = 0
    shipping_address: Optional[str] = ""
    billing_address: Optional[str] = ""

    # Identity (onsite branch)
    id_type: Optional[str] = ""
    id_number: Optional[str] = ""

    # VPN / anonymization (Option A flags)
    vpn_detected: Optional[bool] = False
    vpn_provider: Optional[str] = ""
    tor_exit_node: Optional[bool] = False
    cloud_host_ip: Optional[bool] = False
    ip_risk_score: Optional[int] = 0


# ===========================
# FastAPI endpoint
# ===========================
@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:
    """
    Accepts a transaction payload and returns:
    - ML fraud probability (raw)
    - Isolation Forest anomaly score (raw)
    - Normalised 0–100 fraud and anomaly scores
    - ML risk label (based on 0–100 fraud score)
    - Triggered deterministic rules
    - Highest severity from rules
    - Final combined risk
    """
    # Convert Pydantic model to dict
    payload_dict = payload.dict()

    # Decide currency field
    currency = payload.selected_currency or payload.Currency or "INR"

    # --- ML scoring ---
    fraud_prob, anomaly_score, ml_label = score_transaction_ml(
        supervised_pipeline,
        iforest_pipeline,
        payload_dict,
        convert_to_inr=False,
        currency=currency,
    )

    fraud_score = normalize_score(fraud_prob, min_val=0.0, max_val=0.02)
    anomaly_score_norm = normalize_score(anomaly_score, min_val=0.0, max_val=0.10)

    # --- Rule evaluation ---
    rules_triggered, rules_highest = evaluate_rules(payload_dict, currency)

    # --- Final risk ---
    final_risk = combine_final_risk(ml_label, rules_highest)

    # Return structured result
    return {
        "ML": {
            "FraudProbabilityRaw": fraud_prob,
            "AnomalyScoreRaw": anomaly_score,
            "FraudRiskScore": round(fraud_score, 2),          # 0–100
            "AnomalyRiskScore": round(anomaly_score_norm, 2), # 0–100
            "MLRiskLabel": ml_label,
        },
        "Rules": {
            "TriggeredRules": rules_triggered,
            "HighestSeverity": rules_highest,
        },
        "FinalRisk": final_risk,
        "Meta": {
            "Currency": currency,
        },
    }
