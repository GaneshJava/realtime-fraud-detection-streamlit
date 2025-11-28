Conversation opened. 13 messages. All messages read.

Skip to content
Using Gmail with screen readers
app_api-py 
2 of 4
Source code
Inbox

Anas Awan
Attachments
Thu, Nov 20, 7:13 PM (8 days ago)
 
7

Anas Awan
Sun, Nov 23, 2:46 AM (5 days ago)
I acknowledge. I will share the updated codes on Monday, will connect with you too.

Anas Awan
Attachments
Mon, Nov 24, 12:38 AM (4 days ago)
to me

Here are the updated files with all the concerns addressed.
 5 Attachments
  •  Scanned by Gmail

B N S Ganesh Prasad
Mon, Nov 24, 1:10 PM (4 days ago)
Hi Anas, Thanks for your support.. Please let me know when we can connect today on next steps. I propose 4 PM - 6 PM, anytime in between. Also, do let me know y

Anas Awan
Mon, Nov 24, 2:48 PM (4 days ago)
Yes, I will be available at that time. Just send me the mail and I will be there

B N S Ganesh Prasad <ganesh.java7@gmail.com>
Mon, Nov 24, 4:14 PM (4 days ago)
to Anas

sent a meeting invite for 4:30 PM, 
Here is the link for your reference: https://calendar.app.google/668rYLzPm5yBCnh79 

Thanks & Regards
Ganesh
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

app = FastAPI(title="Fraud Detection API", version="1.0")

# ===========================
# Load ML artifacts once at startup
# ===========================
supervised_pipeline, iforest_pipeline = load_artifacts()


# ===========================
# Pydantic model for transaction payload
# ===========================
class TransactionPayload(BaseModel):
    Amount: float
    TransactionType: str
    Channel: str
    Location: Optional[str] = "Unknown"
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
# Root route for health check
# ===========================
@app.get("/", summary="API Health Check")
def root() -> Dict[str, str]:
    return {"message": "Fraud Detection API is running!"}


# ===========================
# Prediction endpoint
# ===========================
@app.post("/predict", summary="Predict Fraud Risk")
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
    data = payload.dict()

    # --- ML scoring ---
    ml_prob, anomaly_score, ml_label = score_transaction_ml(
        supervised_pipeline, iforest_pipeline, data
    )

    # --- Rule evaluation ---
    rules_triggered, rules_highest = evaluate_rules(data, data.get("selected_currency", "INR"))

    # --- Final combined risk ---
    final_risk = combine_final_risk(ml_label, rules_highest)

    # Return structured response
    return {
        "ML": {
            "FraudProbability": round(ml_prob * 100, 2),
            "AnomalyScore": round(anomaly_score * 100, 2),
            "MLRiskLabel": ml_label,
        },
        "Rules": {
            "TriggeredRules": rules_triggered,
            "HighestSeverity": rules_highest,
        },
        "FinalRisk": final_risk,
    }
app_api.py
Displaying app_api.py.
