"""
fraud_logic.py

Optimized fraud logic for both Streamlit and FastAPI.
Includes:
- Currency conversion
- ML scoring
- Deterministic rules
- Final risk aggregation
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import pandas as pd

# ===========================
# 0) Currency configuration (INR base)
# ===========================
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "SAR": 22.2,
}

CURRENCY_OPTIONS = list(INR_PER_UNIT.keys())

# ===========================
# 1) Helper functions
# ===========================
def haversine_km(lat1: Optional[float], lon1: Optional[float],
                 lat2: Optional[float], lon2: Optional[float]) -> Optional[float]:
    """Return distance in km between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

def escalate(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b

def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    return amount_in_inr / INR_PER_UNIT.get(currency, 1.0)

# ===========================
# 2) Load ML artifacts
# ===========================
def load_artifacts(models_dir: str = "models") -> Tuple:
    models_dir = Path(models_dir)
    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {name}") from e
    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")
    return supervised_pipeline, iforest_pipeline

# ===========================
# 3) ML thresholds
# ===========================
FRAUD_MED, FRAUD_HIGH, FRAUD_CRIT = 0.00005, 0.00023328, 0.01732857
ANOM_MED, ANOM_HIGH, ANOM_CRIT = 0.04, 0.05, 0.08

def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    return "LOW"

# ===========================
# 4) Base thresholds (INR)
# ===========================
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

# ===========================
# 5) Rule engine
# ===========================
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    # Convert thresholds
    thresholds = {k: inr_to_currency(v, currency) for k, v in BASE_THRESHOLDS_INR.items()}

    rules: List[Dict] = []

    # Extract payload fields with defaults
    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0))
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))
    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)
    card_country = str(payload.get("card_country", "")).lower()
    cvv_provided = payload.get("cvv_provided", True)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)
    ip_country = str(payload.get("ip_country", "")).lower()
    declared_country = str(payload.get("declared_country", "")).lower()
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    last_device = str(payload.get("device_last_seen", "")).lower()
    curr_device = str(payload.get("DeviceID", "")).lower()
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0))

    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # ------------------ CRITICAL ------------------
    if amt >= thresholds["absolute_crit_amount"]:
        add_rule("Absolute very large amount", "CRITICAL",
                 f"Transaction {amt:.2f} {currency} >= critical {thresholds['absolute_crit_amount']:.2f}")

    impossible_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)
    device_new = not last_device
    location_changed = impossible_distance is not None and impossible_distance > 500
    if device_new and location_changed and amt > thresholds["medium_amount_threshold"]:
        add_rule("New device + Impossible travel + High amount", "CRITICAL",
                 f"Device unseen and travel {impossible_distance:.1f} km; amount {amt:.2f} {currency}")

    if channel == "atm" and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH",
                 f"ATM is {atm_distance_km:.1f} km from last known location")

    if beneficiaries_added_24h >= 3 and amt > thresholds["high_amount_threshold"]:
        add_rule("Multiple beneficiaries added recently + high transfer", "CRITICAL",
                 f"{beneficiaries_added_24h} beneficiaries added and transfer {amt:.2f} {currency}")

    # ------------------ HIGH ------------------
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} transactions in last 1 hour")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} transactions in last 24 hours")
    if ip_country and declared_country and ip_country != declared_country and channel not in ("bank", "atm"):
        sev = "HIGH" if amt > thresholds["high_amount_threshold"] else "MEDIUM"
        add_rule("IP / Declared country mismatch", sev, f"IP {ip_country} != declared {declared_country}")
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed login attempts")
    if new_benef and amt >= thresholds["medium_amount_threshold"]:
        add_rule("New beneficiary + significant amount", "HIGH", "Transfer to new beneficiary above threshold")
    if suspicious_ip_flag and amt > thresholds["medium_amount_threshold"]/4:
        add_rule("IP flagged by threat intelligence", "HIGH", "IP flagged as suspicious")

    # ------------------ MEDIUM ------------------
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > thresholds["medium_amount_threshold"]:
        add_rule("Large spike vs monthly avg", "HIGH", f"Amount {amt:.2f} >=5x monthly avg {monthly_avg:.2f}")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > thresholds["medium_amount_threshold"]/2:
        add_rule("Spike vs 7-day avg", "MEDIUM", f"Amount {amt:.2f} >=3x 7-day avg {rolling_avg_7d:.2f}")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > thresholds["medium_amount_threshold"]/2:
        add_rule("Above monthly usual", "MEDIUM", f"Amount {amt:.2f} >=2x monthly avg {monthly_avg:.2f}")

    if 0 <= hour <= 5 and monthly_avg < thresholds["medium_amount_threshold"]*2 and amt > thresholds["medium_amount_threshold"]/10:
        add_rule("Late-night transaction for low-activity customer", "MEDIUM",
                 f"Transaction at hour {hour}; amount {amt:.2f}")

    if last_device and curr_device and last_device != curr_device and channel not in ("bank", "atm"):
        add_rule("Device mismatch", "MEDIUM", f"Device changed from {last_device} to {curr_device}")

    # ------------------ CHANNEL SPECIFIC ------------------
    if channel in ("credit card", "online purchase") and card_small_attempts >= 6:
        add_rule("Card micro-test detected", "HIGH", f"{card_small_attempts} small charges detected")
    if channel == "atm" and amt >= thresholds["atm_high_withdrawal"]:
        add_rule("Large ATM withdrawal", "HIGH", f"ATM withdrawal {amt:.2f} >= {thresholds['atm_high_withdrawal']:.2f}")
    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS")
    if channel in ("bank", "netbanking") and beneficiary_added_minutes < 10 and amt >= thresholds["medium_amount_threshold"]:
        add_rule("Immediate transfer to new beneficiary", "HIGH",
                 f"Beneficiary added {beneficiary_added_minutes} mins ago; transfer {amt:.2f}")

    # ------------------ FINAL ------------------
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest

# ===========================
# 6) Combine ML + Rules
# ===========================
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)

# ===========================
# 7) ML scoring
# ===========================
def score_transaction_ml(supervised_pipeline, iforest_pipeline, model_payload: Dict) -> Tuple[float, float, str]:
    df = pd.DataFrame([{
        "Amount": model_payload.get("Amount", 0.0),
        "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
        "Location": model_payload.get("Location", "Unknown"),
        "DeviceID": model_payload.get("DeviceID", "Unknown"),
        "Channel": model_payload.get("Channel", "Other"),
        "hour": model_payload.get("hour", 0),
        "day_of_week": model_payload.get("day_of_week", 0),
        "month": model_payload.get("month", 0),
    }])
    try:
        fraud_prob = float(supervised_pipeline.predict_proba(df)[0, 1])
    except Exception:
        fraud_prob = 0.0
    try:
        anomaly_score = -float(iforest_pipeline.decision_function(df)[0])
    except Exception:
        anomaly_score = 0.0
    ml_label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, ml_label
