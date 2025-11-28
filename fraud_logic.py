"""
fraud_logic.py

Fully self-contained fraud logic for both Streamlit and FastAPI.
Includes:
- Currency conversion
- ML scoring
- Deterministic rules
- VPN / IP reputation rules (Option A: manual flags)
- Final risk aggregation
"""

from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

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

CURRENCY_OPTIONS = ["INR", "USD", "GBP", "EUR", "AED", "SAR"]

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

def escalate(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b

def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    return amount_in_inr / INR_PER_UNIT.get(currency, 1.0)

# ===========================
# Helpers
# ===========================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Return distance in km between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371

def normalize_score(x: float, min_val: float = 0.0, max_val: float = 0.02) -> float:
    """
    Normalize an ML score into a 0–100 range for business interpretability.

    For fraud probability:
      - We assume most interesting values are in [0, 0.02] (0%–2%),
      - 0 => 0, 0.02 => 100.
    For anomaly score you can pass a larger max_val, e.g. 0.10.
    """
    if x is None:
        return 0.0
    try:
        val = float(x)
    except Exception:
        return 0.0
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val) * 100.0

# ===========================
# Load ML artifacts
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
# ML thresholds (for anomaly backstop only)
# ===========================
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857
ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    """
    Map model fraud probability into LOW / MEDIUM / HIGH / CRITICAL
    using normalized fraud risk score (0–100):

      0–30   -> LOW
      30–60  -> MEDIUM
      60–90  -> HIGH
      90–100 -> CRITICAL
    """
    fraud_score = normalize_score(fraud_prob, min_val=0.0, max_val=0.02)
    if fraud_score >= 90.0:
        return "CRITICAL"
    if fraud_score >= 60.0:
        return "HIGH"
    if fraud_score >= 30.0:
        return "MEDIUM"
    return "LOW"

# ===========================
# Base thresholds (INR)
# ===========================
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

# ===========================
# Rule engine
# ===========================
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules: List[Dict] = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel_val = str(payload.get("Channel", "") or "")
    channel = channel_val.lower()
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)

    # beneficiary flags
    new_benef = bool(payload.get("new_beneficiary", False))
    existing_benef = bool(payload.get("existing_beneficiary", False))
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    # location / IP fields
    ip_country = str(payload.get("ip_country", "") or "").lower()
    declared_country = str(payload.get("declared_country", "") or "").lower()

    home_city = str(payload.get("home_city", "") or "").lower()
    home_country = str(payload.get("home_country", "") or "").lower()
    txn_city = str(payload.get("txn_city", "") or "").lower()
    txn_country = str(payload.get("txn_country", "") or "").lower()

    last_device = str(payload.get("device_last_seen", "") or "").lower()
    curr_device = str(payload.get("DeviceID", "") or "").lower()

    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)

    card_country = str(payload.get("card_country", "") or "").lower()
    cvv_provided = bool(payload.get("cvv_provided", True))
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")

    suspicious_ip_flag = bool(payload.get("suspicious_ip_flag", False))
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)

    # identity fields (onsite branch / bank)
    id_type = str(payload.get("id_type", "") or "").strip()
    id_number = str(payload.get("id_number", "") or "").strip()

    # VPN / anonymization (Option A: manual flags)
    vpn_detected = bool(payload.get("vpn_detected", False))
    vpn_provider = str(payload.get("vpn_provider", "") or "")
    tor_exit_node = bool(payload.get("tor_exit_node", False))
    cloud_host_ip = bool(payload.get("cloud_host_ip", False))
    ip_risk_score = int(payload.get("ip_risk_score", 0) or 0)

    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # 1) Absolute large amount
    if amt >= ABS_CRIT:
        add_rule(
            "Absolute very large amount",
            "CRITICAL",
            f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.",
        )

    # 2) Impossible travel based on geo-coordinates
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_checks_enabled = channel not in ("bank", "atm")

    if device_checks_enabled:
        device_new = (not last_device) or last_device == "" or (curr_device and curr_device != last_device)
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule(
                "New device + Impossible travel + High amount",
                "CRITICAL",
                f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.",
            )

    # 3) Multiple beneficiaries + high transfer
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule(
            "Multiple beneficiaries added + high transfer",
            "CRITICAL",
            f"{beneficiaries_added_24h} beneficiaries added and amount {amt:.2f} {currency}.",
        )

    # 4) Velocity rules
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} txns in last 24h.")

    # 5) IP / declared mismatch (if you pass declared_country as KYC/home country)
    if ip_country and declared_country and ip_country != declared_country and channel not in ("bank", "atm"):
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule(
            "IP / Declared country mismatch",
            sev,
            f"IP country '{ip_country}' differs from declared '{declared_country}'.",
        )

    # 6) Login security
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed auth attempts.")

    # 7) New beneficiary + amount
    if new_benef and amt >= MED_AMT:
        add_rule(
            "New beneficiary + significant amount",
            "HIGH",
            "Transfer to newly added beneficiary with amount above threshold.",
        )

    # 8) IP flagged as risky
    if suspicious_ip_flag and amt > (MED_AMT / 4):
        add_rule("IP flagged by threat intelligence", "HIGH", "IP flagged and non-trivial amount.")

    # 9) ATM distance from last known location
    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km away.")

    # 10) Card issuing country mismatch vs home
    if card_country and home_country and card_country != home_country and amt > MED_AMT:
        add_rule(
            "Card country mismatch vs home country",
            "HIGH",
            f"Card country {card_country} != home country {home_country}.",
        )

    # 11) Amount vs historical spending patterns
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add_rule(
            "Large spike vs monthly avg",
            "HIGH",
            f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.",
        )
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule(
            "Spike vs 7-day avg",
            "MEDIUM",
            f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.",
        )
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule(
            "Above monthly usual",
            "MEDIUM",
            f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.",
        )

    # 12) Additional velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24h.")

    # 13) Time-of-day rules (unusual times)
    if 0 <= hour <= 5 and monthly_avg < (MED_AMT * 2) and amt > (MED_AMT / 10):
        add_rule(
            "Late-night txn for low-activity customer",
            "MEDIUM",
            f"Txn at hour {hour} for low-activity customer; amt {amt:.2f}.",
        )

    if 0 <= hour <= 4 and amt >= HIGH_AMT:
        add_rule(
            "Very high amount during unusual time",
            "HIGH",
            f"Txn at hour {hour} with amount {amt:.2f} {currency} >= high threshold {HIGH_AMT:.2f}.",
        )

    # 14) Device new + low amount (benign)
    if device_checks_enabled and ((not last_device) or last_device == "") and amt < (MED_AMT / 10):
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    # 15) Recently added beneficiaries but not extreme
    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW", f"{beneficiaries_added_24h} beneficiaries added.")

    # 16) Higher-risk countries based on transaction country
    high_risk_countries = {"nigeria", "romania", "ukraine", "russia"}
    if txn_country and txn_country in high_risk_countries:
        add_rule(
            "Transaction in higher-risk country",
            "MEDIUM",
            f"Transaction country flagged as higher-risk: {txn_country}.",
        )

    # 17) Card testing / micro-charges
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule(
            "Card testing / micro-charges detected",
            "HIGH",
            f"{card_small_attempts} small attempts; micro amount {CARD_TEST_SMALL:.2f} {currency}.",
        )

    # 18) Large ATM withdrawal
    if channel == "atm" and amt >= ATM_HIGH:
        add_rule(
            "Large ATM withdrawal",
            "HIGH",
            f"ATM withdrawal {amt:.2f} {currency} >= {ATM_HIGH:.2f}",
        )

    # 19) POS repeat
    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS.")

    # 20) Immediate transfer to just-added beneficiary (bank / netbanking)
    if channel in ("netbanking", "bank") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add_rule(
            "Immediate transfer to newly added beneficiary",
            "HIGH",
            f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.",
        )

    # 21) Home vs transaction city/country
    if home_country and txn_country and home_country != txn_country:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add_rule(
            "Txn country differs from home country",
            sev,
            f"Home country '{home_country}' vs transaction country '{txn_country}'.",
        )

    if home_city and txn_city and home_city != txn_city and amt >= (MED_AMT / 2):
        add_rule(
            "Txn city differs from home city",
            "MEDIUM",
            f"Home city '{home_city}' vs transaction city '{txn_city}'.",
        )

    # 22) Transfer-specific structural checks
    if str(payload.get("TransactionType", "")).upper() == "TRANSFER":
        from_acc = payload.get("from_account_number")
        to_acc = payload.get("to_account_number")
        if not from_acc or not to_acc:
            add_rule(
                "Missing transfer account data",
                "HIGH",
                "Transfer missing source or destination account details.",
            )

    # 23) Onsite branch transactions without identity
    if channel == "bank":
        if not id_type or not id_number:
            add_rule(
                "Onsite branch transaction without identity",
                "HIGH",
                "No identity document captured for onsite branch transaction.",
            )

    # 24) VPN / TOR / cloud-host IP rules (manual flags)
    if tor_exit_node:
        add_rule(
            "Connection via TOR exit node",
            "CRITICAL",
            "Traffic is coming from a TOR exit node – highly anonymized.",
        )
    if vpn_detected:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add_rule(
            "VPN usage detected",
            sev,
            f"Upstream systems flagged VPN usage ({vpn_provider or 'provider not specified'}).",
        )
    if cloud_host_ip and channel not in ("atm", "bank"):
        add_rule(
            "Connection from cloud hosting provider",
            "MEDIUM",
            "IP appears to belong to a cloud hosting provider, not a residential ISP.",
        )
    if ip_risk_score >= 80:
        add_rule(
            "High IP reputation risk score",
            "HIGH",
            f"IP reputation risk score {ip_risk_score} (>= 80).",
        )
    elif ip_risk_score >= 50:
        add_rule(
            "Elevated IP reputation risk score",
            "MEDIUM",
            f"IP reputation risk score {ip_risk_score} (>= 50).",
        )

    # Determine highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest

# ===========================
# Combine ML + Rules
# ===========================
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)

# ===========================
# ML scoring
# ===========================
def score_transaction_ml(
    supervised_pipeline,
    iforest_pipeline,
    model_payload: Dict,
    convert_to_inr: bool = False,
    currency: str = "INR",
) -> Tuple[float, float, str]:
    amt_for_model = model_payload.get("Amount", 0.0)
    if convert_to_inr:
        amt_for_model = amt_for_model * INR_PER_UNIT.get(currency, 1.0)

    model_df = pd.DataFrame(
        [
            {
                "Amount": amt_for_model,
                "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
                "Location": model_payload.get("txn_city", model_payload.get("Location", "Unknown")),
                "DeviceID": model_payload.get("DeviceID", "Unknown"),
                "Channel": model_payload.get("Channel", "Other"),
                "hour": model_payload.get("hour", 0),
                "day_of_week": model_payload.get("day_of_week", 0),
                "month": model_payload.get("month", 0),
            }
        ]
    )

    try:
        fraud_prob = float(supervised_pipeline.predict_proba(model_df)[0, 1])
    except Exception:
        fraud_prob = 0.0
    try:
        raw = float(iforest_pipeline.decision_function(model_df)[0])
        anomaly_score = -raw
    except Exception:
        anomaly_score = 0.0

    ml_label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, ml_label
