import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.database import get_bills_by_meter, get_connection
from dotenv import load_dotenv

load_dotenv()


def prepare_features(bills):
    """Convert bill records into ML features"""
    data = []
    for bill in bills:
        data.append([
            float(bill["units_consumed"] or 0),
            float(bill["energy_charges"] or 0),
            float(bill["fixed_charges"] or 0),
            float(bill["total_amount"] or 0)
        ])
    return np.array(data)


def train_isolation_forest(meter_id):
    """Train anomaly detection model on historical bills"""
    print(f"Training Isolation Forest for meter {meter_id}...")

    # Get historical bills from database
    bills = get_bills_by_meter(meter_id)

    if len(bills) < 3:
        print("❌ Not enough historical data to train model!")
        return None, None

    # Prepare features
    X = prepare_features(bills)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.1,  # expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)

    print(f"✅ Model trained on {len(bills)} bills!")
    return model, scaler


def predict_fraud(new_bill, model, scaler):
    """Predict if a new bill is fraudulent"""
    if model is None:
        return {"is_fraud": False, "fraud_score": 0.0, "label": "⚠️ Insufficient data"}

    # Prepare features for new bill
    features = np.array([[
        float(new_bill.get("units_consumed") or 0),
        float(new_bill.get("energy_charges") or 0),
        float(new_bill.get("fixed_charges") or 0),
        float(new_bill.get("total_amount") or 0)
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict (-1 = anomaly, 1 = normal)
    prediction = model.predict(features_scaled)[0]

    # Get anomaly score (more negative = more anomalous)
    raw_score = model.score_samples(features_scaled)[0]

    # Convert to 0-1 fraud score (higher = more suspicious)
    fraud_score = round(1 - (raw_score + 0.5), 2)
    fraud_score = max(0.0, min(1.0, fraud_score))  # clamp between 0 and 1

    is_fraud = prediction == -1

    # Assign label
    if fraud_score > 0.7:
        label = "🔴 Fraud Detected"
    elif fraud_score > 0.4:
        label = "🟡 Suspicious"
    else:
        label = "🟢 Normal"

    result = {
        "is_fraud": is_fraud,
        "fraud_score": fraud_score,
        "label": label,
        "units_consumed": new_bill.get("units_consumed"),
        "total_amount": new_bill.get("total_amount")
    }

    return result


def update_fraud_in_db(bill_id, is_fraud, fraud_score):
    """Update fraud result in database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE bills SET is_fraud = %s, fraud_score = %s WHERE id = %s",
        (is_fraud, fraud_score, bill_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Updated fraud status in database for bill ID {bill_id}")


def run_full_analysis(new_bill, meter_id):
    """Complete fraud detection pipeline"""
    print("\n" + "="*50)
    print("VIDYUTWATCH FRAUD DETECTION")
    print("="*50)

    # Train model
    model, scaler = train_isolation_forest(meter_id)

    # Predict
    result = predict_fraud(new_bill, model, scaler)

    print(f"\n--- FRAUD ANALYSIS RESULT ---")
    print(f"Meter ID      : {meter_id}")
    print(f"Units Consumed: {result['units_consumed']}")
    print(f"Total Amount  : ₹{result['total_amount']}")
    print(f"Fraud Score   : {result['fraud_score']}")
    print(f"Verdict       : {result['label']}")

    return result


if __name__ == "__main__":
    # Test with normal bill
    print("\n--- TEST 1: Normal Bill ---")
    normal_bill = {
        "units_consumed": 95.0,
        "energy_charges": 324.4,
        "fixed_charges": 90.0,
        "total_amount": 470.0
    }
    result1 = run_full_analysis(normal_bill, "1412")

    # Test with suspicious bill (very high units)
    print("\n--- TEST 2: Suspicious Bill ---")
    suspicious_bill = {
        "units_consumed": 500.0,
        "energy_charges": 1705.0,
        "fixed_charges": 90.0,
        "total_amount": 1825.0
    }
    result2 = run_full_analysis(suspicious_bill, "1412")

    # Test with fraud bill (suspiciously low units)
    print("\n--- TEST 3: Possible Meter Tampering ---")
    fraud_bill = {
        "units_consumed": 5.0,
        "energy_charges": 17.05,
        "fixed_charges": 90.0,
        "total_amount": 107.0
    }
    result3 = run_full_analysis(fraud_bill, "1412")

    print("\n--- SUMMARY ---")
    print(f"Normal bill   : {result1['label']}")
    print(f"Suspicious    : {result2['label']}")
    print(f"Meter tamper  : {result3['label']}")
    