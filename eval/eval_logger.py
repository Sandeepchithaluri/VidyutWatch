import csv
import json
import os
from datetime import datetime

LOG_FILE = "eval/eval_logs.csv"

def setup_eval_log():
    """Create CSV log file with headers if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "meter_id",
                "units_consumed",
                "total_amount",
                "llm_parsed_correctly",
                "fraud_verdict",
                "fraud_score",
                "rag_context",
                "actual_label",  # manually set later
                "correct_prediction"  # calculated later
            ])
        print("✅ Eval log file created!")


def log_prediction(meter_id, parsed_bill, fraud_result, rag_context, actual_label=None):
    """Log a single prediction to CSV"""
    setup_eval_log()

    # Check if LLM parsed correctly
    llm_parsed = all([
        parsed_bill.get("units_consumed") is not None,
        parsed_bill.get("total_amount") is not None,
        parsed_bill.get("meter_id") is not None
    ])

    # Check if prediction matches actual label
    correct = None
    if actual_label:
        predicted_fraud = fraud_result.get("is_fraud", False)
        correct = (predicted_fraud == actual_label)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        meter_id,
        parsed_bill.get("units_consumed"),
        parsed_bill.get("total_amount"),
        llm_parsed,
        fraud_result.get("label"),
        fraud_result.get("fraud_score"),
        rag_context[:100] if rag_context else "",  # first 100 chars
        actual_label,
        correct
    ]

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"✅ Prediction logged for meter {meter_id}")


def get_accuracy_stats():
    """Calculate accuracy from logs"""
    if not os.path.exists(LOG_FILE):
        return {"error": "No logs found"}

    total = 0
    correct = 0
    llm_correct = 0
    fraud_count = 0
    normal_count = 0

    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)

    if total == 0:
        return {"error": "No predictions logged yet"}

    for row in rows:
        # LLM accuracy
        if row["llm_parsed_correctly"] == "True":
            llm_correct += 1

        # Fraud counts
        if "Fraud" in str(row["fraud_verdict"]):
            fraud_count += 1
        elif "Normal" in str(row["fraud_verdict"]):
            normal_count += 1

        # Overall accuracy (only where actual label exists)
        if row["correct_prediction"] == "True":
            correct += 1

    labeled = sum(1 for r in rows if r["correct_prediction"] != "")

    stats = {
        "total_predictions": total,
        "llm_parse_accuracy": f"{round(llm_correct/total*100, 1)}%",
        "fraud_detected": fraud_count,
        "normal_bills": normal_count,
        "labeled_predictions": labeled,
        "accuracy": f"{round(correct/labeled*100, 1)}%" if labeled > 0 else "N/A"
    }

    return stats


def show_recent_logs(n=10):
    """Show last n predictions"""
    if not os.path.exists(LOG_FILE):
        print("No logs found!")
        return []

    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    recent = rows[-n:] if len(rows) >= n else rows

    print(f"\n--- LAST {len(recent)} PREDICTIONS ---")
    for row in recent:
        print(f"[{row['timestamp']}] Meter:{row['meter_id']} "
              f"Units:{row['units_consumed']} "
              f"Verdict:{row['fraud_verdict']} "
              f"Score:{row['fraud_score']}")

    return recent


if __name__ == "__main__":
    # Test the logger
    setup_eval_log()

    # Simulate 5 predictions
    test_cases = [
        {
            "parsed": {"meter_id": "1412", "units_consumed": 95.0, "total_amount": 470.0},
            "fraud": {"label": "🟢 Normal", "fraud_score": 0.2, "is_fraud": False},
            "rag": "Normal consumption compared to history",
            "actual": False
        },
        {
            "parsed": {"meter_id": "1412", "units_consumed": 500.0, "total_amount": 1825.0},
            "fraud": {"label": "🔴 Fraud Detected", "fraud_score": 0.9, "is_fraud": True},
            "rag": "Units 5x higher than average — suspicious!",
            "actual": True
        },
        {
            "parsed": {"meter_id": "1412", "units_consumed": 102.0, "total_amount": 455.0},
            "fraud": {"label": "🟢 Normal", "fraud_score": 0.15, "is_fraud": False},
            "rag": "Normal consumption compared to history",
            "actual": False
        },
        {
            "parsed": {"meter_id": "1412", "units_consumed": 5.0, "total_amount": 107.0},
            "fraud": {"label": "🟡 Suspicious", "fraud_score": 0.55, "is_fraud": True},
            "rag": "Units 95% lower than average — meter tampering possible!",
            "actual": True
        },
        {
            "parsed": {"meter_id": "1412", "units_consumed": 88.0, "total_amount": 420.0},
            "fraud": {"label": "🟢 Normal", "fraud_score": 0.18, "is_fraud": False},
            "rag": "Normal consumption compared to history",
            "actual": False
        }
    ]

    print("Logging 5 test predictions...")
    for case in test_cases:
        log_prediction(
            meter_id=case["parsed"]["meter_id"],
            parsed_bill=case["parsed"],
            fraud_result=case["fraud"],
            rag_context=case["rag"],
            actual_label=case["actual"]
        )

    # Show stats
    print("\n--- EVALUATION STATS ---")
    stats = get_accuracy_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Show recent logs
    show_recent_logs(5)