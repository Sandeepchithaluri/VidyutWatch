import chromadb
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.database import get_bills_by_meter
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB (stores locally — no internet needed)
chroma_client = chromadb.PersistentClient(path="rag/chroma_store")
collection = chroma_client.get_or_create_collection(name="electricity_bills")


def bill_to_text(bill):
    """Convert a bill dictionary into readable text for embedding"""
    return (
        f"Meter {bill['meter_id']} for {bill['billing_period']}: "
        f"{bill['units_consumed']} units consumed, "
        f"energy charges ₹{bill['energy_charges']}, "
        f"total amount ₹{bill['total_amount']}"
    )


def store_bills_in_chromadb(meter_id):
    """Fetch bills from MySQL and store as embeddings in ChromaDB"""
    print(f"Fetching bills for meter {meter_id} from database...")
    bills = get_bills_by_meter(meter_id)

    if not bills:
        print("No bills found in database!")
        return

    documents = []
    metadatas = []
    ids = []

    for bill in bills:
        text = bill_to_text(bill)
        documents.append(text)
        metadatas.append({
            "meter_id": str(bill["meter_id"]),
            "billing_period": str(bill["billing_period"]),
            "units_consumed": float(bill["units_consumed"] or 0),
            "total_amount": float(bill["total_amount"] or 0)
        })
        ids.append(f"bill_{bill['id']}")

    # Store in ChromaDB
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Stored {len(documents)} bills in ChromaDB!")


def retrieve_similar_bills(new_bill_text, meter_id, n_results=5):
    """Find similar past bills using semantic search"""
    print(f"\nSearching for similar past bills...")

    results = collection.query(
        query_texts=[new_bill_text],
        n_results=n_results,
        where={"meter_id": str(meter_id)}
    )

    similar_bills = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            similar_bills.append({
                "text": doc,
                "metadata": meta
            })
            print(f"  Found: {doc}")

    return similar_bills


def analyze_with_rag(new_bill, meter_id):
    """
    Full RAG pipeline:
    1. Convert new bill to text
    2. Retrieve similar past bills
    3. Return context for fraud analysis
    """
    new_bill_text = bill_to_text(new_bill)
    print(f"\nAnalyzing new bill: {new_bill_text}")

    # Retrieve similar past bills
    similar_bills = retrieve_similar_bills(new_bill_text, meter_id)

    if not similar_bills:
        return {"context": "No historical data found", "similar_bills": []}

    # Build context summary
    avg_units = sum(b["metadata"]["units_consumed"] for b in similar_bills) / len(similar_bills)
    avg_amount = sum(b["metadata"]["total_amount"] for b in similar_bills) / len(similar_bills)

    context = (
        f"Historical average for meter {meter_id}: "
        f"{avg_units:.1f} units/month, ₹{avg_amount:.2f}/month. "
        f"New bill shows {new_bill.get('units_consumed')} units and ₹{new_bill.get('total_amount')}."
    )

    # Check for anomaly
    units = new_bill.get("units_consumed") or 0
    if units > avg_units * 1.5:
        context += " ⚠️ Units consumed is 50% higher than average — suspicious!"
    elif units < avg_units * 0.5:
        context += " ⚠️ Units consumed is 50% lower than average — possible meter tampering!"
    else:
        context += " ✅ Consumption looks normal compared to history."

    print(f"\n--- RAG CONTEXT ---")
    print(context)

    return {
        "context": context,
        "similar_bills": similar_bills,
        "avg_units": avg_units,
        "avg_amount": avg_amount
    }


if __name__ == "__main__":
    # Step 1 — Store all bills from MySQL into ChromaDB
    store_bills_in_chromadb("1412")

    # Step 2 — Test with a new bill
    new_bill = {
        "meter_id": "1412",
        "units_consumed": 95.0,
        "energy_charges": 324.4,
        "fixed_charges": 90.0,
        "total_amount": 470.0,
        "billing_period": "Mar 2024"
    }

    result = analyze_with_rag(new_bill, "1412")

    print("\n--- FINAL RAG RESULT ---")
    print(f"Context: {result['context']}")
    print(f"Similar bills found: {len(result['similar_bills'])}")