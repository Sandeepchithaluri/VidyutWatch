from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.ocr_reader import extract_text_from_bill
from llm.llm_parser import parse_bill_with_llm
from rag.rag_pipeline import analyze_with_rag, store_bills_in_chromadb
from ml.anomaly_detector import train_isolation_forest, predict_fraud
from ml.database import insert_bill, get_bills_by_meter

app = FastAPI(
    title="VidyutWatch API",
    description="AI-powered electricity bill fraud detection system",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def root():
    return {
        "message": "Welcome to VidyutWatch API!",
        "version": "1.0.0",
        "endpoints": [
            "POST /upload-bill",
            "POST /analyze",
            "GET /history/{meter_id}"
        ]
    }


@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...)):
    """
    Step 1: Upload bill image
    - Saves image temporarily
    - Runs OCR to extract text
    - Returns extracted raw text
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run OCR
        extracted_text = extract_text_from_bill(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from image")

        return {
            "status": "success",
            "filename": file.filename,
            "extracted_text": extracted_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_bill(file: UploadFile = File(...)):
    """
    Full pipeline:
    1. OCR — extract text from image
    2. LLM — parse into structured JSON
    3. RAG — compare with historical bills
    4. ML — detect anomalies
    5. Return fraud verdict
    """
    try:
        # Save uploaded file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1 — OCR
        print("Step 1: Running OCR...")
        extracted_text = extract_text_from_bill(temp_path)
        os.remove(temp_path)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="OCR failed")

        # Step 2 — LLM Parsing
        print("Step 2: Parsing with LLM...")
        parsed_bill = parse_bill_with_llm(extracted_text)

        if not parsed_bill:
            raise HTTPException(status_code=400, detail="LLM parsing failed")

        # Step 3 — Save to database
        print("Step 3: Saving to database...")
        bill_id = insert_bill(parsed_bill)

        # Step 4 — RAG Analysis
        print("Step 4: Running RAG analysis...")
        meter_id = parsed_bill.get("meter_id", "unknown")
        store_bills_in_chromadb(meter_id)
        rag_result = analyze_with_rag(parsed_bill, meter_id)

        # Step 5 — ML Fraud Detection
        print("Step 5: Running ML fraud detection...")
        model, scaler = train_isolation_forest(meter_id)
        fraud_result = predict_fraud(parsed_bill, model, scaler)

        # Step 6 — Log prediction
        from eval.eval_logger import log_prediction
        log_prediction(
            meter_id=meter_id,
            parsed_bill=parsed_bill,
            fraud_result=fraud_result,
            rag_context=rag_result["context"]
        )
        
        # Final response
        return {
            "status": "success",
            "bill_id": bill_id,
            "parsed_bill": parsed_bill,
            "rag_analysis": {
                "context": rag_result["context"],
                "avg_units": rag_result.get("avg_units"),
                "avg_amount": rag_result.get("avg_amount")
            },
            "fraud_detection": {
                "verdict": fraud_result["label"],
                "fraud_score": fraud_result["fraud_score"],
                "is_fraud": fraud_result["is_fraud"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{meter_id}")
def get_history(meter_id: str):
    """
    Get all bills for a specific meter ID
    """
    try:
        bills = get_bills_by_meter(meter_id)

        if not bills:
            return {"status": "success", "meter_id": meter_id, "bills": [], "count": 0}

        return {
            "status": "success",
            "meter_id": meter_id,
            "count": len(bills),
            "bills": bills
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{meter_id}")
def get_stats(meter_id: str):
    """
    Get consumption statistics for a meter
    """
    try:
        bills = get_bills_by_meter(meter_id)

        if not bills:
            return {"status": "no data"}

        units = [b["units_consumed"] for b in bills if b["units_consumed"]]
        amounts = [b["total_amount"] for b in bills if b["total_amount"]]

        return {
            "status": "success",
            "meter_id": meter_id,
            "total_bills": len(bills),
            "avg_units": round(sum(units) / len(units), 2),
            "avg_amount": round(sum(amounts) / len(amounts), 2),
            "max_units": max(units),
            "min_units": min(units),
            "fraud_count": sum(1 for b in bills if b["is_fraud"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))