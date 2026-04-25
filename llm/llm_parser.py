import json
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

load_dotenv(override=True)
print("KEY BEING USED:", os.getenv("GROQ_API_KEY")[:15])
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def parse_bill_with_llm(raw_text):
    print("Sending extracted text to Groq LLM...")

    prompt = f"Extract electricity bill data into JSON: {raw_text}"
    
    # System instructions help define the 'persona' and rules more strictly
    system_prompt = (
        "You are an Indian utility bill parser. Return ONLY a JSON object. "
        "Fields: consumer_name, meter_id, units_consumed (float), energy_charges (float), "
        "fixed_charges (float), electricity_duty (float), total_amount (float), "
        "billing_date (DD/MM/YYYY), billing_period. Use null for missing values."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"} # Force JSON mode
        )

        result_text = response.choices[0].message.content.strip()
        return json.loads(result_text)

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None
if __name__ == "__main__":
    # Read from your actual OCR output
    with open("ocr/extracted_text.txt", "r") as f:
        test_text = f.read()

    result = parse_bill_with_llm(test_text)

    if result:
        print("\n--- STRUCTURED BILL DATA ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        # Save to file
        with open("llm/parsed_bill.json", "w") as f:
            import json
            json.dump(result, f, indent=2)
        print("\n✅ Saved to llm/parsed_bill.json")