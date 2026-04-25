import mysql.connector
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD"),
        database="vidyutwatch"
    )

def setup_database():
    # First connect without database to create it
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("MYSQL_PASSWORD")
    )
    cursor = conn.cursor()

    # Create database
    cursor.execute("CREATE DATABASE IF NOT EXISTS vidyutwatch")
    cursor.execute("USE vidyutwatch")

    # Create bills table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            id INT AUTO_INCREMENT PRIMARY KEY,
            meter_id VARCHAR(50),
            consumer_name VARCHAR(100),
            units_consumed FLOAT,
            energy_charges FLOAT,
            fixed_charges FLOAT,
            electricity_duty FLOAT,
            total_amount FLOAT,
            billing_date VARCHAR(20),
            billing_period VARCHAR(20),
            is_fraud BOOLEAN DEFAULT FALSE,
            fraud_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database and table created successfully!")

def insert_bill(parsed_data):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO bills 
        (meter_id, consumer_name, units_consumed, energy_charges, 
         fixed_charges, electricity_duty, total_amount, billing_date, billing_period)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        parsed_data.get("meter_id"),
        parsed_data.get("consumer_name"),
        parsed_data.get("units_consumed"),
        parsed_data.get("energy_charges"),
        parsed_data.get("fixed_charges"),
        parsed_data.get("electricity_duty"),
        parsed_data.get("total_amount"),
        parsed_data.get("billing_date"),
        parsed_data.get("billing_period")
    )

    cursor.execute(query, values)
    conn.commit()
    bill_id = cursor.lastrowid
    cursor.close()
    conn.close()
    print(f"✅ Bill saved to database with ID: {bill_id}")
    return bill_id

def get_bills_by_meter(meter_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM bills WHERE meter_id = %s ORDER BY created_at DESC", (meter_id,))
    bills = cursor.fetchall()
    cursor.close()
    conn.close()
    return bills

def generate_synthetic_data():
    """Generate 12 months of fake bill data for testing"""
    import random
    conn = get_connection()
    cursor = conn.cursor()

    months = [
        ("Jan 2024","01/01/2024"), ("Feb 2024","01/02/2024"),
        ("Mar 2024","01/03/2024"), ("Apr 2024","01/04/2024"),
        ("May 2024","01/05/2024"), ("Jun 2024","01/06/2024"),
        ("Jul 2024","01/07/2024"), ("Aug 2024","01/08/2024"),
        ("Sep 2024","01/09/2024"), ("Oct 2024","01/10/2024"),
        ("Nov 2024","01/11/2024"), ("Dec 2024","01/12/2024")
    ]

    for period, date in months:
        units = random.randint(80, 120)  # normal range
        energy = round(units * 3.41, 2)
        fixed = 90.0
        duty = round(energy * 0.0175, 2)
        total = round(energy + fixed + duty, 2)

        cursor.execute("""
            INSERT INTO bills 
            (meter_id, consumer_name, units_consumed, energy_charges,
             fixed_charges, electricity_duty, total_amount, billing_date, billing_period)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, ("1412", "Chithaluri", units, energy, fixed, duty, total, date, period))

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 12 months synthetic data generated!")


if __name__ == "__main__":
    setup_database()

    # Insert real parsed bill
    with open("llm/parsed_bill.json", "r") as f:
        parsed = json.load(f)
    insert_bill(parsed)

    # Generate synthetic historical data
    generate_synthetic_data()

    # Test retrieval
    bills = get_bills_by_meter("1412")
    print(f"\n✅ Found {len(bills)} bills for meter 1412")
    for bill in bills[:3]:
        print(f"  {bill['billing_period']}: {bill['units_consumed']} units — ₹{bill['total_amount']}")