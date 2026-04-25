import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="VidyutWatch",
    page_icon="⚡",
    layout="wide"
)

# API URL
API_URL = "http://127.0.0.1:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1a56a0;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .normal { background-color: #d4edda; color: #155724; }
    .suspicious { background-color: #fff3cd; color: #856404; }
    .fraud { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="main-title">⚡ VidyutWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Electricity Bill Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚡ VidyutWatch")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["🔍 Analyze Bill", "📊 Bill History", "📈 Statistics"])

# ─────────────────────────────────────────
# PAGE 1 — Analyze Bill
# ─────────────────────────────────────────
if page == "🔍 Analyze Bill":
    st.markdown("## 📤 Upload Electricity Bill")
    st.markdown("Upload a photo or scan of your electricity bill for instant fraud analysis.")

    uploaded_file = st.file_uploader(
        "Choose bill image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of your electricity bill"
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded Bill", use_column_width=True)

        with col2:
            st.markdown("### Bill Details")
            if st.button("🔍 Analyze Now", type="primary", use_container_width=True):
                with st.spinner("Running AI analysis... Please wait..."):
                    try:
                        # Call FastAPI
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                        response = requests.post(f"{API_URL}/analyze", files=files, timeout=120)

                        if response.status_code == 200:
                            data = response.json()

                            # Parsed Bill Data
                            st.markdown("#### 📋 Extracted Bill Data")
                            parsed = data["parsed_bill"]
                            bill_df = pd.DataFrame([{
                                "Field": k.replace("_", " ").title(),
                                "Value": v if v is not None else "Not found"
                            } for k, v in parsed.items()])
                            st.dataframe(bill_df, use_container_width=True, hide_index=True)

                            # Fraud Verdict
                            st.markdown("#### 🎯 Fraud Detection Result")
                            fraud = data["fraud_detection"]
                            verdict = fraud["verdict"]
                            score = fraud["fraud_score"]

                            if "Normal" in verdict:
                                css_class = "normal"
                            elif "Suspicious" in verdict:
                                css_class = "suspicious"
                            else:
                                css_class = "fraud"

                            st.markdown(
                                f'<div class="verdict-box {css_class}">{verdict}<br>'
                                f'<small>Fraud Score: {score}</small></div>',
                                unsafe_allow_html=True
                            )

                            # RAG Context
                            st.markdown("#### 🧠 AI Analysis")
                            st.info(data["rag_analysis"]["context"])

                        else:
                            st.error(f"API Error: {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API! Make sure FastAPI is running:\n```\nuvicorn api.main:app --reload\n```")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ─────────────────────────────────────────
# PAGE 2 — Bill History
# ─────────────────────────────────────────
elif page == "📊 Bill History":
    st.markdown("## 📊 Bill History")

    meter_id = st.text_input("Enter Meter ID", value="1412", placeholder="e.g. 1412")

    if st.button("📂 Load History", type="primary"):
        with st.spinner("Loading bill history..."):
            try:
                response = requests.get(f"{API_URL}/history/{meter_id}", timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    bills = data["bills"]

                    if not bills:
                        st.warning("No bills found for this meter ID.")
                    else:
                        st.success(f"Found {data['count']} bills for meter {meter_id}")

                        # Show table
                        df = pd.DataFrame(bills)
                        display_cols = ["billing_period", "units_consumed",
                                      "energy_charges", "total_amount",
                                      "is_fraud", "fraud_score"]
                        df_display = df[[c for c in display_cols if c in df.columns]]
                        df_display.columns = ["Period", "Units", "Energy ₹",
                                            "Total ₹", "Fraud", "Score"]
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

                        # Consumption graph
                        st.markdown("### 📈 Consumption Trend")
                        if "units_consumed" in df.columns and "billing_period" in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            periods = df["billing_period"].tolist()[-12:]
                            units = df["units_consumed"].tolist()[-12:]

                            colors = ["red" if f else "steelblue"
                                    for f in df["is_fraud"].tolist()[-12:]]

                            ax.bar(periods, units, color=colors, alpha=0.8)
                            ax.set_xlabel("Billing Period")
                            ax.set_ylabel("Units Consumed")
                            ax.set_title(f"Consumption History — Meter {meter_id}")
                            ax.axhline(y=sum(units)/len(units),
                                      color="orange", linestyle="--",
                                      label="Average")
                            plt.xticks(rotation=45)
                            plt.legend()
                            plt.tight_layout()
                            st.pyplot(fig)

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─────────────────────────────────────────
# PAGE 3 — Statistics
# ─────────────────────────────────────────
elif page == "📈 Statistics":
    st.markdown("## 📈 Meter Statistics")

    meter_id = st.text_input("Enter Meter ID", value="1412")

    if st.button("📊 Load Stats", type="primary"):
        with st.spinner("Loading statistics..."):
            try:
                response = requests.get(f"{API_URL}/stats/{meter_id}", timeout=30)

                if response.status_code == 200:
                    stats = response.json()

                    if stats.get("status") == "no data":
                        st.warning("No data found for this meter.")
                    else:
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Bills", stats["total_bills"])
                        with col2:
                            st.metric("Avg Units/Month", stats["avg_units"])
                        with col3:
                            st.metric("Avg Amount/Month", f"₹{stats['avg_amount']}")
                        with col4:
                            st.metric("Fraud Detected", stats["fraud_count"],
                                    delta=None if stats["fraud_count"] == 0 else "⚠️")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"🔺 Max Units: {stats['max_units']}")
                        with col2:
                            st.info(f"🔻 Min Units: {stats['min_units']}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888;'>"
    "VidyutWatch v1.0 — Built by Chithaluri Sandeep | AI/ML Engineer"
    "</div>",
    unsafe_allow_html=True
)