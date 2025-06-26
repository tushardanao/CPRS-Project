import streamlit as st
import requests
import pickle
import pandas as pd
from fpdf import FPDF
import base64
import re

# Load selected features
with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

API_URL = "http://127.0.0.1:8000/predict"

# Store history
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# Encode logo image as base64
def get_logo_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Generate PDF
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', str(text))

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Churn Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for key, value in data.items():
        clean_key = remove_emojis(str(key))
        clean_value = remove_emojis(str(value))
        pdf.multi_cell(0, 10, f"{clean_key}: {clean_value}")
    return pdf.output(dest="S").encode("latin-1")

# Page setup
st.set_page_config(page_title="CPRS Churn Predictor", layout="wide")

# Show centered logo + title
logo_base64 = get_logo_base64("company_logo.png")
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" width="300" style="border-radius: 10px;" />
        <h2 style="margin-top: 10px; color: #1F4E79;">📉 Customer Churn Prediction System</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("🎛️ Customer Input Features")
inputs = {
    "loans_accessed": st.sidebar.slider("Loans Accessed", 0, 10, 0),
    "loans_taken": st.sidebar.slider("Loans Taken", 0, 10, 0),
    "login_total": st.sidebar.slider("Login Total (0-1)", 0.0, 1.0, 0.0, 0.01),
    "overdraft_events": st.sidebar.slider("Overdraft Events", 0, 10, 0),
    "tickets_raised": st.sidebar.slider("Tickets Raised", 0, 10, 0),
    "sentiment_score": st.sidebar.slider("Sentiment Score (0-1)", 0.0, 1.0, 0.0, 0.01),
    "monthly_avg_balance": st.sidebar.slider("Monthly Avg Balance", 0, 100000, 0, 500),
    "credit_score": st.sidebar.slider("Credit Score", 0, 850, 0, 1),
}

input_data = {"data": [inputs[feat] for feat in selected_features]}

# Predict button
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()

                # Display prediction
                st.success("✅ Prediction Complete")
                churn_text = "🟥 Churn" if result["prediction"] == 1 else "🟩 No Churn"
                st.markdown(f"### 🔎 Prediction: {churn_text}")
                st.markdown(f"**Churn Probability:** `{result['churn_probability']}`")
                st.markdown(f"**Risk Segment:** {result['risk_segment']}")
                st.markdown(f"**Churn Reason:** {result['churn_reason']}")
                st.markdown(f"**Message:** {result['message']}")

                with st.expander("📋 Recommended Strategy"):
                    st.markdown(f"**Strategy:** {result['recommended_strategy']}")
                    st.markdown(f"**Final Action:** {result['final_action']}")

                # Store history
                full_result = {**inputs, **result}
                st.session_state.prediction_log.append(full_result)

                # Export buttons
                pdf = generate_pdf(full_result)
                st.download_button("📄 Download PDF", pdf, file_name="churn_report.pdf", mime="application/pdf")

                df = pd.DataFrame([full_result])
                st.download_button("📊 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="churn_prediction.csv", mime="text/csv")

            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Request Failed: {str(e)}")

# History Table
if st.session_state.prediction_log:
    st.markdown("## 📈 Prediction History")
    hist_df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(hist_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Full Prediction History",
        hist_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_prediction_history.csv",
        mime="text/csv"
    )
