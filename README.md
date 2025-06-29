# 🔍 Customer Churn Prediction and Retention System (CPRS)


---

## 🧠 Overview

**CPRS** is a machine learning-powered system designed to predict customer churn and suggest proactive retention strategies. It combines an interpretable XGBoost model, a FastAPI-based backend, and an interactive Streamlit dashboard for business users.

---

## 📦 Features

- ✅ Predicts customer churn using behavior, engagement, and financial features.
- 📊 Calculates churn probability and segments customers by risk.
- 🧠 Provides rule-based explanations for churn reasons.
- 💡 Suggests personalized retention strategies.
- 📄 Generates downloadable PDF & CSV reports.
- 🖥️ Streamlit UI with sliders and prediction history.
- 🌐 FastAPI backend for scalable deployment.

---

## 🚀 Project Architecture

```
User Inputs (Streamlit) --> FastAPI API --> XGBoost Model --> Response
                                ↑
               selected_features.pkl | xgboost_model_shap.pkl
```

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Model:** XGBoost with SHAP explanations
- **Reporting:** FPDF
- **Deployment:** Render

https://github.com/tushardanao/CPRS-Project
---

## 🛠️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/tushardanao/CPRS-Project.git
cd CPRS-Project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Start FastAPI server

```bash
uvicorn crps:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start Streamlit frontend

```bash
streamlit run Churn_prediction.py
```

---

## 🌐 Deployment (Render)

### Files used:
- `render.yaml` – defines build and start steps
- `requirements.txt`, `runtime.txt` – specify environment

To deploy:

1. Push the code to a GitHub repo.
2. Log in to [Render](https://render.com).
3. Create a new **Web Service**.
4. Connect your GitHub repo.
5. Render will automatically pick up `render.yaml` and deploy the app.

📈 Live Demo
FastAPI Link - https://cprs-project-1.onrender.com/ (Firstly run this link)
Streamlit interface link - https://cprs-project-3.onrender.com/ (Then run this link)

---

## 📊 Input Features

| Feature Name           | Description                             |
|------------------------|-----------------------------------------|
| loans_accessed         | Number of loans accessed                |
| loans_taken            | Number of loans applied/taken           |
| login_total            | App login activity score (0-1)          |
| overdraft_events       | Number of overdraft events              |
| tickets_raised         | Customer support tickets raised         |
| sentiment_score        | Sentiment score from feedback (0-1)     |
| monthly_avg_balance    | Monthly average account balance         |
| credit_score           | Customer’s credit score (0-850)         |

---

## 🧠 Rule-Based Churn Explanation

The backend uses logical rules to explain why a customer might churn. For example:
- Low sentiment → “low customer sentiment”
- High overdraft → “frequent overdraft events”
- Few logins → “very low app login activity”

---

## 📁 Project Structure

```
.
├── Churn_prediction.py        # Streamlit UI
├── crps.py                    # FastAPI backend
├── selected_features.pkl      # Features used by model
├── xgboost_model_shap.pkl     # Trained model
├── company_logo.png           # Branding logo
├── requirements.txt           # Dependencies
├── runtime.txt                # Python version
├── render.yaml                # Render deployment config
```

---

## 📌 Sample Output

```json
{
  "prediction": 1,
  "churn_probability": 0.9412,
  "risk_segment": "🔴 High Risk",
  "message": "Churn risk detected – customer requires retention intervention.",
  "churn_reason": "The customer is likely to churn because very low app login activity; low customer sentiment.",
  "recommended_strategy": "🚨 Assign retention agent within 24 hours and offer loyalty benefit.",
  "final_action": "🔴 High Risk → 🚨 Assign retention agent within 24 hours and offer loyalty benefit."
}
