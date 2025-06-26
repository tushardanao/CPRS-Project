from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model and features
model = joblib.load("xgboost_model_shap.pkl")
features = joblib.load("selected_features.pkl")

app = FastAPI(title="CPRS | Churn Prediction API")

class ChurnInput(BaseModel):
    data: list

# Rule-based churn reason engine
def churn_reason(row):
    reasons = []
    if row.get("loans_accessed", 1) == 0:
        reasons.append("customer hasn’t accessed loans recently")
    if row.get("loans_taken", 0) > 4:
        reasons.append("multiple declined loan attempts")
    if row.get("login_total", 1.0) < 0.3:
        reasons.append("very low app login activity")
    if row.get("overdraft_events", 0) > 2:
        reasons.append("frequent overdraft events")
    if row.get("tickets_raised", 0) > 3:
        reasons.append("multiple support tickets raised")
    if row.get("sentiment_score", 1.0) < 0.3:
        reasons.append("low customer sentiment")
    if row.get("monthly_avg_balance", 10000) < 1000:
        reasons.append("consistently low average balance")
    if row.get("credit_score", 850) < 400:
        reasons.append("very poor credit score")
    return reasons

# Risk segmentation
def final_risk_segment(prob):
    if prob > 0.80:
        return "🔴 High Risk"
    elif prob >= 0.40:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

# Strategy recommendation
def recommended_action(prob):
    if prob > 0.80:
        return "🚨 Assign retention agent within 24 hours and offer loyalty benefit."
    elif prob >= 0.40:
        return "⚠️ Send engagement email and monitor usage weekly."
    else:
        return "✅ Send appreciation note and track monthly."

# Churn summary message
def churn_message(pred):
    return (
        "Churn risk detected – customer requires retention intervention."
        if pred == 1 else
        "Customer appears stable – no immediate churn indicators."
    )

@app.get("/")
def home():
    return {"message": "Welcome to the CPRS Churn Prediction API"}

@app.post("/predict")
def predict_churn(input_data: ChurnInput):
    try:
        if len(input_data.data) != len(features):
            raise HTTPException(status_code=400, detail=f"Expected {len(features)} features.")

        # Prepare input for prediction
        X = np.array(input_data.data).reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        # Row for reason logic
        row = dict(zip(features, input_data.data))
        risk_label = final_risk_segment(proba)
        action = recommended_action(proba)
        message = churn_message(pred)

        # Final churn reason logic
        if pred == 1:
            extracted = churn_reason(row)
            if extracted:
                reason = "The customer is likely to churn because " + "; ".join(extracted) + "."
            else:
                if risk_label == "🟠 Medium Risk":
                    reason = "Churn predicted due to moderate behavioral signals not captured in current rule set."
                else:
                    reason = "Churn predicted, but no dominant behavioral churn signals identified."
        else:
            reason = "No strong churn signals detected. Customer appears healthy based on current usage behavior."

        # Return structured response
        return {
            "prediction": int(pred),
            "churn_probability": round(float(proba), 4),
            "risk_segment": risk_label,
            "message": message,
            "churn_reason": reason,
            "recommended_strategy": action,
            "final_action": f"{risk_label} → {action}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("crps:app", host="0.0.0.0", port=8000, reload=True)
