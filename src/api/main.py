# Block 1: Imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Block 2: Create App and Load Model
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a customer will churn or not",
    version="1.0.0"
)

model = joblib.load("models/churn_model.pkl")

# Block 3: Input Schema
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# Block 4: Endpoints

# Endpoint 1: Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Endpoint 2: Prediction
@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.dict()])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if probability >= 0.7:
            risk = "HIGH"
        elif probability >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "churn_probability": round(float(probability), 4),
            "risk_category": risk,
            "prediction": int(prediction),
            "message": "Churn predicted" if prediction == 1 else "No churn predicted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))