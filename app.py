from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------
# FastAPI Setup
# -----------------------

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# -----------------------
# Load Model
# -----------------------

model = joblib.load("churn_model.joblib")

# -----------------------
# Homepage
# -----------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
# -----------------------
# Input Schema
# -----------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float

# -----------------------
# Feature Engineering
# -----------------------

def create_tenure_group(tenure):
    if tenure <= 9:
        return "Q1"
    elif tenure <= 29:
        return "Q2"
    elif tenure <= 55:
        return "Q3"
    else:
        return "Q4"

# -----------------------
# Prediction Route
# -----------------------

@app.post("/predict")
def predict(customer: CustomerData):

    data = customer.dict()

    # Create tenure group
    data["tenure_group"] = create_tenure_group(data["tenure"])

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Force exact column order
    expected_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "tenure_group"
    ]

    df = df[expected_columns]

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    threshold = 0.35

    prediction = "Yes" if prob >= threshold else "No"

    return {
        "prediction": prediction,
        "probability": round(float(prob), 4)
    }