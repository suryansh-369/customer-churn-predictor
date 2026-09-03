📉 Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn probability through a REST API, with MLflow experiment tracking and Dockerized deployment.







🚀 Project Overview

Customer churn prediction helps identify customers who are likely to leave a service.

This project takes customer information, processes it through a reusable preprocessing pipeline, and uses a Logistic Regression classifier to predict:

Whether the customer is likely to churn

The probability of churn

A simple risk category: LOW, MEDIUM, or HIGH

The trained pipeline is exposed through a FastAPI REST API and can be run inside a Docker container.

🧠 ML Pipeline

Customer Churn Dataset
        │
        ▼
   Data Cleaning
        │
        ▼
 Train / Test Split
        │
        ▼
 ┌───────────────────────────┐
 │      Preprocessing        │
 │                           │
 │ Numeric → Impute → Scale  │
 │ Categorical → Impute      │
 │              → Encode     │
 └───────────────────────────┘
        │
        ▼
 Logistic Regression
        │
        ├───────────────┐
        ▼               ▼
  Cross Validation   Test Set
        │               │
        └───────┬───────┘
                ▼
       AUC / F1 / Report
                │
                ▼
          MLflow Tracking
                │
                ▼
       churn_model.pkl
                │
                ▼
           FastAPI API
                │
                ▼
             Docker

✨ Features

🧹 Automated data cleaning

🔄 Reusable scikit-learn preprocessing pipeline

🤖 Logistic Regression churn classifier

📊 5-fold ROC-AUC cross-validation

📈 Test ROC-AUC and F1 evaluation

🔬 MLflow experiment tracking

💾 Saved model pipeline with Joblib

🚀 FastAPI REST API

✅ Pydantic request validation

❤️ Health-check endpoint

🐳 Dockerized deployment

📚 Interactive Swagger API documentation

🛠️ Tech Stack

Technology

Purpose

Python 3.11

Core development

Pandas

Data loading and manipulation

NumPy

Numerical operations

scikit-learn 1.7.1

Preprocessing, evaluation and model

Logistic Regression

Churn classification

MLflow

Experiment tracking and model logging

Joblib

Model persistence

FastAPI

REST API

Pydantic

Request validation

Uvicorn

ASGI server

Docker

Containerized deployment

📁 Project Structure

Churn project/
│
├── data/
│   └── raw/
│       └── churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   └── models/
│       └── train.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

Local MLflow tracking files such as mlflow.db and mlruns/ are intentionally excluded from version control.

🔬 Machine Learning

Data Preparation

The training script:

Loads data/raw/churn.csv

Removes customerID

Converts the Churn target from Yes/No to 1/0

Converts TotalCharges to numeric values

Removes rows where TotalCharges cannot be converted

Feature Processing

Numeric Features

The numeric features are:

tenure
MonthlyCharges
TotalCharges
SeniorCitizen

They are processed using:

Median Imputation
       ↓
StandardScaler

Categorical Features

Categorical customer attributes are processed using:

Most-Frequent Imputation
       ↓
OrdinalEncoder

Unknown categories are encoded with -1.

Both preprocessing paths are combined using a ColumnTransformer.

🤖 Model

The current model is:

Logistic Regression

with:

C = 1.0
max_iter = 1000

The preprocessing and classifier are stored together in a single scikit-learn Pipeline.

This allows the API to receive raw customer fields and pass them directly through the same preprocessing steps used during training.

📊 Model Evaluation

The training workflow uses:

5-fold ROC-AUC cross-validation

Test ROC-AUC

Test F1 score

Classification report

The training script prints the cross-validation AUC, test AUC, test F1 score, and classification report after training.

Model performance values are intentionally not hard-coded here because they depend on the current training run. Run the training script to generate the latest metrics.

🔬 MLflow Experiment Tracking

MLflow is used to record the training experiment.

The experiment is:

churn-prediction

The training run records:

Parameters

C
max_iter

Metrics

test_auc
test_f1
cv_auc
cv_std

Model Artifact

The complete scikit-learn pipeline is logged to MLflow and registered as:

ChurnPredictor

MLflow is configured to use a local SQLite tracking database during development.

🚀 REST API

The API is built with FastAPI.

Start the API locally

From the project root:

python -m uvicorn src.api.main:app --reload

The API will be available at:

http://localhost:8000

Interactive API Documentation

Open:

http://localhost:8000/docs

FastAPI automatically provides an interactive Swagger UI where you can test the endpoints.

❤️ Health Check

GET /health

Checks whether the API is running and whether the model has been loaded.

Example response:

{
  "status": "healthy",
  "model_loaded": true
}

🔮 Churn Prediction

POST /predict

Send customer information to receive a churn prediction.

Example request

{
  "tenure": 2,
  "MonthlyCharges": 70.35,
  "TotalCharges": 140.70,
  "SeniorCitizen": 0,
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}

Example response

{
  "churn_probability": 0.6634,
  "risk_category": "MEDIUM",
  "prediction": 1,
  "message": "Churn predicted"
}

Risk categories

Probability >= 0.70  → HIGH
Probability >= 0.40  → MEDIUM
Probability <  0.40  → LOW

The API also validates incoming data through a Pydantic schema before passing it to the model.

🐳 Run with Docker

The API is containerized using Docker.

1. Build the image

From the project root:

docker build -t churn-api .

2. Run the container

docker run -p 8000:8000 churn-api

The API will then be available at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs

Docker architecture

Dockerfile
    │
    ▼
Docker Image
    │
    ▼
Docker Container
    │
    ├── FastAPI
    ├── Python dependencies
    └── Saved churn model
            │
            ▼
       REST API :8000

✅ Verification

The project has been tested through the complete deployment flow:

Model training              ✅
MLflow tracking             ✅
Saved model loading         ✅
FastAPI local prediction   ✅
Docker image build          ✅
Docker container startup   ✅
Docker /health             ✅
Docker /predict            ✅

Example Docker prediction verification:

{
  "churn_probability": 0.0045,
  "risk_category": "LOW",
  "prediction": 0,
  "message": "No churn predicted"
}

🔮 Future Improvements

Potential next steps for the project:

Add automated tests for API endpoints

Add CI/CD with GitHub Actions

Add model performance visualization

Compare multiple classification algorithms

Add hyperparameter tuning

Add a frontend/dashboard for predictions

Improve model monitoring

Deploy the API to a cloud platform

Add automated model retraining

📌 Key Learning Outcomes

This project demonstrates an end-to-end ML workflow rather than only model training:

Data
 ↓
Preprocessing
 ↓
Model Training
 ↓
Evaluation
 ↓
Experiment Tracking
 ↓
Model Persistence
 ↓
REST API
 ↓
Containerization

It brings together machine learning, model tracking, API development, and containerization in a single project.

👨‍💻 Author

Suryansh

If you found this project useful, feel free to ⭐ the repository.