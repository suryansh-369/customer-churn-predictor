#Block 1: IMORTS FOR TRAINING#
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

# Always save to this database
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# Block 2: Load Data
def load_data():
    df = pd.read_csv("data/raw/churn.csv")
    
    # Drop customer ID — it's just a label, not useful for prediction
    df = df.drop(columns=['customerID'])
    
    # Convert target column from Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Fix TotalCharges column — it has some empty spaces that
    # prevent Python from treating it as a number
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remove the ~11 rows where TotalCharges couldn't be converted
    df = df.dropna(subset=['TotalCharges'])
    
    return df

# Block 3: Define Features
NUMERIC_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'
]

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Block 4: Build Preprocessor
def build_preprocessor():
    
    # For numeric columns → fill missing values then scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # For categorical columns → fill missing values then encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Combine both transformers
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor 

# Block 5: Train Function with MLflow
def train():

    # --- Load and split data ---
    df = load_data()
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --- Define hyperparameters ---
    # We put them in a dictionary so MLflow can log them easily
    params = {
    "C": 1.0,
    "max_iter": 1000
    }

    # --- Tell MLflow which experiment this belongs to ---
    mlflow.set_experiment("churn-prediction")

    # --- Start recording this run ---
    with mlflow.start_run(run_name="logistic-regression-v1"):

        # Build pipeline
        preprocessor = build_preprocessor()
        model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(**params))
])

        # Train
        model.fit(X_train, y_train)

        # Cross validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='roc_auc'
        )

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_auc = roc_auc_score(y_test, y_prob)
        test_f1  = f1_score(y_test, y_pred)
        cv_auc   = cv_scores.mean()
        cv_std   = cv_scores.std()

        # ----------------------------------------
        # MLFLOW LOGGING — the 3 key commands
        # ----------------------------------------

        # 1. Save the settings you used
        mlflow.log_params(params)

        # 2. Save the performance numbers
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1",  test_f1)
        mlflow.log_metric("cv_auc",   cv_auc)
        mlflow.log_metric("cv_std",   cv_std)

        # 3. Save the actual model file
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="ChurnPredictor"
        )

        # ----------------------------------------

        # Also save locally for FastAPI later
        joblib.dump(model, "models/churn_model.pkl")

        # Print results to terminal
        print(f"\n{'='*45}")
        print(f"  CV  AUC : {cv_auc:.4f} (+/- {cv_std:.4f})")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Test F1 : {test_f1:.4f}")
        print(f"{'='*45}\n")
        print(classification_report(
            y_test, y_pred,
            target_names=['No Churn', 'Churn']
        ))


# Block 6: Run the script
if __name__ == "__main__":
    train()