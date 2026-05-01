import pandas as pd
df=pd.read_csv('data.csv')
df.head()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['tenure_group'] = pd.qcut(
    df['tenure'],
    q=4,
    labels=['Q1','Q2','Q3','Q4']
)
print(df.groupby('tenure_group')['tenure'].agg(['min', 'max']))
service_cols = [
    'PhoneService',
    'OnlineSecurity',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

df['service_count'] = (
    df[service_cols]
    .replace({
        'Yes':1,
        'No':0,
        'No internet service':0,
        'No phone service':0
    })
    .infer_objects(copy=False)  
    .sum(axis=1)
)
#df['charge_group'] = pd.qcut(df['MonthlyCharges'], q=4)
pd.set_option('future.no_silent_downcasting', True)
x=df.drop(['customerID','Churn'],axis=1)
y = df['Churn'].map({'No':0, 'Yes':1})

#seprating column features 
numerical_feat=['tenure','MonthlyCharges','TotalCharges']
categorical_feat=['tenure_group','gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
                  'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                  'StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

#creating a pipeline
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

preproessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_feat),
        ('cat', OneHotEncoder(), categorical_feat)
    ]
)
transformed_data = preproessor.fit_transform(df)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from pyexpat import model

from sklearn.linear_model import LogisticRegression
model_pipeline = Pipeline([
    ('preproessor', preproessor),
    ('model', 
     LogisticRegression(
    C=0.5,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
))
])
model_pipeline.fit(x_train,y_train)
y_pred=model_pipeline.predict(x_test)
y_prob = model_pipeline.predict_proba(x_test)[:,1]

threshold = 0.35

y_pred_custom = (y_prob >= threshold).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_custom))     
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred_custom))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_custom))

from joblib import dump
dump(model_pipeline,'churn_model.joblib')
print("Model saved successfully!")