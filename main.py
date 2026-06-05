from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI(title="Financial Fraud Detection API")

model = joblib.load("Models/isolation_forest_model.pkl")
scalaler = joblib.load("Models/scaler.pkl")
imputer = joblib.load("Models/imputer.pkl")
test_maxscore = joblib.load("Models/maxscore.pkl")
test_minscore = joblib.load("Models/minscore.pkl")

class Transactions(BaseModel):
    AccountID: str
    AmountToBalanceRatio: float
    TimeSincePreviousTransaction: float
    IsNewLocation: int
    IsNewIP: int
    IsHighLoginAttempt: int
    TransactionHour: int
    TransactionDayOfWeek: int
    IsOnlineTransaction: int
    CustomerAgeGroup: int

@app.get("/health")
def health_check():
    return {"status": "API is healthy"}

@app.post("/predict")
def predict(transactions: Transactions):
    data = pd.DataFrame([transactions.dict()])
    
    data_imputed = imputer.transform(data.drop(columns=["AccountID"]))
    data_scaled = scalaler.transform(data_imputed)

    anomaly_score = model.decision_function(data_scaled)

    risk_score = 100 * (
    (test_maxscore - anomaly_score) /
    (test_maxscore - test_minscore)
    )

    logger.debug(f"Anomaly Score: {anomaly_score}, test_minscore: {test_minscore}, test_maxscore: {test_maxscore}, Risk Score: {risk_score}")

    if risk_score[0] >= 60:
        risk_level = "High Risk"
    elif risk_score[0] >= 30:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    return {"AccountID": transactions.AccountID, "RiskScore": risk_score[0], "RiskLevel": risk_level}