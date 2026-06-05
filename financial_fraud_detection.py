
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

try:
    from google.colab import drive
    drive.mount('/content/drive')
    path = "/content/drive/MyDrive/Bank_Fraud_Detection_Project/bank_transactions_data_2.csv"
    print("Google Drive mounted successfully.")
except ModuleNotFoundError:
    path = "bank_transactions_data_2.csv"
    print("Google Drive not available. Using local file path.")

df = pd.read_csv(path)
new_col = [(16,"AmountToBalanceRatio",np.nan),(17,"TimeSincePreviousTransaction",np.nan),(18,"IsNewLocation",np.nan),(19,"IsNewIP",np.nan),(20,"IsHighLoginAttempt",np.nan),(21,"TransactionHour",np.nan),(22,"TransactionDayOfWeek",np.nan),(23,"IsOnlineTransaction",np.nan),(24,"CustomerAgeGroup",np.nan)]
for i in new_col:
  df.insert(i[0],i[1],i[2],allow_duplicates=True)
df.head()

df = df.sort_values(by=["AccountID","TransactionDate"],ascending=True)
df.insert(25,"PreviousLocation",df.groupby("AccountID")["Location"].shift(1))
df.insert(26,"LastIP",df.groupby("AccountID")["IP Address"].shift(1))
df.head()

#AmountToBalanceRatio
df["AmountToBalanceRatio"] = df["TransactionAmount"] / df["AccountBalance"]
#TimeSincePreviousTransaction
df["TimeSincePreviousTransaction"] = pd.to_datetime(df["PreviousTransactionDate"]) - pd.to_datetime(df["TransactionDate"])#pd.to_datetime(df["TransactionDate"]) - pd.to_datetime(df["PreviousTransactionDate"])
#IsNewLocation
df["IsNewLocation"] = np.where(df["Location"] == df["PreviousLocation"],0,1)
#IsNewIP
df["IsNewIP"] = np.where(df["IP Address"] == df["LastIP"],0,1)
#IsHighLoginAttempt
df["IsHighLoginAttempt"] = np.where(df["LoginAttempts"] > 3,1,0)

#TransactionHour
df["TransactionHour"] = pd.to_datetime(df["TransactionDate"]).dt.hour
#TransactionDayOfWeek
df["TransactionDayOfWeek"] = pd.to_datetime(df["TransactionDate"]).dt.dayofweek
#IsOnlineTransaction
df["IsOnlineTransaction"] = np.where(df["Channel"] == "Online", 1 , 0)
#CustomerAgeGroup
conditions = [
    (df["CustomerAge"] >= 18) & (df["CustomerAge"] <= 25),
    (df["CustomerAge"] >= 26) & (df["CustomerAge"] <= 35),
    (df["CustomerAge"] >= 36) & (df["CustomerAge"] <= 50),
    (df["CustomerAge"] >= 51) & (df["CustomerAge"] <= 65),
    (df["CustomerAge"] >= 66)
]
choices = [0, 1, 2, 3, 4]
df["CustomerAgeGroup"] = np.select(conditions, choices, default=np.nan)

df.head()

risk_features = [
    "AmountToBalanceRatio",
    "TimeSincePreviousTransaction",
    "IsNewLocation",
    "IsNewIP",
    "IsHighLoginAttempt",
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsOnlineTransaction",
    "CustomerAgeGroup"
]


X_with_account_id = df[['AccountID'] + risk_features].copy()

if pd.api.types.is_timedelta64_dtype(X_with_account_id["TimeSincePreviousTransaction"]):
    X_with_account_id["TimeSincePreviousTransaction"] = (X_with_account_id["TimeSincePreviousTransaction"].dt.total_seconds() / 3600)

X_with_account_id[risk_features] = X_with_account_id[risk_features].fillna(0)

X_train_full, X_test_full = train_test_split(
    X_with_account_id,
    test_size=0.2,
    random_state=42
)

X_train_account_id = X_train_full['AccountID']
X_test_account_id = X_test_full['AccountID']

X_train_features = X_train_full[risk_features]
X_test_features = X_test_full[risk_features]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train_features)
X_test_imputed = imputer.transform(X_test_features)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

IsolationForest_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

IsolationForest_model.fit(X_train_scaled)

test_anomaly_score = IsolationForest_model.decision_function(X_test_scaled)

test_risk_score = 100 * (
    (test_anomaly_score - test_anomaly_score.min()) /
    (test_anomaly_score.max() - test_anomaly_score.min()))

X_test_result = X_test_features.copy()
X_test_result["AccountID"] = X_test_account_id
X_test_result["RiskScore"] = test_risk_score

X_test_result["RiskLevel"] = pd.cut(
    X_test_result["RiskScore"],
    bins=[-1, 30, 60, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

X_test_result = X_test_result[['AccountID'] + risk_features + ['RiskScore', 'RiskLevel']]

print(X_test_result.head())

joblib.dump(IsolationForest_model, "isolation_forest_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(test_anomaly_score.max(), "maxscore.pkl")
joblib.dump(test_anomaly_score.min(), "minscore.pkl")