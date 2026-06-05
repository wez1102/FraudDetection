from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is healthy"}

def test_predict():
    test_data = {
        "AccountID": "12345",
        "AmountToBalanceRatio": 0.1,
        "TimeSincePreviousTransaction": 500,
        "IsNewLocation": 1,
        "IsNewIP": 1,
        "IsHighLoginAttempt": 1,
        "TransactionHour": 23,
        "TransactionDayOfWeek": 5,
        "IsOnlineTransaction": 1,
        "CustomerAgeGroup": 4
    }
    response = client.post("/predict", json=test_data)
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200
    assert "AccountID" in response.json()
    assert "RiskScore" in response.json()
    assert "RiskLevel" in response.json()