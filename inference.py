from datetime import date
from functools import lru_cache
from schemas import PredictRequest, PredictResponse
from pathlib import Path
import joblib
import pandas as pd
import redis
import requests
import time

BASE_DIR = Path(__file__).parent

feature_names = joblib.load(BASE_DIR / "ml" / "feature_names.pkl")
location_freq_map = joblib.load(BASE_DIR / "ml" / "location_freq_map.pkl")
model = joblib.load(BASE_DIR / "ml" / "isolation_forest.pkl")
preprocessor = joblib.load(BASE_DIR / "ml" / "preprocessor.pkl")

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def check_proxy(ip: str) -> int:
    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,proxy,hosting,country,isp"
        response = requests.get(url, timeout=3)
        data = response.json()
    except (requests.RequestException, ValueError):
        return 0

    if data.get("status") != "success":
        return 0
    time.sleep(1)

    return int(data.get("proxy", False))

@lru_cache()
def check_proxy_cached(ip: str) -> int:
    return check_proxy(ip)

def get_ip_transactions_per_day(ip: 'str') -> int:
    key = f"ip_count:{ip}:{date.today()}"
    count = r.incr(key)
    r.expire(key, 86400)
    return count
def get_ips_per_account(account_id: str, ip: str) -> int:
    key = f"account_ips:{account_id}"
    r.sadd(key, ip)
    return r.scard(key)

def preprocess_data(request: PredictRequest) -> pd.DataFrame:
    data = request.model_dump()
    data['is_proxy'] = check_proxy_cached(data['ip'])
    data['location_freq'] = location_freq_map.get(data['location'], 0)
    data['Channel_Branch'] = int(data['channel'] == "branch")
    data['Channel_Online'] = int(data['channel'] == "online")
    data['TransactionType_Debit'] = int(data['transaction_type'] == 'debit')
    data['ip_transactions_per_day'] = get_ip_transactions_per_day(data['ip'])
    data['ips_per_acc'] = get_ips_per_account(data['account_id'], data['ip'])
    data['TransactionAmount'] = data['transaction_amount']

    df = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)
    return pd.DataFrame(preprocessor.transform(df), columns=feature_names)

def predict(request: PredictRequest) -> PredictResponse:
    data = preprocess_data(request)
    prediction = model.predict(data)[0]
    score = -model.decision_function(data)[0]
    return PredictResponse(is_fraud=bool(prediction == -1), fraud_score=round(float(score), 2))


if __name__ == '__main__':
    print(feature_names)
    print(predict(PredictRequest(**{
        "transaction_amount": 1450.75,
        "location": "New York",
        "channel": "online",
        "transaction_type": "debit",
        "ip": "8.8.8.8",
        "account_id": "acc_123456"
      }
    )))
