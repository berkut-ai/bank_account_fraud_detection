# Bank Account Fraud Detection API

A REST API for assessing fraud risk in banking transactions. The service accepts either a single transaction or a CSV file, builds model features, and returns an `IsolationForest` prediction.

## Features

- Single-transaction scoring via `POST /v1/predict`.
- Batch CSV scoring via `POST /v1/predict-file`.
- Returns a fraud flag and a numeric fraud score.
- Uses Redis to track transactions per IP address and IP addresses per account.
- Checks whether an IP uses a proxy through `ip-api.com` and caches the result in process memory.
- Rate-limits requests with SlowAPI.
- Writes rotating logs to stdout and `logs/app.log`.

## Project structure

```text
.
├── main.py             # FastAPI application and HTTP endpoints
├── inference.py        # feature preparation and model inference
├── schemas.py          # Pydantic API schemas
├── logging_config.py   # logging configuration loader
├── logging.json        # handlers, format, and log rotation settings
├── ml/                 # trained model and related artifacts
└── requirements.txt
```

## Quick start

Create a virtual environment, install dependencies, and start Redis:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
redis-server
```

Start the API in another terminal:

```bash
uvicorn main:app --reload
```

Once running, the following endpoints are available:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Health check: `http://127.0.0.1:8000/`

> Uvicorn reloads the application whenever a watched file changes. If `change detected` is logged continuously, a generated file such as `logs/app.log` is probably being watched. Exclude the logs directory when starting the server: `uvicorn main:app --reload --reload-exclude 'logs/*'`.

## API

### Health check

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{"status":"ok"}
```

### Score one transaction

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_amount": 1450.75,
    "location": "New York",
    "channel": "online",
    "transaction_type": "debit",
    "ip": "8.8.8.8",
    "account_id": "acc_123456"
  }'
```

Request fields:

| Field | Type | Allowed values / description |
| --- | --- | --- |
| `transaction_amount` | number | Transaction amount |
| `location` | string | Transaction location |
| `channel` | string | `branch`, `online`, or `atm` |
| `transaction_type` | string | `debit` or `credit` |
| `ip` | string | Client IP address |
| `account_id` | string | Account identifier |

Example response:

```json
{
  "is_fraud": false,
  "fraud_score": 0.12
}
```

`is_fraud` is `true` when Isolation Forest classifies the transaction as anomalous. `fraud_score` is the model's relative internal score, not a fraud probability.

### Score a CSV file

The CSV file must include headers matching the single-transaction request fields. Minimal example:

```csv
transaction_amount,location,channel,transaction_type,ip,account_id
1450.75,New York,online,debit,8.8.8.8,acc_123456
50.00,London,atm,credit,1.1.1.1,acc_987654
```

Send the file:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict-file \
  -F 'file=@transactions.csv' \
  -o predictions.csv
```

The response is a CSV containing the original columns plus `is_fraud` and `fraud_score`.

## Request limits

Limits are applied per client IP address:

| Endpoint | Limit |
| --- | --- |
| `GET /` | 30 requests per minute |
| `POST /v1/predict` | 30 requests per minute |
| `POST /v1/predict-file` | 20 requests per minute |

## Logging

Logs are written to the console and to `logs/app.log`. The application uses `RotatingFileHandler`: each file is limited to 5 MiB and up to three backup files are retained.

