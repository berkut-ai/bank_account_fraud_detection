from pydantic import BaseModel
from typing import Literal

class PredictRequest(BaseModel):
    transaction_amount: float
    location: str
    channel: Literal['branch', 'online', 'atm']
    transaction_type: Literal['debit', 'credit']
    ip: str
    account_id: str


class PredictResponse(BaseModel):
    is_fraud: bool
    fraud_score: float