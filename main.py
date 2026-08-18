from fastapi import FastAPI, Body, Request
from inference import predict
from typing import Annotated
from schemas import PredictResponse, PredictRequest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {
        "status": "ok"
    }

@app.post("/v1/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
async def create_predict(request: Request, data: Annotated[PredictRequest, Body()]):
    return predict(data)