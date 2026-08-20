from fastapi import FastAPI, Body, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from inference import predict, predict_file
from io import BytesIO, StringIO
from typing import Annotated
from schemas import PredictResponse, PredictRequest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import pandas as pd

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
    return PredictResponse(**predict(data.model_dump()))

@app.post("/v1/predict-file")
@limiter.limit("20/minute")
async def create_predict_file(request: Request, file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File should have .csv extension.")

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read the file: {e}")

    results = predict_file(df)
    output = StringIO()
    results.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'}
    )
