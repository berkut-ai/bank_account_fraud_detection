import logging
from logging_config import setup_logging

setup_logging()

from fastapi import FastAPI, Body, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from inference import predict, predict_file
from io import BytesIO, StringIO
from time import perf_counter
from typing import Annotated
from schemas import PredictResponse, PredictRequest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import pandas as pd

logger = logging.getLogger(__name__)
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
    logger.info(f"Accepted request with transaction.")
    try:
        start = perf_counter()
        prediction = predict(data.model_dump())
        end = perf_counter()
    except Exception as err:
        logger.error("Failed to predict.", exc_info=err)
        raise HTTPException(status_code=500, detail="Failed to predict.")
    else:
        logger.info(f"Request processed successfully for {round(end-start, 3)} sec. Results: {prediction}")
        return PredictResponse(**prediction)

@app.post("/v1/predict-file")
@limiter.limit("20/minute")
async def create_predict_file(request: Request, file: UploadFile):
    logger.info(f"Accepted request with file: {file.filename}, size: {file.size if file.size else 0}.")
    if not file.filename.endswith(".csv"):
        logger.error(f"File with .{file.filename.split('.')[-1]} extension (not csv).")
        raise HTTPException(status_code=400, detail="File should have .csv extension.")

    content = await file.read()

    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as err:
        logger.error(f"Failed to read the file.", exc_info=err)
        raise HTTPException(status_code=400, detail=f"Failed to read the file: {err}")

    try:
        start = perf_counter()
        results = predict_file(df)
        end = perf_counter()
        output = StringIO()
        results.to_csv(output, index=False)
        output.seek(0)
    except Exception as err:
        logger.error("Failed to predict for file.", exc_info=err)
        raise HTTPException(status_code=500, detail="Failed to predict.")
    else:
        logger.info(f"File request processed successfully for {round(end-start, 3)} sec.")
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="predictions.csv"'}
        )
