"""KoBERT Sentiment Analysis API Server."""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import SentimentPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

predictor: SentimentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    predictor = SentimentPredictor(
        model_repo=os.getenv("MODEL_REPO", "HyeonSang/kobert-sentiment"),
    )
    logger.info("Model loaded. Server ready.")
    yield


app = FastAPI(
    title="KoBERT Sentiment Analysis",
    description="한국어 문장 감정 분석 API (긍정/부정)",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/predict")
async def predict(input_data: str = Form(...)):
    if not input_data.strip():
        return JSONResponse({"error": "input_data is empty"}, status_code=400)
    result = predictor.predict(input_data)
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
