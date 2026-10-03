"""
FastAPI backend for the spam detection model — Vercel entry point.

Routes:
  GET  /              -> serves the UI (HTML response)
  GET  /health        -> health check JSON
  POST /predict       -> classify a single message
  POST /predict-batch -> classify multiple messages at once
  GET  /docs          -> Swagger UI
"""

import re
import string
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List

# ---------------------------------------------------------------
# Paths — resolved relative to THIS file so they work on Vercel.
# Repo layout:
#   /api/index.py          <- this file
#   /spam_model.joblib     <- one level up
#   /vectorizer.joblib     <- one level up
#   /frontend/ui/index.html
# ---------------------------------------------------------------
_ROOT = Path(__file__).parent.parent          # repo root
MODEL_PATH      = _ROOT / "spam_model.joblib"
VECTORIZER_PATH = _ROOT / "vectorizer.joblib"
HTML_PATH       = _ROOT / "frontend" / "ui" / "index.html"

# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
app = FastAPI(title="Spam Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Load model + vectorizer once at startup
# ---------------------------------------------------------------
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Could not load model/vectorizer. "
        f"Expected at: {MODEL_PATH} and {VECTORIZER_PATH}. "
        f"Original error: {e}"
    )

# ---------------------------------------------------------------
# Text cleaning (must match training)
# ---------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Text message to classify")

class BatchMessageRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, description="List of messages to classify")

class PredictionResponse(BaseModel):
    message: str
    prediction: str
    spam_probability: float

# ---------------------------------------------------------------
# Core classify helper
# ---------------------------------------------------------------
def classify(message: str) -> PredictionResponse:
    cleaned = clean_text(message)
    vector  = vectorizer.transform([cleaned])
    pred    = model.predict(vector)[0]
    prob    = model.predict_proba(vector)[0][1]
    return PredictionResponse(
        message=message,
        prediction="SPAM" if pred == 1 else "HAM",
        spam_probability=round(float(prob), 4),
    )

# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serve the frontend HTML directly — works on Vercel serverless."""
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Spam Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return classify(request.message)


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchMessageRequest):
    return [classify(m) for m in request.messages if m.strip()]
