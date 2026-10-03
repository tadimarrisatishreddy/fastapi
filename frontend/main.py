"""
FastAPI backend for the spam detection model.

Loads the saved model + vectorizer once at startup, then exposes:
  GET  /              -> health check
  POST /predict        -> classify a single message
  POST /predict-batch  -> classify multiple messages at once

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI to test it.
"""

import re
import string
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List

# ---------------------------------------------------------------------
# CONFIG — paths are relative to this file so they work on Vercel too
# ---------------------------------------------------------------------
_HERE = Path(__file__).parent
MODEL_PATH = _HERE.parent / "spam_model.joblib"
VECTORIZER_PATH = _HERE.parent / "vectorizer.joblib"

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="Spam Detection API", version="1.0")

# Allow your frontend (running on a different port) to call this API.
# For local dev, "*" is fine. Lock this down to your actual frontend
# origin (e.g. "http://localhost:5173") before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static UI from frontend/ui/
_UI_DIR = _HERE / "ui"
app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def serve_ui():
    """Redirect root to the frontend UI."""
    return FileResponse(_UI_DIR / "index.html")

# ---------------------------------------------------------------------
# Load model + vectorizer once at startup (not per-request - that would
# be slow and wasteful)
# ---------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Could not load model/vectorizer files. Make sure "
        f"'{MODEL_PATH}' and '{VECTORIZER_PATH}' are in the same folder "
        f"as this script (or update the paths above). Original error: {e}"
    )


# ---------------------------------------------------------------------
# Same cleaning function used during training - MUST match exactly,
# otherwise predictions will be inconsistent with what the model learned.
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The text message to classify")


class BatchMessageRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1, description="List of messages to classify")


class PredictionResponse(BaseModel):
    message: str
    prediction: str
    spam_probability: float


# ---------------------------------------------------------------------
# Core prediction logic, reused by both endpoints
# ---------------------------------------------------------------------
def classify(message: str) -> PredictionResponse:
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]  # probability of spam
    return PredictionResponse(
        message=message,
        prediction="SPAM" if pred == 1 else "HAM",
        spam_probability=round(float(prob), 4),
    )


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
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