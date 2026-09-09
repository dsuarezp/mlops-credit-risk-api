"""FastAPI service that serves credit risk predictions."""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model + column order once, at startup (not on every request)
bundle = joblib.load("model.pkl")
model = bundle["model"]
columns = bundle["columns"]

app = FastAPI(title="Credit Risk API")


class PredictionRequest(BaseModel):
    features: dict  # {column_name: value}


@app.get("/")
def health():
    """Health check — used by Docker/AWS to verify the service is alive."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    # Rebuild the row in the exact column order used during training
    row = pd.DataFrame([request.features]).reindex(columns=columns)
    proba = float(model.predict_proba(row)[0][1])
    return {"probability_good_payer": round(proba, 4)}