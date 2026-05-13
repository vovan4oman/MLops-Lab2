"""Веб-сервіс для інференсу моделі класифікації вина (Wine dataset).
Варіант 10: Random Forest Classifier.
"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import WineFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

CLASS_NAMES = ["class_0", "class_1", "class_2"]

app = FastAPI(
    title="Wine Classification ML API",
    description="REST API для класифікації вина за хімічними характеристиками (варіант 10)",
    version="1.0.0",
)

# Завантажуємо модель один раз при старті, а не при кожному запиті
model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Wine Classification ML API", "variant": 10}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.array([[
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315,
        features.proline,
    ]])

    class_id = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0, class_id])

    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        probability=round(proba, 4),
    )
