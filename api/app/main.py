"""API d'analyse du sentiment des avis clients — Escale Atlantique (mission FastIA)."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from app.inference import load_pipeline, predict_sentiment  # noqa: F401 (utilisée en tâche 3)
from app.schemas import ReviewIn, SentimentOut

MODEL_NAME = os.getenv("MODEL_NAME", "cmarkea/distilcamembert-base-sentiment")

# Objets partagés par toutes les requêtes (ici : le pipeline de classification)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au 1er lancement, le modèle est téléchargé (≈ 270 Mo) : ça peut prendre quelques minutes
    logger.info("Chargement du modèle {}…", MODEL_NAME)
    state["classifier"] = load_pipeline(MODEL_NAME)
    logger.info("Modèle chargé")
    yield
    state.clear()


app = FastAPI(
    title="Escale Atlantique — Sentiment des avis clients",
    description="Classe un avis client en négatif, neutre ou positif.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.get("/info")
def info() -> dict:
    """Quel modèle tourne, et que renvoie l'API ?"""
    return {
        "model_name": MODEL_NAME,
        "sortie_modele": "1 à 5 étoiles",
        "sortie_api": ["négatif", "neutre", "positif"],
    }


@app.post("/predict", response_model=SentimentOut)
def predict(review: ReviewIn) -> SentimentOut:
    """Analyse le sentiment d'un avis.

    """
    raise HTTPException(status_code=501, detail="Route /predict pas encore implémentée")
