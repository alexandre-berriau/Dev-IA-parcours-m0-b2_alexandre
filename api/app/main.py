"""API d'analyse du sentiment des avis clients — Escale Atlantique (mission FastIA)."""

import os
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from loguru import logger

from app import journal
from app.inference import (
    SEUIL_NEGATIF,
    SEUIL_POSITIF,
    load_pipeline,
    predict_batch,
    predict_sentiment,
)
from app.schemas import (
    LOT_MAX,
    BatchIn,
    BatchOut,
    LogsOut,
    ReviewIn,
    SentimentOut,
)

MODEL_NAME = os.getenv("MODEL_NAME", "cmarkea/distilcamembert-base-sentiment")

# Objets partagés par toutes les requêtes (ici : le pipeline de classification)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    journal.configurer()
    # Au 1er lancement, le modèle est téléchargé (≈ 270 Mo) : ça peut prendre quelques minutes
    logger.info("Chargement du modèle {}…", MODEL_NAME)
    state["classifier"] = load_pipeline(MODEL_NAME)
    logger.info("Modèle chargé")
    yield
    state.clear()


app = FastAPI(
    title="Escale Atlantique — Sentiment des avis clients",
    description="Classe un avis client en négatif, neutre ou positif.",
    version="0.2.0",
    lifespan=lifespan,
)


def _classifier():
    """Le pipeline, ou 503 tant que le modèle n'est pas chargé."""
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return classifier


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.get("/info")
def info() -> dict:
    """Quel modèle tourne, que renvoie l'API, et avec quels seuils par défaut ?"""
    return {
        "model_name": MODEL_NAME,
        "sortie_modele": "1 à 5 étoiles",
        "sortie_api": ["négatif", "neutre", "positif"],
        # L'interface initialise ses curseurs là-dessus : une seule source pour les valeurs
        "seuils_defaut": {"seuil_negatif": SEUIL_NEGATIF, "seuil_positif": SEUIL_POSITIF},
        "lot_max": LOT_MAX,
    }


@app.post("/predict", response_model=SentimentOut)
def predict(review: ReviewIn) -> SentimentOut:
    """Analyse le sentiment d'un avis client."""
    resultat = predict_sentiment(
        _classifier(), review.text, review.seuil_negatif, review.seuil_positif
    )
    journal.tracer(review.text, resultat["sentiment"], resultat["latence_ms"])
    return SentimentOut(**resultat)


@app.post("/predict/batch", response_model=BatchOut)
def predict_lot(lot: BatchIn) -> BatchOut:
    """Analyse un lot d'avis, dans l'ordre d'entrée, avec les seuils fournis."""
    resultats, duree_ms = predict_batch(
        _classifier(),
        [avis.texte for avis in lot.avis],
        lot.seuil_negatif,
        lot.seuil_positif,
    )
    # La classe attendue ne sert pas à l'inférence : on la recolle pour l'appelant
    for resultat, avis in zip(resultats, lot.avis):
        resultat["sentiment_attendu"] = avis.sentiment_attendu

    journal.tracer_lot(len(resultats), duree_ms, Counter(r["sentiment"] for r in resultats))

    return BatchOut(
        total=len(resultats),
        duree_ms=duree_ms,
        seuils={"seuil_negatif": lot.seuil_negatif, "seuil_positif": lot.seuil_positif},
        resultats=resultats,
    )


@app.get("/logs", response_model=LogsOut)
def logs(limite: int = Query(20, ge=1, le=journal.TAMPON_MAX)) -> LogsOut:
    """Les dernières prédictions, pour affichage dans l'interface qualité."""
    entrees = journal.dernieres(limite)
    return LogsOut(total=len(entrees), entrees=entrees)
