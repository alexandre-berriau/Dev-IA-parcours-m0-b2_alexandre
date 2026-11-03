"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Literal

from pydantic import BaseModel


class ReviewIn(BaseModel):
    """Un avis client à analyser.
   
    """

    text: str


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Literal["négatif", "neutre", "positif"]
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float
