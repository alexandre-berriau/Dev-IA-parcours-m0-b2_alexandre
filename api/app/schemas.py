"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

# Le modèle accepte 512 tokens, soit ~370 mots : 2 000 caractères restent sous la limite
LONGUEUR_MAX = 2000


class ReviewIn(BaseModel):
    """Un avis client à analyser."""

    # strip_whitespace : un texte fait de seuls espaces devient vide, donc refusé en 422
    text: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=LONGUEUR_MAX,
        ),
    ]


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Literal["négatif", "neutre", "positif"]
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float
