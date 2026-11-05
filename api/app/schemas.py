"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.inference import NOTE_MAX, NOTE_MIN, SEUIL_NEGATIF, SEUIL_POSITIF

# Le modèle accepte 512 tokens, soit ~370 mots : 2 000 caractères restent sous la limite
LONGUEUR_MAX = 2000

# Plafond d'un lot : au-delà, un fichier déposé par erreur bloquerait l'API plusieurs minutes
LOT_MAX = 200

Sentiment = Literal["négatif", "neutre", "positif"]

Texte = Annotated[
    str,
    # strip_whitespace : un texte fait de seuls espaces devient vide, donc refusé en 422
    StringConstraints(strip_whitespace=True, min_length=1, max_length=LONGUEUR_MAX),
]


class Seuils(BaseModel):
    """Les deux seuils de la conversion note → classe, réglables par requête."""

    seuil_negatif: Annotated[float, Field(ge=NOTE_MIN, le=NOTE_MAX)] = SEUIL_NEGATIF
    seuil_positif: Annotated[float, Field(ge=NOTE_MIN, le=NOTE_MAX)] = SEUIL_POSITIF

    @model_validator(mode="after")
    def _seuils_ordonnes(self):
        """Des seuils inversés rendraient la classe « neutre » impossible à atteindre."""
        if self.seuil_negatif >= self.seuil_positif:
            raise ValueError("seuil_negatif doit être strictement inférieur à seuil_positif")
        return self


class ReviewIn(Seuils):
    """Un avis client à analyser."""

    text: Texte


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Sentiment
    # La note moyenne pondérée : c'est la valeur sur laquelle les seuils tranchent
    note: float
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float


class AvisIn(BaseModel):
    """Un avis d'un lot, avec sa classe attendue quand le fichier est annoté."""

    texte: Texte
    # Absent d'un fichier de production : sans lui, pas de matrice, juste une répartition
    sentiment_attendu: Sentiment | None = None


class BatchIn(Seuils):
    """Requête de /predict/batch : un lot d'avis et les seuils à leur appliquer."""

    avis: Annotated[list[AvisIn], Field(min_length=1, max_length=LOT_MAX)]


class AvisOut(BaseModel):
    """Un avis analysé. La classe attendue est renvoyée telle quelle."""

    sentiment: Sentiment
    note: float
    # Réémis pour que l'appelant construise sa matrice sans réaligner sur son fichier
    sentiment_attendu: Sentiment | None = None


class BatchOut(BaseModel):
    """Réponse de /predict/batch, dans l'ordre d'entrée."""

    total: int
    duree_ms: float
    # Rappel des seuils appliqués : une matrice sans ses seuils n'est pas interprétable
    seuils: dict[str, float]
    resultats: list[AvisOut]


class LogEntry(BaseModel):
    """Une prédiction journalisée, telle que la renvoie /logs."""

    horodatage: str
    # Pas d'extrait de l'avis : il reste dans logs/api.log, côté serveur
    sentiment: Sentiment
    latence_ms: float


class LogsOut(BaseModel):
    """Réponse de /logs, les prédictions les plus récentes d'abord."""

    total: int
    entrees: list[LogEntry]
