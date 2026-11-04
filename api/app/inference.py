"""Chargement du modèle Hugging Face et inférence."""

import time

from transformers import pipeline

# Ordre figé des labels : le pipeline renvoie la liste triée par score, pas par étoile
ETOILES = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")

# Seuils de la conversion note → classe métier, volontairement asymétriques
SEUIL_NEGATIF = 2.8
SEUIL_POSITIF = 3.6


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache)."""
    return pipeline("text-classification", model=model_name, top_k=None)


def note_attendue(scores_etoiles: dict[str, float]) -> float:
    """Moyenne des 5 étoiles pondérée par leurs probabilités."""
    return sum((i + 1) * scores_etoiles[etoile] for i, etoile in enumerate(ETOILES))


def convertir_note(note: float) -> str:
    """Convertit une note de 1 à 5 étoiles en classe métier.

    La note attendue utilise toute la distribution, là où l'étoile la plus probable
    jette l'information : sur un avis à 3★ 0.44 et 4★ 0.46, l'argmax tranche sur
    0.02 d'écart sans laisser aucun réglage.

    Les seuils ne sont pas symétriques autour de 3, et c'est un choix métier : un
    mécontent classé « neutre » reste sans réponse, un neutre classé « négatif »
    coûte deux minutes de lecture. Le seuil bas passe donc de 2.5 à 2.8.
    Justification chiffrée dans le README.
    """
    if note < SEUIL_NEGATIF:
        return "négatif"
    if note < SEUIL_POSITIF:
        return "neutre"
    return "positif"


def predict_sentiment(classifier, text: str) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis."""
    debut = time.perf_counter()
    # truncation : au-delà de 512 tokens le modèle lève une RuntimeError
    brut = classifier(text, truncation=True)
    latence_ms = (time.perf_counter() - debut) * 1000

    # Le pipeline renvoie une liste par texte soumis : on n'en passe qu'un
    scores_etoiles = {d["label"]: round(d["score"], 4) for d in brut[0]}

    return {
        "sentiment": convertir_note(note_attendue(scores_etoiles)),
        # Remis dans l'ordre des étoiles, plus lisible pour la traçabilité
        "scores_etoiles": {etoile: scores_etoiles[etoile] for etoile in ETOILES},
        "latence_ms": round(latence_ms, 1),
    }
