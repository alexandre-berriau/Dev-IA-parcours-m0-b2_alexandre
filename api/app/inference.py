"""Chargement du modèle Hugging Face et inférence."""

import time

from transformers import pipeline

# Ordre figé des labels : le pipeline renvoie la liste triée par score, pas par étoile
ETOILES = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")

# Bornes de la note : le modèle note de 1 à 5 étoiles, un seuil hors de là n'a pas de sens
NOTE_MIN = 1.0
NOTE_MAX = 5.0

# Seuils par défaut de la conversion note → classe métier, volontairement asymétriques
SEUIL_NEGATIF = 2.8
SEUIL_POSITIF = 3.6


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache)."""
    return pipeline("text-classification", model=model_name, top_k=None)


def note_attendue(scores_etoiles: dict[str, float]) -> float:
    """Moyenne des 5 étoiles pondérée par leurs probabilités."""
    return sum((i + 1) * scores_etoiles[etoile] for i, etoile in enumerate(ETOILES))


def convertir_note(
    note: float,
    seuil_negatif: float = SEUIL_NEGATIF,
    seuil_positif: float = SEUIL_POSITIF,
) -> str:
    """Convertit une note de 1 à 5 étoiles en classe métier.

    La note attendue utilise toute la distribution, là où l'étoile la plus probable
    jette l'information : sur un avis à 3★ 0.44 et 4★ 0.46, l'argmax tranche sur
    0.02 d'écart sans laisser aucun réglage.

    Les seuils ne sont pas symétriques autour de 3, et c'est un choix métier : un
    mécontent classé « neutre » reste sans réponse, un neutre classé « négatif »
    coûte deux minutes de lecture. Le seuil bas passe donc de 2.5 à 2.8.
    Justification chiffrée dans le README.

    Ils sont réglables par requête pour que l'interface qualité puisse les déplacer
    et observer la matrice de confusion sans redéployer l'API.
    """
    if note < seuil_negatif:
        return "négatif"
    if note < seuil_positif:
        return "neutre"
    return "positif"


def _note_du_brut(scores_bruts: list[dict]) -> tuple[dict[str, float], float]:
    """Remet les scores dans l'ordre des étoiles et en déduit la note."""
    scores = {d["label"]: round(d["score"], 4) for d in scores_bruts}
    return {etoile: scores[etoile] for etoile in ETOILES}, note_attendue(scores)


def predict_sentiment(
    classifier,
    text: str,
    seuil_negatif: float = SEUIL_NEGATIF,
    seuil_positif: float = SEUIL_POSITIF,
) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis."""
    debut = time.perf_counter()
    # truncation : au-delà de 512 tokens le modèle lève une RuntimeError
    brut = classifier(text, truncation=True)
    latence_ms = (time.perf_counter() - debut) * 1000

    # Le pipeline renvoie une liste par texte soumis : on n'en passe qu'un
    scores_etoiles, note = _note_du_brut(brut[0])

    return {
        "sentiment": convertir_note(note, seuil_negatif, seuil_positif),
        "note": round(note, 3),
        # Remis dans l'ordre des étoiles, plus lisible pour la traçabilité
        "scores_etoiles": scores_etoiles,
        "latence_ms": round(latence_ms, 1),
    }


def predict_batch(
    classifier,
    textes: list[str],
    seuil_negatif: float = SEUIL_NEGATIF,
    seuil_positif: float = SEUIL_POSITIF,
) -> tuple[list[dict], float]:
    """Analyse une liste d'avis en un seul passage et renvoie (résultats, durée totale).

    La liste part entière au pipeline plutôt qu'avis par avis : il regroupe les textes
    en lots et paie une seule fois le passage Python → torch.
    """
    debut = time.perf_counter()
    brut = classifier(textes, truncation=True)
    duree_ms = (time.perf_counter() - debut) * 1000

    resultats = []
    for scores_bruts in brut:
        _, note = _note_du_brut(scores_bruts)
        resultats.append(
            {
                "sentiment": convertir_note(note, seuil_negatif, seuil_positif),
                "note": round(note, 3),
            }
        )
    # L'ordre d'entrée est préservé : l'appelant peut réaligner sur ses propres lignes
    return resultats, round(duree_ms, 1)
