"""Journalisation des prédictions : fichier pour la traçabilité, mémoire pour l'affichage."""

import os
from collections import deque
from pathlib import Path

from loguru import logger

# Un avis peut contenir un nom, un numéro de chambre, une adresse e-mail : jamais en entier
EXTRAIT_MAX = 80

# Nombre de prédictions gardées en mémoire pour /logs ; le fichier, lui, garde tout
TAMPON_MAX = 50

DOSSIER_LOGS = Path(os.getenv("LOG_DIR", "logs"))

# Les plus récentes en tête
_recentes: deque[dict] = deque(maxlen=TAMPON_MAX)


def extrait(text: str) -> str:
    """Tronque un avis pour ne jamais journaliser une donnée personnelle complète."""
    propre = " ".join(text.split())
    return propre if len(propre) <= EXTRAIT_MAX else propre[:EXTRAIT_MAX] + "…"


def _memoriser(message) -> None:
    """Sink Loguru : empile la prédiction dans le tampon lu par /logs."""
    enregistrement = message.record
    _recentes.appendleft(
        {
            "horodatage": enregistrement["time"].isoformat(timespec="seconds"),
            **enregistrement["extra"]["prediction"],
        }
    )


def configurer() -> None:
    """Branche les deux destinations : le fichier du volume et le tampon mémoire."""
    DOSSIER_LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(
        DOSSIER_LOGS / "api.log",
        level="INFO",
        rotation="1 MB",
        retention=3,
        encoding="utf-8",
    )
    # filter : seules les prédictions entrent dans le tampon, pas les messages de démarrage
    logger.add(_memoriser, filter=lambda r: "prediction" in r["extra"])


def tracer(text: str, sentiment: str, latence_ms: float) -> None:
    """Une ligne par appel à /predict : le fichier garde l'extrait, /logs non."""
    # Seules ces clés remontent dans /logs, via extra["prediction"]
    donnees = {"sentiment": sentiment, "latence_ms": latence_ms}
    # L'extrait n'est qu'un argument de formatage : il ne peut pas atteindre le tampon
    logger.bind(prediction=donnees).info(
        "predict | {sentiment} | {latence_ms} ms | {extrait}",
        extrait=extrait(text),
        **donnees,
    )


def dernieres(limite: int) -> list[dict]:
    """Les prédictions les plus récentes d'abord."""
    return list(_recentes)[:limite]
