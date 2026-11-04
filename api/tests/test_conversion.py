"""Conversion des étoiles et troncature des logs : aucun modèle à charger."""

import pytest

from app.inference import SEUIL_NEGATIF, SEUIL_POSITIF, convertir_note, note_attendue
from app.journal import EXTRAIT_MAX, extrait

ETOILES = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")


@pytest.mark.parametrize(
    "note, attendu",
    [
        (1.0, "négatif"),
        (2.79, "négatif"),
        (2.8, "neutre"),
        (3.0, "neutre"),
        (3.59, "neutre"),
        (3.6, "positif"),
        (5.0, "positif"),
    ],
)
def test_convertir_note_aux_bornes(note, attendu):
    assert convertir_note(note) == attendu


def test_les_seuils_penchent_vers_le_negatif():
    """Choix métier : un mécontent raté coûte plus cher qu'un neutre relu pour rien."""
    # Des seuils centrés sur 3 vaudraient 2.5 et 3.5 : les nôtres sont décalés vers le haut
    assert SEUIL_NEGATIF > 2.5
    assert SEUIL_POSITIF > 3.5


def test_note_attendue_sur_une_certitude():
    scores = dict.fromkeys(ETOILES, 0.0) | {"5 stars": 1.0}
    assert note_attendue(scores) == pytest.approx(5.0)


def test_note_attendue_sur_une_distribution_plate():
    assert note_attendue(dict.fromkeys(ETOILES, 0.2)) == pytest.approx(3.0)


def test_note_attendue_lisse_une_hesitation_entre_deux_extremes():
    # Le modèle hésite entre 1 et 5 étoiles : la note moyenne vaut 3, là où l'argmax trancherait
    scores = dict.fromkeys(ETOILES, 0.0) | {"1 star": 0.5, "5 stars": 0.5}
    assert note_attendue(scores) == pytest.approx(3.0)


def test_extrait_laisse_un_avis_court_intact():
    assert extrait("Séjour parfait !") == "Séjour parfait !"


def test_extrait_ne_depasse_jamais_la_limite():
    # Un caractère de plus pour l'ellipse, qui signale la coupure au relecteur
    assert len(extrait("a" * 500)) == EXTRAIT_MAX + 1


def test_extrait_ecrase_les_retours_a_la_ligne():
    """Un avis collé depuis Booking peut contenir des sauts de ligne : une ligne de log, une seule."""
    assert extrait("Chambre sale.\n\nServiettes tachées.") == "Chambre sale. Serviettes tachées."
