"""Route /predict/batch : contrat, ordre de sortie, seuils, lots refusés."""

import pytest

from app.journal import TAMPON_MAX
from app.schemas import LOT_MAX

POSITIF = "Séjour parfait ! Piscine magnifique et personnel aux petits soins."
NEGATIF = "Chambre sale à notre arrivée, cheveux dans la douche. Inadmissible."


def test_batch_respecte_son_contrat(client):
    reponse = client.post("/predict/batch", json={"avis": [{"texte": POSITIF}]})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["total"] == 1
    assert corps["duree_ms"] > 0
    assert corps["seuils"] == {"seuil_negatif": 2.8, "seuil_positif": 3.6}
    assert corps["resultats"][0]["sentiment"] == "positif"
    assert 1.0 <= corps["resultats"][0]["note"] <= 5.0


def test_le_lot_sort_dans_l_ordre_d_entree(client):
    """L'appelant réaligne la matrice sur son fichier : un lot réordonné la fausserait."""
    avis = [{"texte": NEGATIF}, {"texte": POSITIF}, {"texte": NEGATIF}]
    reponse = client.post("/predict/batch", json={"avis": avis})
    sentiments = [r["sentiment"] for r in reponse.json()["resultats"]]
    assert sentiments == ["négatif", "positif", "négatif"]


def test_la_classe_attendue_est_renvoyee_telle_quelle(client):
    avis = [
        {"texte": POSITIF, "sentiment_attendu": "positif"},
        {"texte": NEGATIF, "sentiment_attendu": "neutre"},
        {"texte": POSITIF},
    ]
    reponse = client.post("/predict/batch", json={"avis": avis})
    resultats = reponse.json()["resultats"]
    # La classe attendue ne doit jamais influencer la prédiction, seulement l'accompagner
    assert [r["sentiment_attendu"] for r in resultats] == ["positif", "neutre", None]
    assert resultats[1]["sentiment"] == "négatif"


def test_des_seuils_ecartes_rendent_tout_neutre(client):
    """Le réglage des seuils arrive par la requête : c'est ce qui rend les curseurs utiles."""
    charge = {
        "avis": [{"texte": POSITIF}, {"texte": NEGATIF}],
        "seuil_negatif": 1.01,
        "seuil_positif": 4.99,
    }
    reponse = client.post("/predict/batch", json=charge)
    assert [r["sentiment"] for r in reponse.json()["resultats"]] == ["neutre", "neutre"]
    assert reponse.json()["seuils"] == {"seuil_negatif": 1.01, "seuil_positif": 4.99}


def test_les_seuils_valent_aussi_pour_un_avis_unitaire(client):
    charge = {"text": POSITIF, "seuil_negatif": 1.01, "seuil_positif": 4.99}
    assert client.post("/predict", json=charge).json()["sentiment"] == "neutre"


@pytest.mark.parametrize(
    "charge, cas",
    [
        ({"avis": []}, "lot vide"),
        ({"avis": [{"texte": POSITIF}] * (LOT_MAX + 1)}, "lot trop gros"),
        ({"avis": [{"texte": ""}]}, "avis vide"),
        ({"avis": [{"texte": "a" * 2001}]}, "avis trop long"),
        ({"avis": [{"texte": POSITIF, "sentiment_attendu": "mitigé"}]}, "classe inconnue"),
        ({"avis": [{"texte": POSITIF}], "seuil_negatif": 4.0, "seuil_positif": 2.0}, "seuils inversés"),
        ({"avis": [{"texte": POSITIF}], "seuil_negatif": 3.0, "seuil_positif": 3.0}, "seuils confondus"),
        ({"avis": [{"texte": POSITIF}], "seuil_negatif": 0.5}, "seuil hors des 1 à 5 étoiles"),
        ({}, "champ avis absent"),
    ],
)
def test_lot_invalide_renvoie_422(client, charge, cas):
    assert client.post("/predict/batch", json=charge).status_code == 422, cas


def test_un_lot_ne_remplit_pas_le_tampon_des_logs(client):
    """200 avis d'un coup chasseraient toutes les prédictions unitaires de /logs."""
    avant = client.get("/logs", params={"limite": TAMPON_MAX}).json()["total"]
    client.post("/predict/batch", json={"avis": [{"texte": POSITIF}, {"texte": NEGATIF}]})
    apres = client.get("/logs", params={"limite": TAMPON_MAX}).json()["total"]
    assert apres == avant


def test_info_publie_les_seuils_par_defaut(client):
    """L'interface initialise ses curseurs dessus : une seule source pour ces valeurs."""
    corps = client.get("/info").json()
    assert corps["seuils_defaut"] == {"seuil_negatif": 2.8, "seuil_positif": 3.6}
    assert corps["lot_max"] == LOT_MAX
