"""Route /logs : contenu, bornes, et non-divulgation du texte client."""

import pytest

from app.journal import TAMPON_MAX


def test_logs_expose_la_derniere_prediction(client):
    client.post("/predict", json={"text": "Punaises de lit dans la chambre 108."})
    corps = client.get("/logs", params={"limite": 1}).json()
    assert corps["total"] == 1
    assert corps["entrees"][0]["sentiment"] == "négatif"
    assert corps["entrees"][0]["latence_ms"] > 0


def test_logs_ne_laisse_sortir_aucun_texte_client(client):
    """L'extrait reste dans logs/api.log, côté serveur : la route ne le renvoie pas."""
    client.post("/predict", json={"text": "Réservation au nom de M. Dupont, chambre 214."})
    corps = client.get("/logs").json()
    assert "Dupont" not in str(corps)
    assert set(corps["entrees"][0]) == {"horodatage", "sentiment", "latence_ms"}


@pytest.mark.parametrize(
    "limite, attendu",
    [
        (0, 422),
        (1, 200),
        (TAMPON_MAX, 200),
        (TAMPON_MAX + 1, 422),
    ],
)
def test_bornes_du_parametre_limite(client, limite, attendu):
    assert client.get("/logs", params={"limite": limite}).status_code == attendu
