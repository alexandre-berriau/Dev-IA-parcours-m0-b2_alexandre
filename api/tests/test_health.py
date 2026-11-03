def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_info_announces_three_classes(client):
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["sortie_api"] == ["négatif", "neutre", "positif"]
