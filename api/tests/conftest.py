import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # scope="session" : le modèle n'est chargé qu'une fois pour toute la suite de tests
    with TestClient(app) as test_client:
        yield test_client
