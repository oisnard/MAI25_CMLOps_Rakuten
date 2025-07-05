import pytest
from fastapi.testclient import TestClient
from src.api.service import app
from src.api.middleware import create_jwt_token

@pytest.fixture(scope="module")
def test_client():
    """
    Fixture qui retourne un client de test FastAPI.
    """
    client = TestClient(app)
    return client

@pytest.fixture(scope="module")
def valid_token():
    """
    Fixture qui génère un JWT valide pour un utilisateur fictif.
    """
    return create_jwt_token("test_user")

@pytest.fixture(scope="module")
def auth_headers(valid_token):
    """
    Fixture qui retourne les en-têtes d'autorisation avec un token JWT valide.
    """
    return {"Authorization": f"Bearer {valid_token}"}

