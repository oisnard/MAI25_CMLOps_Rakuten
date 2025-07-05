import pytest
import requests 
from fastapi.testclient import TestClient
from src.api.service import app

#client = TestClient(app)
    

def test_predict_text_endpoint(test_client, auth_headers):
    """Test the /predict_text endpoint with a valid request.
    """
    endpoint = "/predict_text"
    data = {
        "designation": "Chaussures de sport",
        "description": "Chaussures pour la course à pied et l'entraînement."
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 200
    assert "predicted prdtypecode" in response.json()

def test_predict_text_endpoint_missing_designation(test_client, auth_headers):
    """Test the /predict_text endpoint with a missing designation.
    """
    endpoint = "/predict_text"
    data = {
        "description": "Chaussures pour la course à pied et l'entraînement."
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_batch_endpoint(test_client, auth_headers):
    """Test the /predict_text_batch endpoint with a valid request.
    """
    endpoint = "/predict_text_batch"
    data = {
        "designations": ["Chaussures de sport", "Chaussures de randonnée"],
        "descriptions": ["Chaussures pour la course à pied et l'entraînement.", "Chaussures robustes pour la randonnée en montagne."]
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_batch_endpoint_mismatched_lengths(test_client, auth_headers):
    """Test the /predict_text_batch endpoint with mismatched lengths of designations and descriptions.
    """
    endpoint = "/predict_text_batch"
    data = {
        "designations": ["Chaussures de sport", "Chaussures de randonnée"],
        "descriptions": ["Chaussures pour la course à pied et l'entraînement.", "Chaussures robustes pour la randonnée en montagne."]
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_batch_endpoint_mismatched_lengths(test_client, auth_headers):
    """Test the /predict_text_batch endpoint with mismatched lengths of designations and descriptions.
    """
    endpoint = "/predict_text_batch"
    data = {
        "designations": ["Chaussures de sport", "Chaussures de randonnée"],
        "descriptions": ["Chaussures pour la course à pied et l'entraînement."]
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 400
    assert "Designations and descriptions must have the same length." in response.json()["detail"]

def test_predict_text_batch_endpoint_missing_designations(test_client, auth_headers):
    """Test the /predict_text_batch endpoint with missing designations.
    """
    endpoint = "/predict_text_batch"
    data = {
        "descriptions": ["Chaussures pour la course à pied et l'entraînement.", "Chaussures robustes pour la randonnée en montagne."]
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_text_list_endpoint(test_client, auth_headers):
    """Test the /predict_text_list endpoint with a valid request.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": ["Chaussures pour la course à pied et l'entraînement.", 
                 "Ce produit est excellent pour les cheveux secs."]
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_list_endpoint_invalid_input(test_client, auth_headers):
    """Test the /predict_text_list endpoint with invalid input.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": "This is not a list of strings."
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_list_endpoint_empty_list(test_client, auth_headers):
    """Test the /predict_text_list endpoint with an empty list.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": []
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 400
    assert "Input must be a list of strings." in response.json()["detail"]

def test_predict_text_list_endpoint_missing_text(test_client, auth_headers):
    """Test the /predict_text_list endpoint with missing text.

    """
    endpoint = "/predict_text_list"
    data = {}
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_text_list_endpoint_non_list_input(test_client, auth_headers):
    """Test the /predict_text_list endpoint with non-list input.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": "Ce produit est excellent pour les cheveux secs."
    }
    response = test_client.post(endpoint, json=data, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity
