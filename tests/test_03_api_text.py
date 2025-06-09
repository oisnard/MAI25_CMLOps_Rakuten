import pytest
import requests 
from fastapi.testclient import TestClient
from src.api.service import app

client = TestClient(app)


def test_predict_text_endpoint():
    """Test the /predict_text endpoint with a valid request.
    """
    endpoint = "/predict_text"
    data = {
        "designation": "Test Product",
        "description": "This is a test product description."
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecode" in response.json()

def test_predict_text_endpoint_missing_designation():
    """Test the /predict_text endpoint with a missing designation.
    """
    endpoint = "/predict_text"
    data = {
        "description": "This is a test product description."
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_batch_endpoint():
    """Test the /predict_text_batch endpoint with a valid request.
    """
    endpoint = "/predict_text_batch"
    data = {
        "designations": ["Test Product 1", "Test Product 2"],
        "descriptions": ["Description for product 1", "Description for product 2"]
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_batch_endpoint_mismatched_lengths():
    """Test the /predict_text_batch endpoint with mismatched lengths of designations and descriptions.
    """
    endpoint = "/predict_text_batch"
    data = {
        "designations": ["Test Product 1", "Test Product 2"],
        "descriptions": ["Description for product 1"]
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 400
    assert "Designations and descriptions must have the same length." in response.json()["detail"]

def test_predict_text_batch_endpoint_missing_designations():
    """Test the /predict_text_batch endpoint with missing designations.
    """
    endpoint = "/predict_text_batch"
    data = {
        "descriptions": ["Description for product 1", "Description for product 2"]
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_text_list_endpoint():
    """Test the /predict_text_list endpoint with a valid request.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": ["This is a test text for prediction.", "Another test text."]
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_list_endpoint_invalid_input():
    """Test the /predict_text_list endpoint with invalid input.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": "This is not a list of strings."
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_list_endpoint_empty_list():
    """Test the /predict_text_list endpoint with an empty list.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": []
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 400
    assert "Input must be a list of strings." in response.json()["detail"]

def test_predict_text_list_endpoint_missing_text():
    """Test the /predict_text_list endpoint with missing text.
    """
    endpoint = "/predict_text_list"
    data = {}
    response = client.post(endpoint, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_list_endpoint_non_list_input():
    """Test the /predict_text_list endpoint with non-list input.
    """
    endpoint = "/predict_text_list"
    data = {
        "text": "This is a single string, not a list."
    }
    response = client.post(endpoint, json=data)
    assert response.status_code == 422  # Unprocessable Entity
