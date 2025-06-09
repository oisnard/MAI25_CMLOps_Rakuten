import pytest
import requests 


@pytest.fixture(scope="module")
def base_url():
    """
    Fixture to provide the base URL for the API.
    """
    return "http://localhost:8000"


def test_predict_text_endpoint(base_url):
    """Test the /predict_text endpoint with a valid request.
    """
    url = f"{base_url}/predict_text"
    data = {
        "designation": "Test Product",
        "description": "This is a test product description."
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecode" in response.json()

def test_predict_text_endpoint_missing_designation(base_url):
    """Test the /predict_text endpoint with a missing designation.
    """
    url = f"{base_url}/predict_text"
    data = {
        "description": "This is a test product description."
    }
    response = requests.post(url, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_batch_endpoint(base_url):
    """Test the /predict_text_batch endpoint with a valid request.
    """
    url = f"{base_url}/predict_text_batch"
    data = {
        "designations": ["Test Product 1", "Test Product 2"],
        "descriptions": ["Description for product 1", "Description for product 2"]
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_batch_endpoint_mismatched_lengths(base_url):
    """Test the /predict_text_batch endpoint with mismatched lengths of designations and descriptions.
    """
    url = f"{base_url}/predict_text_batch"
    data = {
        "designations": ["Test Product 1", "Test Product 2"],
        "descriptions": ["Description for product 1"]
    }
    response = requests.post(url, json=data)
    assert response.status_code == 400
    assert "Designations and descriptions must have the same length." in response.json()["detail"]

def test_predict_text_batch_endpoint_missing_designations(base_url):
    """Test the /predict_text_batch endpoint with missing designations.
    """
    url = f"{base_url}/predict_text_batch"
    data = {
        "descriptions": ["Description for product 1", "Description for product 2"]
    }
    response = requests.post(url, json=data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_text_list_endpoint(base_url):
    """Test the /predict_text_list endpoint with a valid request.
    """
    url = f"{base_url}/predict_text_list"
    data = {
        "text": ["This is a test text for prediction.", "Another test text."]
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "predicted prdtypecodes" in response.json()

def test_predict_text_list_endpoint_invalid_input(base_url):
    """Test the /predict_text_list endpoint with invalid input.
    """
    url = f"{base_url}/predict_text_list"
    data = {
        "text": "This is not a list of strings."
    }
    response = requests.post(url, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_list_endpoint_empty_list(base_url):
    """Test the /predict_text_list endpoint with an empty list.
    """
    url = f"{base_url}/predict_text_list"
    data = {
        "text": []
    }
    response = requests.post(url, json=data)
    assert response.status_code == 400
    assert "Input must be a list of strings." in response.json()["detail"]

def test_predict_text_list_endpoint_missing_text(base_url):
    """Test the /predict_text_list endpoint with missing text.
    """
    url = f"{base_url}/predict_text_list"
    data = {}
    response = requests.post(url, json=data)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_text_list_endpoint_non_list_input(base_url):
    """Test the /predict_text_list endpoint with non-list input.
    """
    url = f"{base_url}/predict_text_list"
    data = {
        "text": "This is a single string, not a list."
    }
    response = requests.post(url, json=data)
    assert response.status_code == 422  # Unprocessable Entity
