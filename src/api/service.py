from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.models.predict_text import predict_text
from src.models.predict_txt_img import predict_text_image
from src.data.preprocessing import remove_all_html_tags
from src.api.middleware import JWTAuthMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
import os

app = FastAPI(title="Rakuten Product Prediction API",
              description="API for predicting product categories",
              version="1.0.0")

Instrumentator().instrument(app).expose(app)

app.add_middleware(JWTAuthMiddleware)

class TextRequest(BaseModel):
    text: List[str]

class ProductDefinition(BaseModel):
    designation: str
    description: str
    image_filepath: str = None  # Optional field for image file path

class ListProductDefinition(BaseModel):
    designations: List[str]
    descriptions: List[str]
    image_filepaths: List[str] = None  # Optional field for image file paths

@app.post("/predict_product", summary="Predict a product prdtypecode from its designation and description")
async def predict_product_endpoint(request: ProductDefinition):
    """
    Predict product type category (prdtypecode) for a given product designation and description.
    
    Args:
        request (ProductDefinition): Product details containing designation and description and filepath of product image.
        
    Returns:
        dict: Predicted categories for the product.
    """
    if not request.designation:
        raise HTTPException(status_code=400, detail="At least a designation is required.")
    
    text = f"{request.designation}. {request.description or ''}"
    text = remove_all_html_tags(text)  # Clean the text from HTML tags
    if request.image_filepath:
        image_filepaths = [request.image_filepath]
    else:
        image_filepaths = None
    try:
        if image_filepaths:
            predictions = predict_text_image([text], image_filepaths)
        else:
            predictions = predict_text([text])
        if -1 in predictions:
            raise HTTPException(status_code=500, detail="Model prediction not available for predictions")
        return {"predicted prdtypecode": predictions[0]}  # Return the first prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_product_batch", summary="Predict a product prdtypecode from a batch of designations and descriptions")
async def predict_product_batch_endpoint(request: ListProductDefinition):
    """
    Predict product type categories (prdtypecode) for a batch of products.
    Args:
        request (ListProductDefinition): List of product details containing designations and descriptions.
    Returns:
        dict: Predicted categories for each product.
    """
    if len(request.designations) != len(request.descriptions):
        raise HTTPException(status_code=400, detail="Designations and descriptions must have the same length.")
    if not request.designations:
        raise HTTPException(status_code=400, detail="At least one designation is required.")
    texts = [f"{d}. {desc or ''}" for d, desc in zip(request.designations, request.descriptions)]
    texts = [remove_all_html_tags(text) for text in texts]  # Clean the text from HTML tags
    if request.image_filepaths and len(request.image_filepaths) != len(texts):
        raise HTTPException(status_code=400, detail="Image file paths must match the number of texts.")
    else:
        image_filepaths = request.image_filepaths

    try:
        if image_filepaths:
            predictions = predict_text_image(texts, image_filepaths)
        else:
            predictions = predict_text(texts)
        if -1 in predictions:
            raise HTTPException(status_code=500, detail="Model prediction not available for predictions")
        return {"predicted prdtypecodes": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_text_list", summary="Predict product categories from a list of texts")
async def predict_text_list(request: TextRequest):
    """
    Predict product categories for a list of texts relating to a list of products.
    Args:
        request (TextRequest): List of texts to predict categories for.
    Returns:
        dict: Predicted categories for each text in the list.
    """
    if not request.text or not isinstance(request.text, list):
        raise HTTPException(status_code=400, detail="Input must be a list of strings.")
    try:
        predictions = predict_text(request.text)
        return {"predicted prdtypecodes": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes probes
    Returns application status and basic system info
    """
    try:
        # Check if data files are accessible
        data_path = "/app/data/processed/mapping_dict.json"
        data_accessible = os.path.exists(data_path)
        
        # Check if models are loaded
        models_loaded = True  # You can add actual model checks here
        
        return {
            "status": "healthy" if data_accessible and models_loaded else "degraded",
            "timestamp": datetime.now().isoformat(),
            "data_accessible": data_accessible,
            "models_loaded": models_loaded,
            "environment": os.getenv("ENVIRONMENT", "unknown"),
            "version": "v1"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "environment": os.getenv("ENVIRONMENT", "unknown")
        }

@app.get("/health/ready")
async def readiness_check():
    """
    Readiness probe endpoint
    Checks if the application is ready to serve requests
    """
    try:
        # Check critical dependencies
        data_path = "/app/data/processed/mapping_dict.json"
        if not os.path.exists(data_path):
            return {"status": "not_ready", "reason": "Data files not accessible"}
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}

@app.get("/health/live")
async def liveness_check():
    """
    Liveness probe endpoint
    Checks if the application is alive and responsive
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


