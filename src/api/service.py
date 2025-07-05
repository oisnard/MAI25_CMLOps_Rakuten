from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.models.predict_text import predict_text 
from src.data.make_dataset import remove_all_html_tags
from src.api.middleware import JWTAuthMiddleware

app = FastAPI(title="Rakuten Product Prediction API",
              description="API for predicting product categories",
              version="1.0.0")

app.add_middleware(JWTAuthMiddleware)

class TextRequest(BaseModel):
    text: List[str]

class ProductDefinition(BaseModel):
    designation: str
    description: str

class ListProductDefinition(BaseModel):
    designations: List[str]
    descriptions: List[str]

@app.post("/predict_text", summary="Predict a product prdtypecode from its designation and description")
async def predict_text_endpoint(request: ProductDefinition):
    """
    Predict product type category (prdtypecode) for a given product designation and description.
    
    Args:
        request (ProductDefinition): Product details containing designation and description.
        
    Returns:
        dict: Predicted categories for the product.
    """
    if not request.designation:
        raise HTTPException(status_code=400, detail="At least a designation is required.")
    
    text = f"{request.designation}. {request.description or ''}"
    text = remove_all_html_tags(text)  # Clean the text from HTML tags

    try:
        predictions = predict_text([text])
        return {"predicted prdtypecode": predictions[0]}  # Return the first prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_text_batch", summary="Predict a product prdtypecode from a batch of designations and descriptions")
async def predict_text_batch_endpoint(request: ListProductDefinition):
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

    try:
        predictions = predict_text(texts)
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


