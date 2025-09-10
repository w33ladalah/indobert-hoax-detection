from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.models.model import HoaxDetectionModel
from app.preprocessing.text_processor import TextPreprocessor

router = APIRouter(tags=["prediction"])

# Initialize model and preprocessor
model = HoaxDetectionModel()
preprocessor = TextPreprocessor()

class TextRequest(BaseModel):
    text: str

@router.post("/predict", response_model=Dict[str, Any])
async def predict_hoax(request: TextRequest):
    """
    Predict whether the input text is a hoax or not.

    Args:
        request: TextRequest object containing the text to analyze

    Returns:
        Dict with prediction results including:
        - label: "hoax" or "fact"
        - confidence: confidence score (0-1)
        - processed_text: the preprocessed text used for prediction
    """
    try:
        # Preprocess the text
        processed_text = preprocessor.preprocess(request.text)

        # Make prediction
        prediction = model.predict(processed_text)

        return {
            "label": "hoax" if prediction["label"] == "hoax" else "fact",
            "confidence": prediction["confidence"],
            "processed_text": processed_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
