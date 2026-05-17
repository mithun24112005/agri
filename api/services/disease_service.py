import os
import uuid

from fastapi import HTTPException, UploadFile

from api.config import UPLOAD_FOLDER
from api.utils.image_utils import validate_image, preprocess_image
from api.utils.prediction_utils import decode_prediction, get_top_predictions
from api.models.disease_model import disease_model


def predict_disease(file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    validate_image(file.filename)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        content = file.file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    try:
        processed_image = preprocess_image(file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image preprocessing failed: {str(e)}",
        )

    try:
        predictions = disease_model.predict(processed_image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}",
        )

    try:
        result = decode_prediction(predictions)
        result["top_predictions"] = get_top_predictions(predictions, top_k=3)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction decoding failed: {str(e)}",
        )

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

    return result
