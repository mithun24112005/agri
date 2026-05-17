from fastapi import APIRouter, UploadFile, File, HTTPException

from api.services.disease_service import predict_disease

router = APIRouter(prefix="/disease", tags=["disease-detection"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        result = predict_disease(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during prediction: {str(e)}",
        )

    return result
