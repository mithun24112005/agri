import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import HTTPException
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

from api.config import ALLOWED_EXTENSIONS, IMAGE_SIZE


def validate_image(filename: str) -> None:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


def preprocess_image(image_path: str) -> np.ndarray:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Corrupted or unreadable image: {str(e)}",
        )

    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array
