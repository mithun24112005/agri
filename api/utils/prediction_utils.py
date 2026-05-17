import numpy as np

from api.data.labels import CLASS_NAMES


def decode_prediction(predictions: np.ndarray) -> dict:
    idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    disease = CLASS_NAMES[idx]

    return {
        "disease": disease,
        "confidence": round(confidence, 4),
    }


def get_top_predictions(predictions: np.ndarray, top_k: int = 3) -> list[dict]:
    top_indices = np.argsort(predictions)[0][::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "label": CLASS_NAMES[idx],
            "confidence": round(float(predictions[0][idx]), 4),
        })

    return results
