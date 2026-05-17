import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

from api.config import MODEL_PATH


class DiseaseModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        if self._model is not None:
            return

        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Download it from https://huggingface.co/liriope/PlantDiseaseDetection"
            )

        print(f"Loading model from {MODEL_PATH}...")
        self._model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print("Model loaded successfully.")

    def predict(self, processed_image: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()

        predictions = self._model.predict(processed_image, verbose=0)
        return predictions


disease_model = DiseaseModel()
