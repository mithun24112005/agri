import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = (380, 380)

MODEL_DIR = os.path.join(BASE_DIR, "models", "pretrained")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_efficientnetb4.h5")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

CONFIDENCE_THRESHOLD = 0.50
