import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.disease_routes import router as disease_router
from api.models.disease_model import disease_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    disease_model.load()
    yield


app = FastAPI(
    title="AGRI Disease Detection API",
    description="Plant disease detection using EfficientNet-B4",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "AGRI API Running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(disease_router)
