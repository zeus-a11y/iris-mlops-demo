"""FastAPI inference service for the Iris model, instrumented for Prometheus."""
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.inference import predict

app = FastAPI(title="Iris Classifier API", version="1.0.0")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(features: IrisFeatures):
    try:
        return predict(
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
