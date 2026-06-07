"""FastAPI application for Boston Housing price prediction and frontend UI."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import BASE_DIR, load_config
from src.train import FEATURE_COLUMNS


CONFIG = load_config()
MODEL_PATH = BASE_DIR / CONFIG["paths"]["model_path"]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Boston Housing Prediction API",
    description="API and web UI for predicting Boston Housing target variable medv",
    version="1.2.0",
)


FEATURE_METADATA = [
    {
        "name": "crim",
        "label": "Crime rate",
        "description": "Per capita crime rate by town.",
        "placeholder": 0.02729,
        "type": "float",
    },
    {
        "name": "zn",
        "label": "Residential zoning",
        "description": "Proportion of residential land zoned for large lots.",
        "placeholder": 0.0,
        "type": "float",
    },
    {
        "name": "indus",
        "label": "Business area share",
        "description": "Proportion of non-retail business acres per town.",
        "placeholder": 7.07,
        "type": "float",
    },
    {
        "name": "chas",
        "label": "Near Charles River",
        "description": "1 if tract bounds river, otherwise 0.",
        "placeholder": 0,
        "type": "int",
    },
    {
        "name": "nox",
        "label": "Pollution level",
        "description": "Nitric oxides concentration.",
        "placeholder": 0.469,
        "type": "float",
    },
    {
        "name": "rm",
        "label": "Average rooms",
        "description": "Average number of rooms per dwelling.",
        "placeholder": 7.185,
        "type": "float",
    },
    {
        "name": "age",
        "label": "Old houses share",
        "description": "Proportion of owner-occupied units built before 1940.",
        "placeholder": 61.1,
        "type": "float",
    },
    {
        "name": "dis",
        "label": "Distance to job centers",
        "description": "Weighted distances to employment centers.",
        "placeholder": 4.9671,
        "type": "float",
    },
    {
        "name": "rad",
        "label": "Highway access",
        "description": "Accessibility index to radial highways.",
        "placeholder": 2,
        "type": "int",
    },
    {
        "name": "tax",
        "label": "Property tax",
        "description": "Full-value property tax rate.",
        "placeholder": 242,
        "type": "int",
    },
    {
        "name": "ptratio",
        "label": "Pupil-teacher ratio",
        "description": "Pupil-teacher ratio by town.",
        "placeholder": 17.8,
        "type": "float",
    },
    {
        "name": "black",
        "label": "Dataset feature black",
        "description": "Original Boston Housing dataset feature kept for reproducibility.",
        "placeholder": 392.83,
        "type": "float",
    },
    {
        "name": "lstat",
        "label": "Lower status share",
        "description": "Percentage of lower status population.",
        "placeholder": 4.03,
        "type": "float",
    },
]


class HouseFeatures(BaseModel):
    """Input schema for Boston Housing prediction request."""

    crim: float = Field(..., json_schema_extra={"example": 0.02729})
    zn: float = Field(..., json_schema_extra={"example": 0.0})
    indus: float = Field(..., json_schema_extra={"example": 7.07})
    chas: int = Field(..., json_schema_extra={"example": 0})
    nox: float = Field(..., json_schema_extra={"example": 0.469})
    rm: float = Field(..., json_schema_extra={"example": 7.185})
    age: float = Field(..., json_schema_extra={"example": 61.1})
    dis: float = Field(..., json_schema_extra={"example": 4.9671})
    rad: int = Field(..., json_schema_extra={"example": 2})
    tax: int = Field(..., json_schema_extra={"example": 242})
    ptratio: float = Field(..., json_schema_extra={"example": 17.8})
    black: float = Field(..., json_schema_extra={"example": 392.83})
    lstat: float = Field(..., json_schema_extra={"example": 4.03})


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""

    predicted_medv: float


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load trained model artifact from disk.

    Args:
        model_path: Path to serialized model file.

    Returns:
        Loaded model instance.

    Raises:
        FileNotFoundError: If model artifact does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training first."
        )

    return joblib.load(model_path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve frontend application entrypoint."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
def health_check() -> dict[str, bool | str]:
    """Return application health status."""
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/feature-metadata")
def feature_metadata() -> dict[str, Any]:
    """Return user-friendly metadata for frontend rendering."""
    return {
        "features": FEATURE_METADATA,
        "target": "medv",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    """Generate house price prediction for provided features.

    Args:
        features: Validated input payload.

    Returns:
        PredictionResponse with predicted house price.

    Raises:
        HTTPException: 503 if the trained model artifact is missing,
            422 if the feature payload cannot be converted into a prediction.
    """
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        input_df = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
        prediction = float(model.predict(input_df)[0])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature payload: {exc}",
        ) from exc

    return PredictionResponse(predicted_medv=round(prediction, 4))
