from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.train import FEATURES, MODEL_PATH

app = FastAPI(
    title="Bike Demand Forecasting API",
    description="Predict horuly bike rentals from time and weather features",
    version="0.1.0"
)

#what info user must provide when requesting prediction
class PredictionRequest(BaseModel):
    season: int = Field(ge=1, le=4, description="1=spring, 2=summer, 3=autumn, 4=winter")
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6, description="0=Sunday ... 6=Saturday")
    workingday: int = Field(ge=0, le=1)
    weather: int = Field(ge=1, le=4, description="1=clear, 2=mist, 3=light rain/snow, 4=heavy")
    temp: float = Field(ge=0, le=1, description="Normalised temperature (0-1)")
    atemp: float = Field(ge=0, le=1, description="Normalised feels-like temperature (0-1)")
    humidity: float = Field(ge=0, le=1, description="Normalised humidity (0-1)")
    windspeed: float = Field(ge=0, le=1, description="Normalised wind speed (0-1)")

# response api sends back after prediction
class PredictionResponse(BaseModel):
    predicted_rentals: int

#loads previously trained gradient boosting model 
# cache remebers result of calling get_model()
@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)

#checks if api responds
@app.get("/health")
def health() -> dict[str,str]:
    return {"status" : "ok"}

#recieves info from user - passes it through trained model and returns prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    row = pd.DataFrame([request.model_dump()], columns=FEATURES)
    prediction = float(get_model().predict(row)[0])
    return PredictionResponse(predicted_rentals=max(0, round(prediction)))


