from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team name", example="FC Barcelona")
    away_team: str = Field(..., description="Away team name", example="Real Madrid")
    referee: Optional[str] = Field(None, description="Referee name", example="Michael Oliver")
    venue: Optional[str] = Field(None, description="Stadium name", example="Camp Nou")


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    venue: str
    referee: Optional[str]
    prediction: str
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    cards_prediction: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    feature_count: int
    classes: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    cards_model_loaded: bool
    feature_count: int