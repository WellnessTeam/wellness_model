from fastapi import APIRouter
from app.models.model_predictor import predict_model
from app.schemas import InputSchema

router = APIRouter()

@router.post("/predict")
