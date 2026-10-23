from fastapi import APIRouter, HTTPException, Request
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.services import prever

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    carregado = hasattr(request.app.state, "modelo")
    return {"status": "ok", "modelo_carregado": carregado}

@router.post("/predict", response_model=PredictResponse)
def predict(dados: PredictRequest, request: Request):
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return prever(modelo, dados.pixels)