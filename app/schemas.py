from typing import Annotated
from pydantic import BaseModel, Field

Pixel = Annotated[float, Field(ge=0, le=255)]

class PredictRequest(BaseModel):
    pixels: list[Pixel] = Field(min_length=784, max_length=784, description="Exatamente 784 pixels em escala de cinza (0-255) representando uma imagem 28x28 achatada.")

class PredictResponse(BaseModel):
    digito: int = Field(ge=0, le=9, description="Dígito previsto pelo modelo (0 a 9).")
    confianca: float = Field(ge=0, le=1, description="Probabilidade (de 0 a 1) que o modelo atribuiu ao dígito previsto.")
    probabilidades: list[float] = Field(min_length=10, max_length=10, description="Probabilidade atribuída a cada dígito, na ordem de 0 a 9.")

class HealthResponse(BaseModel):
    status: str = Field(description="Estado do serviço")
    modelo_carregado: bool = Field(description="Indica se o modelo está carregado na memória RAM.")
