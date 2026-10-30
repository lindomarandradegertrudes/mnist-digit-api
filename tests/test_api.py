import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def cliente():
    with TestClient(app) as c:
        yield c
     
    
def test_health_ok(cliente):
    resposta = cliente.get("/health")
    assert resposta.status_code == 200

    assert resposta.json()["modelo_carregado"] is True
 
   
def test_predict_valido(cliente):
    resposta = cliente.post("/predict", json={"pixels": [0] * 783 + [255]})
    assert resposta.status_code == 200
   
   
def test_predict_422(cliente):
    resposta = cliente.post("/predict", json={"pixels": [1]})
    assert resposta.status_code == 422


def test_predict_400_imagem_em_branco(cliente):
    resposta = cliente.post("/predict", json={"pixels": [0] * 784})
    assert resposta.status_code == 400


def test_predict_503_sem_modelo(cliente, monkeypatch):
    monkeypatch.delattr(app.state, "modelo")
    resposta = cliente.post("/predict", json={"pixels": [0] * 783 + [255]})
    assert resposta.status_code == 503
 

def test_predict_500_falha_na_previsao(cliente, monkeypatch):
    monkeypatch.setattr(app.state, "modelo", object())
    resposta = cliente.post("/predict", json={"pixels": [0] * 783 + [255]})
    assert resposta.status_code == 500
