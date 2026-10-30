import numpy as np
import joblib
from pathlib import Path

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "mnist_mlp.joblib"

def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)

def prever(modelo, pixels):
    entrada = np.array(pixels).reshape(1, 784)
    entrada = entrada / 255.0
    probabilidades = modelo.predict_proba(entrada)[0]
    indice = np.argmax(probabilidades)
    return {
        "digito": int(indice),
        "confianca": float(probabilidades[indice]),
        "probabilidades": probabilidades.tolist(),
    }
    