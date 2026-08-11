import os
import joblib
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from api.main import app
from api.model_loader import model, predict

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_db_save():
    """Simula (mockea) la base de datos para que los tests no guarden basura en producción."""
    with patch("api.main.save_prediction", return_value=1) as mock_save, \
         patch("api.main.update_predictions_feedback") as mock_update:
        yield mock_save, mock_update

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

PERFIL_VALIDO = {
    "person_age": 35,
    "person_education": "Master",
    "person_income": 80000,
    "person_emp_exp": 10,
    "person_home_ownership": "RENT",
    "loan_amnt": 5000,
    "loan_intent": "EDUCATION",
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.06,
    "cb_person_cred_hist_length": 5,
    "credit_score": 700,
    "previous_loan_defaults_on_file": "No"
}


# ==============================================================================
# BLOQUE 1: Tests básicos de API
# ==============================================================================

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_devuelve_200():
    response = client.post("/predict", json=PERFIL_VALIDO)
    assert response.status_code == 200


def test_predict_estructura_respuesta():
    response = client.post("/predict", json=PERFIL_VALIDO)
    data = response.json()
    assert "loan_status" in data
    assert "probability" in data


def test_predict_loan_status_es_binario():
    response = client.post("/predict", json=PERFIL_VALIDO)
    data = response.json()
    assert data["loan_status"] in [0, 1]


def test_predict_probabilidad_valida():
    response = client.post("/predict", json=PERFIL_VALIDO)
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_rechaza_datos_invalidos():
    perfil_malo = PERFIL_VALIDO.copy()
    perfil_malo["credit_score"] = 9999  # Fuera del rango permitido (300-850)
    response = client.post("/predict", json=perfil_malo)
    assert response.status_code == 422  # 422 = Unprocessable Entity


# ==============================================================================
# BLOQUE 2: Tests de carga e integridad del modelo
# ==============================================================================

def test_modelo_archivo_existe():
    """Verifica que el archivo model.pkl existe en la ruta esperada."""
    assert os.path.exists(MODEL_PATH), f"No se encontró el modelo en: {MODEL_PATH}"


def test_modelo_es_pipeline_sklearn():
    """Verifica que el objeto cargado es un Pipeline de scikit-learn."""
    assert isinstance(model, Pipeline), f"Se esperaba un Pipeline de sklearn, se obtuvo: {type(model)}"

def test_predict_defaults_siempre_deniega():
    """Si tiene impagos previos, el modelo siempre debe denegar."""
    perfil_default = PERFIL_VALIDO.copy()
    perfil_default["previous_loan_defaults_on_file"] = "Yes"
    response = client.post("/predict", json=perfil_default)
    assert response.json()["loan_status"] == 0

def test_modelo_tiene_pasos_correctos():
    """Verifica que el pipeline tiene exactamente los pasos 'preprocessor' y 'classifier'."""
    nombres_pasos = [nombre for nombre, _ in model.steps]
    assert "preprocessor" in nombres_pasos, "El pipeline no tiene el paso 'preprocessor'"
    assert "classifier" in nombres_pasos, "El pipeline no tiene el paso 'classifier'"


def test_modelo_prediccion_es_determinista():
    """El mismo perfil siempre debe devolver exactamente el mismo resultado."""
    prob1, pred1 = predict(PERFIL_VALIDO)
    prob2, pred2 = predict(PERFIL_VALIDO)
    assert prob1 == prob2, "La probabilidad no es determinista"
    assert pred1 == pred2, "La predicción no es determinista"


def test_modelo_probabilidades_suman_uno():
    """Las probabilidades de clase 0 y clase 1 deben sumar exactamente 1.0."""
    loaded_model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([PERFIL_VALIDO])
    probas = loaded_model.predict_proba(df)[0]
    assert abs(probas[0] + probas[1] - 1.0) < 1e-6, f"Las probabilidades no suman 1: {probas}"


# ==============================================================================
# BLOQUE 3: Tests de Feedback
# ==============================================================================

def test_feedback_endpoint_valido():
    feedback_data = [
        {"prediction_id": 1, "loan_status": 1},
        {"prediction_id": 2, "loan_status": 0}
    ]
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "updated_records": 2}


def test_feedback_rechaza_datos_invalidos():
    feedback_malo = [
        {"prediction_id": 1} # Falta loan_status
    ]
    response = client.post("/feedback", json=feedback_malo)
    assert response.status_code == 422
