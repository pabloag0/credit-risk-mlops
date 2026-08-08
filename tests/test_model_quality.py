import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
VALIDATION_PATH = os.path.join(os.path.dirname(__file__), "validation_sample.csv")

# Umbral mínimo de F1 macro para que el modelo pase a producción
# El modelo baseline obtiene 0.856 en initial_train.csv
F1_MINIMO = 0.70  #HARDCODED


def test_model_quality_gate():
    """
    Quality Gate: el F1 macro del modelo sobre el validation_sample
    debe superar el umbral mínimo definido. Si falla, el modelo no
    debería desplegarse.
    """
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(VALIDATION_PATH)

    if "person_gender" in df.columns:
        df = df.drop(columns=["person_gender"])

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    preds = model.predict(X)
    f1 = f1_score(y, preds, average="macro")

    assert f1 >= F1_MINIMO, (
        f"Quality Gate fallido: F1 macro = {f1:.4f}, mínimo requerido = {F1_MINIMO}. "
        f"El modelo ha empeorado y no debe desplegarse."
    )
