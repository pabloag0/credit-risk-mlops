import os
import json
import joblib
import pandas as pd
from sklearn.metrics import f1_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
VALIDATION_PATH = os.path.join(os.path.dirname(__file__), "validation_sample.csv")

# Tolerancia máxima de caída respecto al baseline guardado
TOLERANCIA = 0.02


def test_model_quality_gate():
    """
    Quality Gate dinámico: el F1 macro del modelo sobre el validation_sample
    no puede caer más de un 2% respecto al baseline guardado en metrics.json.
    Así cualquier modificación que empeore el modelo será detectada automáticamente.
    """
    # Leer el F1 baseline del último entrenamiento aceptado
    with open(METRICS_PATH, "r") as f:
        baseline = json.load(f)
    f1_baseline = baseline["f1_macro"]

    # Evaluar el modelo actual sobre el validation sample
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(VALIDATION_PATH)

    if "person_gender" in df.columns:
        df = df.drop(columns=["person_gender"])

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    preds = model.predict(X)
    f1_actual = f1_score(y, preds, average="macro")

    umbral = round(f1_baseline - TOLERANCIA, 4)

    assert f1_actual >= umbral, (
        f"Quality Gate fallido: F1 actual = {f1_actual:.4f}, "
        f"baseline = {f1_baseline:.4f}, "
        f"mínimo permitido = {umbral:.4f} (tolerancia {TOLERANCIA*100:.0f}%). "
        f"El modelo ha empeorado demasiado respecto al baseline."
    )
