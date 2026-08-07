import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

# Cargar el pipeline de scikit-learn completo (Preprocesado + Modelo)
model = joblib.load(MODEL_PATH)

def predict(data: dict) -> tuple[float, int]:
    # Convertir el diccionario (JSON validado) a un DataFrame de 1 fila
    # El pipeline de sklearn necesita ingerir datos en formato Pandas
    df = pd.DataFrame([data])
    
    # Obtener probabilidades (devuelve un array con probabilidades para clase 0 y clase 1)
    probabilities = model.predict_proba(df)
    
    # Cogemos la probabilidad de la clase 1
    probability = float(probabilities[0][1])
    
    # La clase predicha directamente (0 o 1)
    prediction = int(model.predict(df)[0])

    return probability, prediction