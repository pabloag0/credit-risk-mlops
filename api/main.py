from fastapi import FastAPI
from api.schemas import LoanApplication, PredictionResponse
from api.model_loader import predict

from fastapi.middleware.cors import CORSMiddleware

import os
from api.database import save_prediction

app = FastAPI(title="Credit Risk API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    # 1. Obtener la petición cruda como diccionario
    application_dict = application.model_dump()
    
    # 2. Hacer la predicción con el modelo cargado
    probability, prediction = predict(application_dict)

    # 3. Guardar asíncronamente (o ignorar fallos) en la Base de Datos
    try:
        save_prediction(application_dict, prediction, probability)
        print("Nueva predicción guardada en la base de datos con éxito.")
    except Exception as e:
        print(f"Error al guardar en base de datos: {e}")

    return PredictionResponse(
        loan_status=prediction,
        probability=probability
    )