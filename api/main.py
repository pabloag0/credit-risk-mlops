from fastapi import FastAPI
from api.schemas import LoanApplication, PredictionResponse
from api.model_loader import predict

from fastapi.middleware.cors import CORSMiddleware

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Cargar variables de entorno y configurar conexión a BD
load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = None
if database_url:
    engine = create_engine(database_url)

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
    if engine is not None:
        try:
            # Añadir el resultado de la predicción a los datos del cliente
            db_data = application_dict.copy()
            db_data["loan_status"] = int(prediction)
            
            # Subir a PostgreSQL
            df_to_save = pd.DataFrame([db_data])
            df_to_save.to_sql("loan_data", engine, if_exists="append", index=False)
            print("Nueva predicción guardada en la base de datos con éxito.")
        except Exception as e:
            print(f"Error al guardar en base de datos: {e}")

    return PredictionResponse(
        loan_status=prediction,
        probability=probability
    )