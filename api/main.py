from fastapi import FastAPI
from api.schemas import LoanApplication, PredictionResponse, FeedbackItem
from api.model_loader import predict
from typing import List

from fastapi.middleware.cors import CORSMiddleware

import os
from api.database import save_prediction, update_predictions_feedback

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
   
    application_dict = application.model_dump()
    
    probability, prediction = predict(application_dict)

    prediction_id = None
    try:
        prediction_id = save_prediction(application_dict, prediction, probability)
        print(f"Nueva predicción guardada en la base de datos con éxito (ID: {prediction_id}).")
    except Exception as e:
        print(f"Error al guardar en base de datos: {e}")

    return PredictionResponse(
        loan_status=prediction,
        probability=probability,
        prediction_id=prediction_id
    )
@app.post("/feedback")
def recieve_feedback(feedback_list: List[FeedbackItem]):
    feedback_data = [item.model_dump() for item in feedback_list]
    update_predictions_feedback(feedback_data)
    return {"status": "ok", "updated_records": len(feedback_list)}

