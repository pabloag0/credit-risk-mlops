from fastapi import FastAPI
from api.schemas import LoanApplication, PredictionResponse
from api.model_loader import predict

from fastapi.middleware.cors import CORSMiddleware

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
    # Pasar el diccionario crudo directamente al model_loader
    probability, prediction = predict(application.model_dump())

    return PredictionResponse(
        loan_status=prediction,
        probability=probability
    )