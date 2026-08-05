from fastapi import FastAPI
from api.schemas import LoanApplication, PredictionResponse
from api.model_loader import predict
from api.preprocessing import preprocess_input

app = FastAPI(title="Credit Risk API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    X = preprocess_input(application.model_dump())
    probability, prediction = predict(X)

    return PredictionResponse(
        loan_status=prediction,
        probability=probability
    )