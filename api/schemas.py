from pydantic import BaseModel, Field
from typing import Literal


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_education: Literal["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(..., ge=0, le=60)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    loan_amnt: float = Field(..., gt=0)
    loan_intent: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    loan_int_rate: float
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_cred_hist_length: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    previous_loan_defaults_on_file: Literal["Yes", "No"]


class PredictionResponse(BaseModel):
    loan_status: int
    probability: float