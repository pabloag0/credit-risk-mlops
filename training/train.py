import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)

#HARDCODED
DATA_PATH = os.path.join(BASE_DIR, "initial_train.csv")
#HARDCODED
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "model", "model.pkl")

#HARDCODED
NUMERIC_FEATURES = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

#HARDCODED
CATEGORICAL_FEATURES = [
    'person_home_ownership', 'loan_intent', 
    'previous_loan_defaults_on_file', 'person_education'
]

#HARDCODED
TARGET_COL = 'loan_status'

def main():
    df = pd.read_csv(DATA_PATH)
    
    df = df[
        (df["person_age"] <= 100) &      #HARDCODED
        (df["person_emp_exp"] <= 60) &   #HARDCODED
        (df["person_income"] <= 1000000) #HARDCODED
    ]
    
    X = df.drop(columns=[TARGET_COL])
    if "person_gender" in X.columns:
        X = X.drop(columns=["person_gender"])
    y = df[TARGET_COL]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()