import json

import pandas as pd
import joblib
import mlflow
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from dotenv import load_dotenv



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
    # Cargar variables de entorno y conectar a la BD
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("No se ha encontrado DATABASE_URL en el entorno.")
        
    engine = create_engine(database_url)
    
    print("Descargando datos de entrenamiento desde PostgreSQL...")
    df = pd.read_sql("SELECT * FROM loan_data", engine)
    
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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    with mlflow.start_run():
        # Entrenamiento solo con el 90%
        model.fit(X_train, y_train)

        # Cálculo de métricas sobre el 10% de validación (datos no vistos)
        from sklearn.metrics import f1_score
        preds = model.predict(X_val)
        accuracy = model.score(X_val, y_val)
        f1 = f1_score(y_val, preds, average="macro")

        # Log de hiperparámetros
        mlflow.log_param("max_iter", 1000) #HARDCODED
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("random_state", 42) #HARDCODED


        # Log de métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)

        # Log del modelo
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["numpy.dtype"])

        print(f"Entrenamiento completado — Accuracy: {accuracy:.4f} | F1 macro: {f1:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    # Evaluar sobre el mismo validation_sample.csv que usa el Quality Gate del CI
    # (así metrics.json y el test comparten exactamente el mismo punto de referencia)
    VALIDATION_PATH = os.path.join(BASE_DIR, "..", "tests", "validation_sample.csv")
    df_val = pd.read_csv(VALIDATION_PATH)
    if "person_gender" in df_val.columns:
        df_val = df_val.drop(columns=["person_gender"])
    X_gate = df_val.drop(columns=[TARGET_COL])
    y_gate = df_val[TARGET_COL]
    preds_gate = model.predict(X_gate)
    f1_gate = f1_score(y_gate, preds_gate, average="macro")

    # Guardar métricas como baseline para el Quality Gate del CI
    metrics_path = os.path.join(os.path.dirname(MODEL_OUTPUT_PATH), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": round(accuracy, 4), "f1_macro": round(f1_gate, 4)}, f, indent=2)

    print(f"Métricas guardadas en {metrics_path} (F1 sobre validation_sample: {f1_gate:.4f})")

if __name__ == "__main__":
    main()