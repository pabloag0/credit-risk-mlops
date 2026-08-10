import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Optional

# Cargar variables de entorno (solo necesario en local)
load_dotenv()

# Inicializar motor de base de datos como Singleton (cacheado)
_engine = None

def get_engine():
    """Devuelve el motor de base de datos de SQLAlchemy cacheado."""
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            _engine = create_engine(database_url)
    return _engine

def get_current_model_version():
    """Lee dinámicamente la versión del modelo actual desde metrics.json."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
            return data.get("model_version", "v1.0")
    except Exception:
        return "v1.0" # Fallback por seguridad

def save_prediction(application_data: dict, prediction: int, probability: float) -> Optional[int]:
    """
    Guarda una nueva predicción realizada por la API en la base de datos.
    Devuelve el id autogenerado de la fila insertada o None en caso de fallo.
    """
    engine = get_engine()
    if engine is None:
        return None
    
    db_data = application_data.copy()
    db_data["model_prediction"] = int(prediction)
    db_data["prediction_prob"] = float(probability)
    db_data["data_source"] = "api"
    db_data["model_version"] = get_current_model_version()
    
    columns = ", ".join(db_data.keys())
    placeholders = ", ".join([f":{key}" for key in db_data.keys()])
    query = text(f"INSERT INTO loan_predictions ({columns}) VALUES ({placeholders}) RETURNING id")
    
    try:
        with engine.begin() as connection:
            result = connection.execute(query, db_data)
            inserted_id = result.scalar()
            return inserted_id
    except Exception as e:
        print(f"Error al guardar en base de datos: {e}")
        return None

def update_predictions_feedback(feedback_list: List[dict]):
    """
    Actualiza el loan_status real de múltiples registros en la base de datos.
    Cada elemento de feedback_list debe ser un diccionario con 'prediction_id' y 'loan_status'.
    """
    engine = get_engine()
    if engine is None:
        return
        
    query = text("UPDATE loan_predictions SET loan_status = :loan_status WHERE id = :prediction_id")
    
    try:
        with engine.begin() as connection:
        
            connection.execute(query, feedback_list)
            print(f"Feedback actualizado para {len(feedback_list)} registros con éxito.")
    except Exception as e:
        print(f"Error al actualizar el feedback en base de datos: {e}")

def load_training_data() -> pd.DataFrame:
    """
    Descarga los datos históricos y consolidados para re-entrenar el modelo.
    Solo descarga aquellas filas donde el loan_status (resultado real) ya es conocido.
    """
    engine = get_engine()
    if engine is None:
        raise ValueError("DATABASE_URL no configurada. Imposible leer datos de entrenamiento.")
    
    query = "SELECT * FROM loan_predictions WHERE loan_status IS NOT NULL"
    df = pd.read_sql(query, engine)
    
    columnas_a_ignorar = [
        "id", "created_at", "model_prediction", 
        "prediction_prob", "data_source", "model_version"
    ]
    df = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
    
    return df
