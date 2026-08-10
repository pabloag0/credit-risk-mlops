import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

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

def save_prediction(application_data: dict, prediction: int, probability: float):
    """
    Guarda una nueva predicción realizada por la API en la base de datos.
    El loan_status real es desconocido en este punto (se deja como None/NULL por defecto).
    """
    engine = get_engine()
    if engine is None:
        # En caso de que no haya BD configurada, ignoramos silenciosamente
        return
    
    db_data = application_data.copy()
    db_data["model_prediction"] = int(prediction)
    db_data["prediction_prob"] = float(probability)
    db_data["data_source"] = "api"
    db_data["model_version"] = "v1.0"
    
    # loan_status (real) se ignora aquí, por lo que Pandas/Postgres lo dejarán como NULL/None

    df_to_save = pd.DataFrame([db_data])
    df_to_save.to_sql("loan_predictions", engine, if_exists="append", index=False)

def load_training_data() -> pd.DataFrame:
    """
    Descarga los datos históricos y consolidados para re-entrenar el modelo.
    Solo descarga aquellas filas donde el loan_status (resultado real) ya es conocido.
    """
    engine = get_engine()
    if engine is None:
        raise ValueError("DATABASE_URL no configurada. Imposible leer datos de entrenamiento.")
    
    # Ignorar predicciones recientes de la API que aún no han sido etiquetadas con su outcome real
    query = "SELECT * FROM loan_predictions WHERE loan_status IS NOT NULL"
    df = pd.read_sql(query, engine)
    
    # Limpiar columnas meta-analíticas de la BD antes de devolvérselo a Scikit-Learn
    columnas_a_ignorar = [
        "id", "created_at", "model_prediction", 
        "prediction_prob", "data_source", "model_version"
    ]
    df = df.drop(columns=[col for col in columnas_a_ignorar if col in df.columns])
    
    return df
