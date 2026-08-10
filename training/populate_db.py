import os
import pandas as pd
from api.database import get_engine

def seed_database():
    print("Conectando a la base de datos...")
    engine = get_engine()
    
    if engine is None:
        print("Error: No se ha encontrado DATABASE_URL en el archivo .env")
        return

    csv_path = "training/initial_train.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo {csv_path}")
        return

    print("Leyendo datos del CSV...")
    df = pd.read_csv(csv_path)

    if 'person_gender' in df.columns:
        df = df.drop(columns=['person_gender'])

    # Añadir meta-columnas del nuevo esquema (loan_status real ya viene en el CSV)
    df["data_source"] = "training_csv"
    df["model_version"] = "v1.0"
    df["model_prediction"] = None # No calculado
    df["prediction_prob"] = None  # No calculado

    print(f"Subiendo {len(df)} filas a la base de datos de Render...")
    try:
        df.to_sql("loan_predictions", engine, if_exists="append", index=False)
        print("¡Éxito! Los datos han sido subidos a la tabla 'loan_predictions'.")
    except Exception as e:
        print(f"Error al subir los datos: {e}")

if __name__ == "__main__":
    seed_database()
