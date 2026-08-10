import os
import time
import requests
import pandas as pd
import argparse

# Configuración por defecto
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"
DEFAULT_DATA_PATH = "loan_data.csv"
BATCH_SIZE = 250
DELAY_BETWEEN_REQUESTS = 0.05  # Segundos entre peticiones individuales
DELAY_BETWEEN_BATCHES = 2.0    # Segundos entre tandas (batches)

def main():
    parser = argparse.ArgumentParser(description="Simulador de tráfico de producción para MLOps con Feedback")
    parser.add_argument("--url", type=str, default=DEFAULT_API_URL, help="URL de la API (endpoint /predict)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Ruta al CSV con los datos aislados")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Número de peticiones por tanda")
    parser.add_argument("--limit", type=int, default=None, help="Límite máximo total de peticiones a enviar")
    args = parser.parse_args()

    # Construimos la URL de feedback a partir de la URL de predict
    feedback_url = args.url.replace("/predict", "/feedback")

    print(f"--- Iniciando Simulador de Tráfico MLOps ---")
    print(f"URL Predict: {args.url}")
    print(f"URL Feedback: {feedback_url}")
    print(f"Origen de datos: {args.data}")

    if not os.path.exists(args.data):
        print(f"Error: No se ha encontrado el archivo {args.data}")
        return

    # 1. Cargar datos
    print("Cargando datos reales...")
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return

    # Incluimos loan_status para guardar la verdad real
    columnas_requeridas = [
        "person_age", "person_education", "person_income", "person_emp_exp",
        "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length", "credit_score",
        "previous_loan_defaults_on_file", "loan_status"
    ]
    
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        print(f"Error: Faltan las siguientes columnas en el CSV: {faltantes}")
        return

    df_simulacion = df[columnas_requeridas].copy()
    filas_iniciales = len(df_simulacion)
    df_simulacion = df_simulacion.dropna()
    print(f"Datos cargados: {len(df_simulacion)} registros válidos.")

    if args.limit:
        df_simulacion = df_simulacion.head(args.limit)
        print(f"Limitando simulación a {args.limit} registros.")

    registros = df_simulacion.to_dict(orient="records")

    total_enviados = 0
    exitosos = 0
    fallidos = 0
    
    # 2. Iterar en tandas (batches)
    for i in range(0, len(registros), args.batch_size):
        batch = registros[i:i + args.batch_size]
        print(f"\n[+] Iniciando tanda {i // args.batch_size + 1} (Tamaño: {len(batch)} peticiones)...")
        
        feedback_batch = []
        
        for idx, registro in enumerate(batch):
            try:
                payload = {
                    "person_age": int(registro["person_age"]),
                    "person_education": str(registro["person_education"]),
                    "person_income": float(registro["person_income"]),
                    "person_emp_exp": int(registro["person_emp_exp"]),
                    "person_home_ownership": str(registro["person_home_ownership"]),
                    "loan_amnt": float(registro["loan_amnt"]),
                    "loan_intent": str(registro["loan_intent"]),
                    "loan_int_rate": float(registro["loan_int_rate"]),
                    "loan_percent_income": float(registro["loan_percent_income"]),
                    "cb_person_cred_hist_length": int(registro["cb_person_cred_hist_length"]),
                    "credit_score": int(registro["credit_score"]),
                    "previous_loan_defaults_on_file": str(registro["previous_loan_defaults_on_file"])
                }

                respuesta = requests.post(args.url, json=payload, timeout=5)
                
                if respuesta.status_code == 200:
                    exitosos += 1
                    data_resp = respuesta.json()
                    pred_id = data_resp.get("prediction_id")
                    
                    # Si la API nos devuelve el ID insertado, guardamos la pareja para el feedback
                    if pred_id is not None:
                        real_status = int(registro["loan_status"])
                        feedback_batch.append({
                            "prediction_id": pred_id,
                            "loan_status": real_status
                        })
                else:
                    fallidos += 1
                    print(f"Error {respuesta.status_code} en registro {i+idx}: {respuesta.text}")
            
            except requests.exceptions.RequestException as e:
                fallidos += 1
                print(f"Fallo de conexión en registro {i+idx}: {e}")
            
            total_enviados += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        print(f"Tanda de inferencias finalizada. {exitosos} éxitos acumulados.")

        # 3. Enviar el reporte de feedback de la tanda al nuevo endpoint
        if feedback_batch:
            print(f"Enviando reporte de feedback para {len(feedback_batch)} registros a {feedback_url}...")
            try:
                fb_resp = requests.post(feedback_url, json=feedback_batch, timeout=10)
                if fb_resp.status_code == 200:
                    print(f"✅ Reporte de feedback aceptado por la API con éxito.")
                else:
                    print(f"❌ Error al enviar feedback ({fb_resp.status_code}): {fb_resp.text}")
            except Exception as e:
                print(f"❌ Excepción al enviar feedback: {e}")
        
        if i + args.batch_size < len(registros):
            print(f"Esperando {DELAY_BETWEEN_BATCHES} segundos para la siguiente tanda...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            
    print(f"\n--- Simulación Completada ---")
    print(f"Total procesados: {total_enviados}")
    print(f"Exitosos (guardados en BD): {exitosos}")
    print(f"Fallidos: {fallidos}")

if __name__ == "__main__":
    main()
