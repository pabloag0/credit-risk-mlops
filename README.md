# 🏦 Credit Risk MLOps System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%2B-4169E1.svg)](https://www.postgresql.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

Un sistema de **MLOps End-to-End** de producción diseñado para evaluar y gestionar el **Riesgo de Crédito (Credit Risk)**. El proyecto abarca desde el preprocesamiento y entrenamiento automatizado del modelo, trazabilidad en base de datos PostgreSQL, versión dinámica mediante metadatos, calidad garantizada por CI/CD, hasta una interfaz web estática de alto rendimiento y un arnés de simulación de tráfico real.

---

## 📌 Tabla de Contenidos
- [🎯 Objetivo del Proyecto](#-objetivo-del-proyecto)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [🗄️ Esquema de la Base de Datos](#️-esquema-de-la-base-de-datos)
- [🔄 Pipeline MLOps y Ciclo de Vida](#-pipeline-mlops-y-ciclo-de-vida)
  - [1. Data Ingestion & Seeding](#1-data-ingestion--seeding)
  - [2. Model Training & MLflow Tracking](#2-model-training--mlflow-tracking)
  - [3. Auto-Versioning & Metadata Centralization](#3-auto-versioning--metadata-centralization)
  - [4. Data Access Layer (DAL) & Inference API](#4-data-access-layer-dal--inference-api)
  - [5. CI/CD & Automated Quality Gates](#5-cicd--automated-quality-gates)
  - [6. Production Traffic Simulator](#6-production-traffic-simulator)
- [🚀 Guía de Instalación y Ejecución Local](#-guía-de-instalación-y-ejecución-local)
- [🛡️ Pruebas Automatizadas y Mocks](#️-pruebas-automatizadas-y-mocks)
- [🌐 Despliegue en Producción (Render)](#-despliegue-en-producción-render)

---

## 🎯 Objetivo del Proyecto

El objetivo principal es predecir la probabilidad de impago (`loan_status`: 0 = Aprobado, 1 = Denegado/Riesgo) de un cliente solicitante de crédito basándose en su perfil socioeconómico e historial crediticio.

A diferencia de proyectos académicos aislados, esta solución aborda problemas reales de producción:
1. **Model & Data Drift Monitoring**: Monitorización continua de predicciones vs. datos reales.
2. **Desacoplamiento Total**: Separación entre Frontend (sitio estático), Backend (FastAPI REST API), Capa de Persistencia (PostgreSQL) y Pipeline de ML.
3. **Traceability & Auditing**: Identificación exacta de qué versión de modelo generó cada predicción en la base de datos.

---

## 🏗️ Arquitectura del Sistema

```mermaid
flowchart TD
    subgraph Frontend Layer
        UI[💻 Static UI HTML5/JS/Tailwind]
    end

    subgraph Data & Simulation
        CSV[📄 initial_train.csv]
        SIM[⚡ simulator.py]
    end

    subgraph API & Serving Layer
        API[🚀 FastAPI Server /predict]
        DAL[🔌 Data Access Layer api/database.py]
        ML[🧠 Scikit-Learn Pipeline model.pkl]
    end

    subgraph Database Layer
        DB[(🛢️ PostgreSQL loan_predictions)]
    end

    subgraph CI/CD & Quality
        GA[⚙️ GitHub Actions CI]
        METRIC[📊 metrics.json Baseline & Version]
    end

    CSV -->|populate_db.py| DB
    UI -->|HTTP POST| API
    SIM -->|HTTP POST Batches| API
    API --> ML
    API --> DAL
    DAL --> DB
    GA -->|Quality Gate / Pytest| METRIC
```

---

## 📂 Estructura del Repositorio

```text
credit-risk-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Integración Continua: Linter y Tests unitarios de API
│       └── model_evaluation.yml       # Quality Gate: Verificación de no degradación del modelo (F1)
├── api/
│   ├── database.py                    # Capa de Acceso a Datos (DAL) aislada con SQLAlchemy
│   ├── main.py                        # Servidor principal FastAPI y configuración de CORS
│   ├── model_loader.py                # Carga en memoria del pipeline Scikit-Learn (joblib)
│   └── schemas.py                     # DTOs y validación estricta de entrada con Pydantic
├── frontend/
│   └── index.html                     # Interfaz gráfica SPA de alto rendimiento (HTML5/Tailwind)
├── model/
│   ├── metrics.json                   # Registro ligero de metadatos (Model Version, F1, Accuracy)
│   └── model.pkl                      # Pipeline empaquetado de Scikit-Learn
├── tests/
│   ├── test_api.py                    # Tests de endpoints de API con DB MOCKED (sin basura en Prod)
│   ├── test_model_quality.py          # Test del Quality Gate del modelo contra el baseline
│   └── validation_sample.csv          # Muestra estática aislada para la evaluación del CI
├── training/
│   ├── initial_train.csv              # Dataset inicial de entrenamiento
│   ├── populate_db.py                 # Script de sembrado/migración inicial a la BD PostgreSQL
│   └── train.py                       # Script principal de entrenamiento y tracking con MLflow
├── Dockerfile                         # Receta de construcción del contenedor Docker para la API
├── docker-compose.yml                 # Orquestador para ejecución en entorno de desarrollo local
├── pytest.ini                         # Configuración global de Pytest
├── requirements.txt                   # Dependencias necesarias en producción y CI
└── simulator.py                       # Arnés de simulación de tráfico real en lotes (batch simulation)
```

---

## 🗄️ Esquema de la Base de Datos

Toda la persistencia se gestiona en la tabla **`loan_predictions`** de PostgreSQL, diseñada bajo los principios de observabilidad en MLOps:

```sql
CREATE TABLE loan_predictions (
    id SERIAL PRIMARY KEY,                          -- Identificador único autoincremental
    person_age FLOAT,                              -- Edad del cliente
    person_education VARCHAR(50),                   -- Nivel educativo
    person_income FLOAT,                           -- Ingresos anuales
    person_emp_exp INT,                            -- Experiencia laboral (años)
    person_home_ownership VARCHAR(50),              -- Tipo de vivienda (RENT, MORTGAGE, OWN, OTHER)
    loan_amnt FLOAT,                               -- Cantidad solicitada
    loan_intent VARCHAR(50),                       -- Propósito del préstamo
    loan_int_rate FLOAT,                           -- Tipo de interés esperado
    loan_percent_income FLOAT,                     -- Porcentaje del sueldo que representa el préstamo
    cb_person_cred_hist_length FLOAT,              -- Años de historial crediticio
    credit_score INT,                              -- Puntuación de crédito (300-850)
    previous_loan_defaults_on_file VARCHAR(10),    -- ¿Impagos previos en registro? (Yes/No)
    loan_status INT,                               -- Outcome REAL (0 = Aprobado, 1 = Denegado / NULL en inferencia)
    model_prediction INT,                          -- Predicción realizada por el modelo (0 o 1)
    prediction_prob FLOAT,                         -- Probabilidad estimada por el modelo
    data_source VARCHAR(20),                       -- Origen del dato ('training_csv', 'api')
    model_version VARCHAR(30),                     -- Versión exacta del modelo (ej. 'v-20260810-120000')
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Fecha automática de inserción
);
```

---

## 🔄 Pipeline MLOps y Ciclo de Vida

### 1. Data Ingestion & Seeding
El script `training/populate_db.py` lee el dataset inicial `initial_train.csv`, elimina columnas obsoletas (como `person_gender`), inyecta los metadatos `data_source = 'training_csv'` y la versión del modelo vigente, y puebla la base de datos PostgreSQL.

### 2. Model Training & MLflow Tracking
Al ejecutar `training/train.py`:
- Descarga automáticamente los datos etiquetados desde PostgreSQL mediante la consulta:
  $$\text{SELECT * FROM loan\_predictions WHERE loan\_status IS NOT NULL}$$
- Limpia las variables metaanalíticas (`id`, `created_at`, `model_prediction`, etc.).
- Construye un **Pipeline de Scikit-Learn** completo:
  - **Atributos Numéricos**: `SimpleImputer(strategy='median')` + `StandardScaler()`
  - **Atributos Categóricos**: `SimpleImputer(fill_value='missing')` + `OneHotEncoder(handle_unknown='ignore')`
  - **Clasificador**: `LogisticRegression(max_iter=1000, random_state=42)`
- Loggea parámetros, métricas (Accuracy, F1 Macro) y el propio artefacto del modelo en **MLflow**.

### 3. Auto-Versioning & Metadata Centralization
Al finalizar el entrenamiento con éxito:
1. El script genera una etiqueta de versión basada en un timestamp único: `v-YYYYMMDD-HHMMSS`.
2. Actualiza `model/metrics.json`, registrando la versión, la fecha ISO de entrenamiento y las métricas baseline.
3. Al arrancar la API o ejecutar predicciones, el módulo `api/database.py` lee de manera dinámica esta versión del archivo `metrics.json`, garantizando que **todas las predicciones en producción queden etiquetadas con la versión exacta que está sirviendo la API**.

### 4. Data Access Layer (DAL) & Inference API
- **FastAPI Backend (`api/main.py`)**: Expone el endpoint POST `/predict` protegido con validación Pydantic estricta (`api/schemas.py`).
- **Data Access Layer (`api/database.py`)**:
  - Implementa el patrón **Singleton** para la conexión SQLAlchemy (`get_engine()`), evitando saturar el pool de conexiones de la base de datos.
  - Al realizar una predicción, inserta la fila en `loan_predictions` asignando `data_source = 'api'`, `model_prediction` y `prediction_prob`. El campo `loan_status` se deja intencionadamente como `NULL` a la espera del resultado real.
  - Está envuelto en un bloque `try/except` que garantiza **tolerancia a fallos**: si la base de datos se cae, la API sigue respondiendo la predicción al cliente web.

### 5. CI/CD & Automated Quality Gates
Mediante **GitHub Actions**:
- **CI Pipeline (`ci.yml`)**: Se ejecuta en cada `push` o `pull_request`. Verifica la sintaxis, instala dependencias y lanza la suite de tests unitarios.
- **Model Evaluation Pipeline (`model_evaluation.yml`)**: Se activa ante cambios en el modelo o script de entrenamiento. Ejecuta un **Quality Gate** (`tests/test_model_quality.py`) que evalúa el modelo actual sobre `tests/validation_sample.csv` y exige que su F1 Macro **no caiga más de un 2%** respecto al baseline registrado en `metrics.json`.

### 6. Production Traffic Simulator
El archivo `simulator.py` es un arnés de simulación que lee un dataset real aislado y envía peticiones HTTP POST por lotes (batches) a la API:
```bash
# Ejemplo: enviar peticiones en lotes de 250 contra la API en producción
python simulator.py --url "https://tu-api.onrender.com/predict" --batch-size 250
```

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Clonar el repositorio y configurar el entorno
```bash
git clone https://github.com/tu-usuario/credit-risk-mlops.git
cd credit-risk-mlops

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto:
```env
DATABASE_URL=postgresql://usuario:password@localhost:5432/credit_risk_db
```

### 3. Entrenar el modelo localmente
```bash
python training/train.py
```

### 4. Levantar la API localmente (opciones)
**Vía Python directo:**
```bash
uvicorn api.main:app --reload --port 8000
```
**Vía Docker Compose:**
```bash
docker-compose up --build -d
```

### 5. Probar el Frontend
Abre directamente el archivo `frontend/index.html` en cualquier navegador web.

---

## 🛡️ Pruebas Automatizadas y Mocks

Para ejecutar la batería completa de pruebas:
```bash
pytest tests/ -v
```

> [!IMPORTANT]
> **Protección contra contaminación de datos**: 
> Los tests de la API utilizan un fixture automático (`autouse=True`) en `tests/test_api.py` que intercepta la función `pandas.DataFrame.to_sql` mediante `unittest.mock.patch`. Esto garantiza que la ejecución de `pytest` (local o en GitHub Actions) **nunca inserte registros basura en tu base de datos de producción**.

---

## 🌐 Despliegue en Producción (Render)

El proyecto está diseñado para desplegarse fácilmente en **Render**:
1. **Base de Datos**: Instancia PostgreSQL gestionada en Render.
2. **Web Service (API)**: Conectado al repositorio de GitHub con construcción mediante `Dockerfile`.
   - Variable de entorno en Render: `DATABASE_URL` (configurada con el Internal Database URL comenzando por `postgresql://`).
3. **Frontend**: Alojado como Static Site o consumido directamente vía `index.html`.