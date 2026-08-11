# 🏦 Credit Risk MLOps System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%2B-4169E1.svg)](https://www.postgresql.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

Un sistema de **MLOps End-to-End** de producción diseñado para evaluar y gestionar el **Riesgo de Crédito (Credit Risk)**. El proyecto abarca desde el preprocesamiento y entrenamiento automatizado del modelo, trazabilidad en base de datos PostgreSQL, versión dinámica mediante metadatos, calidad garantizada por CI/CD, hasta una interfaz web estática de alto rendimiento y un arnés de simulación de tráfico real con feedback loop de resultados.

---

## 📌 Tabla de Contenidos
- [🎯 Objetivo del Proyecto](#-objetivo-del-proyecto)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [📂 Estructura del Repositorio](#-estructura-del-repositorio)
- [🗄️ Esquema de la Base de Datos](#️-esquema-de-la-base-de-datos)
- [🔄 Pipeline MLOps y Ciclo de Vida](#-pipeline-mlops-y-ciclo-de-vida)
- [🚀 Guía de Instalación y Ejecución Local](#-guía-de-instalación-y-ejecución-local)
- [🛡️ Pruebas Automatizadas y Mocks](#️-pruebas-automatizadas-y-mocks)
- [🌐 Despliegue en Producción (Render)](#-despliegue-en-producción-render)
- [🚧 Próximos Pasos (Roadmap)](#-próximos-pasos-roadmap)

---

## 🎯 Objetivo del Proyecto

El objetivo principal es predecir la probabilidad de impago (`loan_status`: 0 = Aprobado, 1 = Denegado/Riesgo) de un cliente solicitante de crédito basándose en su perfil socioeconómico e historial crediticio.

A diferencia de proyectos académicos aislados, esta solución aborda problemas reales de producción:
1. **Model & Data Drift Monitoring**: Monitorización continua de predicciones vs. datos reales gracias a un feedback loop integrado.
2. **Desacoplamiento Total**: Separación entre Frontend (sitio estático), Backend (FastAPI REST API), Capa de Persistencia (PostgreSQL) y Pipeline de ML.
3. **Traceability & Auditing**: Identificación exacta de qué versión de modelo generó cada predicción en la base de datos, insertada con SQL nativo robusto.
4. **CI/CD Automático**: Pipeline integral de integración y despliegue automático hacia entornos productivos bajo reglas estrictas (solo en rama `main`).

---

## 🏗️ Arquitectura del Sistema

```mermaid
flowchart TD
    subgraph Frontend Layer
        UI[💻 Static UI HTML5/JS/Tailwind]
    end

    subgraph Data & Simulation
        CSV["📄 initial_train.csv"]
        SIM["⚡ simulator.py"]
    end

    subgraph API & Serving Layer
        API["🚀 FastAPI Server /predict & /feedback"]
        DAL["🔌 Data Access Layer api/database.py"]
        ML["🧠 Scikit-Learn Pipeline model.pkl"]
    end

    subgraph Database Layer
        DB[("🛢️ PostgreSQL loan_predictions")]
    end

    subgraph CI/CD & Quality
        GA["⚙️ GitHub Actions CI/CD"]
        METRIC["📊 metrics.json Baseline & Version"]
    end

    CSV -->|populate_db.py| DB
    UI -->|HTTP POST| API
    SIM -->|"HTTP POST Batches (/predict & /feedback)"| API
    API --> ML
    API --> DAL
    DAL -->|"Native SQL (RETURNING id)"| DB
    GA -->|"Tests, Lints & CD to Render"| API
```

---

## 📂 Estructura del Repositorio

```text
credit-risk-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Pipeline CI/CD unificado (Lint, Tests y Auto-Deploy a Producción)
├── api/
│   ├── database.py                    # Capa de Acceso a Datos (DAL) optimizada con SQL nativo
│   ├── main.py                        # Servidor FastAPI (/predict y /feedback)
│   ├── model_loader.py                # Carga del modelo en memoria (joblib)
│   └── schemas.py                     # DTOs y validación con Pydantic
├── frontend/
│   └── index.html                     # Interfaz gráfica SPA de alto rendimiento (Tailwind)
├── model/
│   ├── metrics.json                   # Metadatos del modelo (Version, F1, Accuracy)
│   └── model.pkl                      # Pipeline Scikit-Learn empaquetado
├── tests/
│   └── test_api.py                    # Tests unitarios con persistencia a base de datos mockeada
├── training/
│   ├── initial_train.csv              # Dataset de entrenamiento inicial
│   ├── populate_db.py                 # Poblado inicial de la DB PostgreSQL
│   └── train.py                       # Entrenamiento y volcado de métricas básicas
├── Dockerfile                         # Contenedor de producción
├── docker-compose.yml                 # Entorno local
└── simulator.py                       # Simulador de inferencia y feedback en lotes (batch)
```

---

## 🗄️ Esquema de la Base de Datos

Toda la persistencia reside en la tabla **`loan_predictions`**, que soporta de forma robusta la trazabilidad del ciclo de vida predictivo y se actualiza a través de un **Feedback Loop**:

```sql
CREATE TABLE loan_predictions (
    id SERIAL PRIMARY KEY,
    -- [Características socioeconómicas omitidas para brevedad]
    loan_status INT,                               -- Outcome REAL enviado a posteriori vía feedback loop
    model_prediction INT,                          -- Predicción inicial
    prediction_prob FLOAT,                         -- Confianza de la predicción
    data_source VARCHAR(20),                       -- Origen ('training_csv', 'api')
    model_version VARCHAR(30),                     -- Versión exacta del modelo que predijo
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔄 Pipeline MLOps y Ciclo de Vida

### 1. Model Training
Se entrena un pipeline (`Imputación` -> `Escalado` -> `OneHot` -> `Regresión Logística`). Al finalizar, se crea una etiqueta de versión (e.g., `v-20260810-120000`) almacenada en `metrics.json` junto con el `F1 Score`.

### 2. Data Access Layer & Feedback Loop
- **FastAPI Backend (`api/main.py`)**: 
  - Expone el endpoint `/predict` que inserta la predicción mediante **SQL nativo** utilizando `RETURNING id`. Devuelve este ID al cliente.
  - Expone un nuevo endpoint `/feedback` que permite recibir los resultados reales a posteriori para actualizar masivamente los registros pendientes (`loan_status`).
- **Tolerancia a fallos**: Diseñado para soportar fallos de BD sin bloquear las respuestas al usuario de la API.

### 3. CI/CD Pipeline Robusto
Gestionado vía **GitHub Actions**:
- **CI (Continuous Integration)**: En cada commit, ejecuta linting de código y pasa la suite de tests en Pytest.
- **CD (Continuous Deployment)**: Si el commit se integra a la rama `main` y todos los tests pasan, lanza un webhook para un **Deploy Automático a Render**. Jamás despliega código desde ramas de desarrollo.

### 4. Production Traffic Simulator
El script `simulator.py` actúa como un orquestador de estrés de la API:
- Envía bloques de peticiones a `/predict`.
- Captura dinámicamente los `prediction_id` generados por la API.
- Reúne el status real de cada préstamo y ejecuta llamadas periódicas en lote (Batch) hacia el endpoint `/feedback` para enriquecer la base de datos de producción con resultados verdaderos.

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Clonar y Configurar Entorno Virtual
```bash
git clone https://github.com/tu-usuario/credit-risk-mlops.git
cd credit-risk-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Base de Datos
Crea el archivo `.env` en la raíz (o usa docker-compose):
```env
DATABASE_URL=postgresql://usuario:password@localhost:5432/credit_risk_db
```

### 3. Levantar Servicios
```bash
docker-compose up -d --build
uvicorn api.main:app --reload
```

---

## 🛡️ Pruebas Automatizadas y Mocks

Para ejecutar la batería completa de pruebas localmente:
```bash
pytest tests/ -v
```

> [!IMPORTANT]
> **Protección contra contaminación de datos (Database Mocking)**:
> Los tests de la API utilizan un interceptor inteligente con `unittest.mock.patch` sobre `api.main.save_prediction` y `api.main.update_predictions_feedback`. Esto garantiza que los test funcionales se verifiquen sin escribir información "basura" en la PostgreSQL productiva.

---

## 🌐 Despliegue en Producción (Render)

- **PostgreSQL**: Instancia alojada con almacenamiento persistente.
- **Web Service API**: Se despliega de forma totalmente automática por GitHub Actions al hacer merge en `main`.
- **Límites de recursos (Free Tier)**: El simulador incluye retardos estratégicos y un mayor timeout para no sobrecargar el servidor gratuito con peticiones por lotes masivas.

---

## 🚧 Próximos Pasos (Roadmap)

La arquitectura actual ya soporta la ingesta de predicciones y su consolidación con el feedback real. A partir de aquí, las próximas grandes evoluciones del MLOps System son:

- [ ] **Data Drift Detection**: Integrar un mecanismo (en el simulador o cronjob) para comparar periódicamente la distribución estadística de los nuevos clientes recibidos en la API frente a la muestra original de entrenamiento.
- [ ] **Model Drift / Concept Drift**: Detectar degradación de rendimiento. Usando el feedback ya disponible en base de datos, construir alarmas automatizadas si métricas clave (como el `F1-Score` o el `Accuracy` en producción) caen por debajo de los umbrales configurados.
- [ ] **Automated Retraining Pipeline**: Activar el flujo de re-entrenamiento (`train.py`) de forma automática y controlada en respuesta a alertas de Drift.
- [ ] **Dashboarding Analítico**: Exponer los datos recogidos en PostgreSQL mediante un panel de control interactivo (p. ej. Grafana o Metabase) en tiempo real para visibilidad de negocio y data science.
- [ ] **MLflow Tracking Server**: Recuperar MLflow (actualmente purgado) para registrar y persistir de manera profesional todas las ejecuciones (experiments), artefactos, hiperparámetros y métricas en un servidor dedicado en lugar de en archivos JSON locales.