# Credit Risk MLOps

Este es un proyecto completo de MLOps para predecir el riesgo de crédito (Credit Risk) utilizando un modelo de Machine Learning empaquetado, una API REST (FastAPI) y una interfaz de usuario web estática (HTML/JS/Tailwind).

## 🚀 Arquitectura del Proyecto

El proyecto sigue una arquitectura desacoplada para facilitar el despliegue y la integración continua:

- **Model Training (`training/`)**: Pipeline de Scikit-Learn (Imputación, Escalado, One-Hot Encoding, Regresión Logística) con registro en MLflow.
- **Backend API (`api/`)**: API REST construida con FastAPI que expone el modelo y valida de forma estricta los datos de entrada usando esquemas de Pydantic. Permite CORS para ser consumida desde cualquier dominio.
- **Frontend UI (`frontend/`)**: Aplicación web puramente estática (HTML5 + Vanilla JS + Tailwind CSS) lista para ser desplegada en plataformas como GitHub Pages o Render.

## 📂 Estructura de Directorios

```text
credit-risk-mlops/
├── api/                   # Código de FastAPI (schemas.py, model_loader.py, main.py)
├── frontend/              # Frontend estático (index.html)
├── model/                 # Artefacto del modelo (model.pkl) y métricas base (metrics.json)
├── tests/                 # Tests unitarios y Quality Gates (pytest)
├── training/              # Scripts de partición, entrenamiento y datos crudos (CSV)
├── Dockerfile             # Receta para construir la API
├── docker-compose.yml     # Orquestador para desarrollo local
└── requirements.txt       # Dependencias de Python
```

## 🛠️ Cómo arrancar el proyecto

### 1. Entrenar el Modelo (Local)
Antes de poder arrancar la API, necesitas generar el artefacto del modelo (`model.pkl`).
Asegúrate de estar en tu entorno virtual y ejecuta:

```bash
python training/train.py
```
*Esto entrenará el pipeline completo de Scikit-Learn, guardará el resultado en la carpeta `model/` y registrará las métricas en MLflow.*

### 2. Levantar la API localmente
Puedes levantar la API de desarrollo usando Docker Compose:

```bash
docker-compose up --build -d
```

### 3. Acceso a las aplicaciones
- **Frontend Web**: Simplemente abre `frontend/index.html` en tu navegador web. No requiere servidor en local.
- **API Documentación (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 💡 Notas de Diseño Arquitectónico

Para garantizar el principio **KISS (Keep It Simple, Stupid)** y agilizar la integración continua (CI/CD), el modelo se exporta usando el formato nativo `.pkl` (con `joblib`) en lugar de formatos de despliegue avanzados como `ONNX`. 

Esto permite que todo el pipeline de preprocesamiento de datos (medias, desviaciones y codificación categórica) viaje encapsulado de forma transparente en un solo archivo, permitiendo a la API consumir directamente JSONs validados sin necesidad de reescribir lógica matemática de limpieza de datos en el servidor web.