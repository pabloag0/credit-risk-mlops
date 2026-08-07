# Credit Risk MLOps

Este es un proyecto completo de MLOps para predecir el riesgo de crédito (Credit Risk) utilizando un modelo de Machine Learning empaquetado, una API REST (FastAPI) y una interfaz de usuario (Streamlit), todo orquestado con Docker.

## 🚀 Arquitectura del Proyecto

El proyecto sigue una arquitectura desacoplada para facilitar el despliegue y la integración continua:

- **Model Training (`training/`)**: Pipeline de Scikit-Learn (Imputación, Escalado, One-Hot Encoding, Regresión Logística) que exporta el modelo encapsulado.
- **Backend API (`api/`)**: API REST construida con FastAPI que expone el modelo y valida de forma estricta los datos de entrada usando esquemas de Pydantic.
- **Frontend UI (`app/`)**: Aplicación interactiva construida con Streamlit que consume la API REST, pensada para usuarios de negocio.

## 📂 Estructura de Directorios

```text
credit-risk-mlops/
├── api/                   # Código de FastAPI (schemas.py, model_loader.py, main.py)
├── app/                   # Frontend de Streamlit (streamlit_app.py)
├── model/                 # Carpeta donde se guarda el artefacto del modelo (model.pkl)
├── training/              # Scripts de partición, entrenamiento y datos crudos (CSV)
├── Dockerfile             # Receta para construir la API
├── Dockerfile.frontend    # Receta para construir Streamlit
├── docker-compose.yml     # Orquestador multicontenedor
└── requirements.txt       # Dependencias de Python
```

## 🛠️ Cómo arrancar el proyecto

### 1. Entrenar el Modelo
Antes de poder arrancar la API, necesitas generar el artefacto del modelo (`model.pkl`).
Asegúrate de estar en tu entorno de trabajo con dependencias (como `scikit-learn` y `pandas`) y ejecuta:

```bash
python training/train.py
```
*Esto leerá `initial_train.csv`, entrenará el pipeline completo de Scikit-Learn y guardará el resultado en la carpeta `model/`.*

### 2. Levantar los servicios con Docker
Una vez tengas el modelo generado, puedes levantar toda la infraestructura web de un solo golpe usando Docker Compose:

```bash
docker-compose up --build -d
```

### 3. Acceso a las aplicaciones
- **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501)
- **API Documentación (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 💡 Notas de Diseño Arquitectónico

Para garantizar el principio **KISS (Keep It Simple, Stupid)** y agilizar la integración continua (CI/CD), el modelo se exporta usando el formato nativo `.pkl` (con `joblib`) en lugar de formatos de despliegue avanzados como `ONNX`. 

Esto permite que todo el pipeline de preprocesamiento de datos (medias, desviaciones y codificación categórica) viaje encapsulado de forma transparente en un solo archivo, permitiendo a la API consumir directamente JSONs validados sin necesidad de reescribir lógica matemática de limpieza de datos en el servidor web.