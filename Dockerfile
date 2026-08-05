# 1. Usamos una imagen oficial de Python ligera (puedes ajustar la versión si usas otra)
FROM python:3.10-slim

# 2. Le decimos a Docker que trabaje dentro de la carpeta /app en el contenedor
WORKDIR /app

# 3. Copiamos el archivo de dependencias y las instalamos
# (Se hace antes de copiar el código para aprovechar el caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiamos las carpetas estrictamente necesarias para que la API funcione
COPY api/ ./api/
COPY model/ ./model/

# 5. Exponemos el puerto 8000 (el mismo que usa Uvicorn)
EXPOSE 8000

# 6. El comando que se ejecutará al encender el contenedor
# El --host 0.0.0.0 es obligatorio en Docker para que acepte conexiones desde tu Mac
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]