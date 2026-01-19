# =====================================================
# Dockerfile - Voz a Texto con Enfoque en Tono Emocional
# Versión: 5.0.0 (Optimizada con gestión de memoria)
# =====================================================

# Base image ligera para CPU (compatible con AMD Ryzen, Intel, etc.)
# Para GPU NVIDIA, usar: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM python:3.11-slim

# Metadatos
LABEL maintainer="Christian"
LABEL description="API de transcripción de voz a texto con análisis emocional"
LABEL version="5.0.0"

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Variables de entorno del proyecto (optimizado para CPU)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Optimizaciones para CPU
    OMP_NUM_THREADS=4 \
    # Configuración del servidor
    HOST=0.0.0.0 \
    PORT=8000

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir httpx psutil

# Copiar el código fuente del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/output /app/models /app/history

# Establecer permisos
RUN chmod -R 755 /app

# Puerto de la API FastAPI
EXPOSE 8000

# Health check optimizado
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar la aplicación
# Usar workers=1 para evitar problemas con modelos grandes en memoria
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
