# =====================================================
# Dockerfile - Voz a Texto con Enfoque en Tono Emocional
# Versión: 8.0.0 (Migración WhisperX)
# =====================================================

# Base image con soporte CUDA para NVIDIA GPU
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Metadatos
LABEL maintainer="Christian"
LABEL description="API de transcripción de voz a texto con análisis emocional"
LABEL version="8.0.0"

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
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
# NOTA: La base image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime 
#       ya incluye torch 2.5.1 con CUDA 12.4. NO reinstalar torch.

# Instalar dependencias de Python
# NOTA: La base image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime 
#       ya incluye torch 2.5.1 con CUDA 12.4. NO reinstalar torch.

# Paso 1: Instalar WhisperX y pyannote primero (git repos)
# Esto asegura que si traen dependencias viejas, requirements.txt las actualice después
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git pyannote.audio

# Paso 2: Instalar requisitos estrictos (transformers actualizado, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Paso 3: Forzar actualización de bibliotecas críticas de ML y corregir binarios de Torch
# Reinstalamos torchvision y torchaudio para asegurar compatibilidad con torch 2.5.1+cu124 de la base image
RUN pip install --no-cache-dir --force-reinstall \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.48.1" \
    accelerate \
    sentencepiece \
    protobuf \
    tokenizers \
    httpx \
    psutil

# Paso 4: Verificación exhaustiva de entorno antes de finalizar build
RUN python -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}'); \
    import transformers; print(f'Transformers: {transformers.__version__}'); \
    from transformers import pipeline; print('Pipeline import: OK'); \
    pipe = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english'); print('Pipeline inference: OK')"

# Copiar el código fuente del proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/output /app/models /app/history /app/feedback_data

# Establecer permisos
RUN chmod -R 755 /app

# Puerto de la API FastAPI
EXPOSE 8000

# Health check optimizado
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar la aplicación
# Usar workers=1 para evitar problemas con modelos grandes en memoria
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--loop", "asyncio"]
