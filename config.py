"""
Configuración centralizada para el proyecto de transcripción y diarización.
"""
import os
from pathlib import Path
from typing import Set

# Paths del proyecto - usar ubicación del archivo para estabilidad
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_BASE_DIR = PROJECT_ROOT / "model"
AUDIO_PATH = PROJECT_ROOT / "test.wav"  # Archivo de audio por defecto

# Directorios de datos y salida
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Configuración de audio
TARGET_SR = 16000  # Sample rate objetivo en Hz

# Configuracion de VAD (Voice Activity Detection)
VAD_MODE = 2  # 0-3 (0 menos agresivo, 3 mas agresivo)
VAD_FRAME_MS = 30  # Frame size en ms (10, 20, 30)

# Configuracion de embeddings y diarizacion
WINDOW_SEC = 2.5  # Ventana para embeddings (segundos)
HOP_SEC = 0.75  # Salto entre ventanas (segundos)
CHANGE_SIM_THRESHOLD = 0.82  # Similitud coseno minima para considerar "misma voz"
MIN_SEG_SEC = 0.25  # Ignorar segmentos muy cortos (segundos)

# Configuracion de API FastAPI
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))  # 50 MB default
ALLOWED_MIME: Set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/x-m4a",
    "audio/mp4"
}

# Configuracion de workers
WORKERS = int(os.getenv("WORKERS", "2"))

# Configuracion de CORS - incluye wildcard para desarrollo y archivos locales
CORS_ORIGINS = [
    "*",  # Permitir todos los orígenes (para desarrollo)
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null",  # Para archivos locales (file://)
]

# Configuracion de modelos
# Usar modelo 'small' por defecto para mejor precision con RTX 4060
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny, base, small, medium, large

# Modelos de HuggingFace para analisis emocional
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # Inglés (7 emociones)
AUDIO_EMOTION_MODEL = "superb/wav2vec2-base-superb-er"  # Emociones en audio
SENTIMENT_MODEL = "tabularisai/multilingual-sentiment-analysis"

# Modelos adicionales para precision mejorada
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"  # Traduccion ES → EN
EMOTION_ES_MODEL = "daveni/twitter-xlm-roberta-emotion-es"  # Emociones en español directo

# Configuracion de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Constantes para conversion de audio
PCM16_MAX = 32767  # Valor maximo para audio PCM de 16 bits
VOSK_CHUNK_FRAMES = 4000  # Numero de frames por iteracion en Vosk
VAD_MERGE_GAP_SEC = 0.2  # Gap máximo para fusionar regiones VAD (segundos)

# Límites de seguridad
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION", 600))  # 10 minutos por defecto

# Configuración de Análisis Multi-Modal

# Pesos para fusión de emociones (texto vs audio)
# Valores más altos = más importancia a esa modalidad
EMOTION_WEIGHT_TEXT = float(os.getenv("EMOTION_WEIGHT_TEXT", "0.6"))
EMOTION_WEIGHT_AUDIO = float(os.getenv("EMOTION_WEIGHT_AUDIO", "0.4"))

# Modo de fusión: 'weighted_average', 'max_confidence', 'voting'
FUSION_MODE = os.getenv("FUSION_MODE", "weighted_average")

# Umbral mínimo de confianza para considerar una emoción válida
MIN_EMOTION_CONFIDENCE = float(os.getenv("MIN_EMOTION_CONFIDENCE", "0.3"))

# Configuración de GPU (CUDA)
# Dispositivo para modelos de deep learning
# 'auto' detecta automáticamente, 'cuda' fuerza GPU, 'cpu' fuerza CPU
DEVICE = os.getenv("DEVICE", "auto")


# Mapeo de emociones SIMPLIFICADO (Solo 4 categorias basicas)
# Usuario requiere SOLO: Feliz, Enojado, Triste, Neutral
# Todas las demás emociones se consolidan en estas 4
EMOTION_MAPPING = {
    # ===== FELIZ (Emociones Positivas) =====
    "joy": "feliz",
    "alegría": "feliz",
    "happiness": "feliz",
    "happy": "feliz",
    "hap": "feliz",  # Short form
    "surprise": "feliz",  # Sorpresa → Feliz (positivo)
    "sorpresa": "feliz",
    
    # ===== ENOJADO (Emociones Negativas Activas) =====
    "anger": "enojado",
    "angry": "enojado",
    "ang": "enojado",  # Short form
    "ira": "enojado",
    "disgust": "enojado",  # Disgusto → Enojado (rechazo activo)
    "disgusto": "enojado",
    
    # ===== TRISTE (Emociones Negativas Pasivas) =====
    "sadness": "triste",
    "sad": "triste",  # Short form
    "tristeza": "triste",
    "fear": "triste",  # Miedo → Triste (vulnerabilidad)
    "fea": "triste",  # Short form
    "miedo": "triste",
    
    # ===== NEUTRAL =====
    "neutral": "neutral",
    "neu": "neutral",  # Short form
    "others": "neutral",  # OTROS → Neutral
    "otros": "neutral",
    "other": "neutral",
    "unknown": "neutral",
}

#suavizado temporal 
TEMPORAL_SMOOTHING_ENABLED = True
TEMPORAL_SMOOTHING_WINDOW = 3 #ventana de segmentos
TEMPORAL_SMOOTHING_ALPHA = 0.6 #peso del segmento actual vs el historico

#supresion nutral (mas agresiva)
NEUTRAL_SUPPRESSION_FACTOR = 0.15 #REDUCIR NEUTRAL SEVERAMENTE
ACTIVE_EMOTION_BOOST= 1.8 #POTENCIAR EMOCIONES ACTIVAS
MIN_EMOTION_TO_PROMOTE = 0.05 #MINIMO PARA PROMOVER SOBRE NEUTRAL

#CIRCUIT BREAKERS
CB_FAILURE_THRESHOLD = 5 #UMBRAL DE FALLA
CB_RESET_TIMEOUT = 30.0 #TIEMPO DE RESETEO

#VALIDACIOON DE AUDIO 
AUDIO_MIN_DURATION_SEC = 0.5
AUDIO_MAX_FILE_SIZE_MB = 100
AUDIO_MIN_RMS_THRESHOLD = 0.001 #detecta el silencio