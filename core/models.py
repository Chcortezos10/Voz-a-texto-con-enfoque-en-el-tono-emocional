"""
Gestión y carga de modelos de ML con caché.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from vosk import Model
from resemblyzer import VoiceEncoder

from config import (
    MODEL_BASE_DIR,
    WHISPER_MODEL,
    TEXT_EMOTION_MODEL,
    AUDIO_EMOTION_MODEL,
    SENTIMENT_MODEL
)


def find_vosk_model(base_dir: Path = MODEL_BASE_DIR) -> str:
    """
    Busca y retorna la ruta al modelo Vosk.

    Args:
        base_dir: Directorio base donde buscar modelos

    Returns:
        Ruta al modelo Vosk

    Raises:
        FileNotFoundError: Si no se encuentra el modelo
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Carpeta base de modelos no encontrada: {base_dir}")

    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No se encontraron subcarpetas de modelo en: {base_dir}")

    # Buscar carpeta que contenga "vosk" o "model" en el nombre
    for c in candidates:
        name = c.name.lower()
        if "vosk" in name or "model" in name:
            return str(c)

    # Si no encuentra ninguna, retornar la primera
    return str(candidates[0])


@lru_cache(maxsize=1)
def load_vosk_model() -> Model:
    """
    Carga el modelo Vosk con caché.

    Returns:
        Modelo Vosk cargado

    Raises:
        FileNotFoundError: Si no se encuentra el modelo
    """
    model_path = find_vosk_model()
    return Model(model_path)


@lru_cache(maxsize=1)
def load_voice_encoder() -> VoiceEncoder:
    """
    Carga el encoder de voz Resemblyzer con caché.

    Returns:
        VoiceEncoder inicializado
    """
    return VoiceEncoder()


@lru_cache(maxsize=1)
def load_whisper_models() -> Dict[str, Any]:
    """
    Carga modelos de Whisper y análisis de emociones/sentimientos.
    Esta función es para la API FastAPI.

    Returns:
        Diccionario con todos los modelos cargados
    """
    import whisper
    from transformers import pipeline
    from pysentimiento import create_analyzer

    whisper_model = whisper.load_model(WHISPER_MODEL)

    text_emotion = pipeline(
        "text-classification",
        model=TEXT_EMOTION_MODEL,
        return_all_scores=True
    )

    audio_emotion = pipeline(
        "audio-classification",
        model=AUDIO_EMOTION_MODEL
    )

    sentiment = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL
    )

    pysentimiento_emotion_es = create_analyzer(task="emotion", lang="es")

    return {
        "whisper": whisper_model,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "sentiment": sentiment,
        "pysentimiento_es": pysentimiento_emotion_es
    }


@lru_cache(maxsize=1)
def load_whisper_model():
    """
    Carga solo el modelo Whisper (para Streamlit simple).

    Returns:
        Modelo Whisper cargado
    """
    import whisper
    return whisper.load_model(WHISPER_MODEL)