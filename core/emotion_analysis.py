"""
Módulo unificado de análisis emocional multi-modal.
Combina análisis de texto (español e inglés) y audio para mayor precisión.
"""
import logging
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from transformers import pipeline

from config import (
    TEXT_EMOTION_MODEL,
    AUDIO_EMOTION_MODEL,
    EMOTION_ES_MODEL,
    EMOTION_WEIGHT_TEXT,
    EMOTION_WEIGHT_AUDIO,
    MIN_EMOTION_CONFIDENCE,
    FUSION_MODE,
    DEVICE,
    EMOTION_MAPPING,
    TARGET_SR
)

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Resultado de análisis emocional."""
    emotions: Dict[str, float]  # Mapa de emoción -> score
    top_emotion: str
    top_score: float
    confidence: float
    source: str  # 'text', 'audio', 'text_es', 'text_en', 'multimodal'


def get_device() -> str:
    """Determina el dispositivo a usar (GPU/CPU)."""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def normalize_emotion_label(label: str) -> str:
    """Normaliza etiquetas de emoción al español."""
    label_lower = label.lower()
    return EMOTION_MAPPING.get(label_lower, label_lower)


@lru_cache(maxsize=1)
def load_text_emotion_en():
    """Carga modelo de emociones en inglés (j-hartmann)."""
    device_num = 0 if get_device() == "cuda" else -1
    logger.info(f"Cargando modelo de emociones en inglés...")
    return pipeline(
        "text-classification",
        model=TEXT_EMOTION_MODEL,
        return_all_scores=True,
        device=device_num
    )


@lru_cache(maxsize=1)
def load_text_emotion_es():
    """Carga modelo de emociones en español (daveni)."""
    device_num = 0 if get_device() == "cuda" else -1
    logger.info(f"Cargando modelo de emociones en español...")
    return pipeline(
        "text-classification",
        model=EMOTION_ES_MODEL,
        return_all_scores=True,
        device=device_num
    )


@lru_cache(maxsize=1)
def load_audio_emotion():
    """Carga modelo de emociones en audio (wav2vec2)."""
    device_num = 0 if get_device() == "cuda" else -1
    logger.info(f"Cargando modelo de emociones de audio...")
    return pipeline(
        "audio-classification",
        model=AUDIO_EMOTION_MODEL,
        device=device_num
    )


def analyze_text_emotion_en(text: str) -> EmotionResult:
    """
    Analiza emoción en texto inglés.
    
    Args:
        text: Texto en inglés
        
    Returns:
        EmotionResult con emociones detectadas
    """
    if not text or not text.strip():
        return EmotionResult(
            emotions={},
            top_emotion="neutral",
            top_score=0.0,
            confidence=0.0,
            source="text_en"
        )
    
    try:
        classifier = load_text_emotion_en()
        results = classifier(text[:512])[0]  # Limitar longitud
        
        emotions = {}
        for r in results:
            label = normalize_emotion_label(r["label"])
            emotions[label] = float(r["score"])
        
        top = max(results, key=lambda x: x["score"])
        return EmotionResult(
            emotions=emotions,
            top_emotion=normalize_emotion_label(top["label"]),
            top_score=float(top["score"]),
            confidence=float(top["score"]),
            source="text_en"
        )
    except Exception as e:
        logger.error(f"Error en análisis de emoción texto EN: {e}")
        return EmotionResult({}, "neutral", 0.0, 0.0, "text_en")


def analyze_text_emotion_es(text: str) -> EmotionResult:
    """
    Analiza emoción en texto español directamente.
    
    Args:
        text: Texto en español
        
    Returns:
        EmotionResult con emociones detectadas
    """
    if not text or not text.strip():
        return EmotionResult(
            emotions={},
            top_emotion="neutral",
            top_score=0.0,
            confidence=0.0,
            source="text_es"
        )
    
    try:
        classifier = load_text_emotion_es()
        results = classifier(text[:512])[0]
        
        emotions = {}
        for r in results:
            label = normalize_emotion_label(r["label"])
            emotions[label] = float(r["score"])
        
        top = max(results, key=lambda x: x["score"])
        return EmotionResult(
            emotions=emotions,
            top_emotion=normalize_emotion_label(top["label"]),
            top_score=float(top["score"]),
            confidence=float(top["score"]),
            source="text_es"
        )
    except Exception as e:
        logger.error(f"Error en análisis de emoción texto ES: {e}")
        return EmotionResult({}, "neutral", 0.0, 0.0, "text_es")


def analyze_audio_emotion(audio_path: str) -> EmotionResult:
    """
    Analiza emoción en audio usando wav2vec2.
    
    Args:
        audio_path: Ruta al archivo de audio WAV
        
    Returns:
        EmotionResult con emociones detectadas
    """
    try:
        classifier = load_audio_emotion()
        results = classifier(audio_path, top_k=None)
        
        emotions = {}
        for r in results:
            label = normalize_emotion_label(r["label"])
            emotions[label] = float(r["score"])
        
        top = max(results, key=lambda x: x["score"])
        return EmotionResult(
            emotions=emotions,
            top_emotion=normalize_emotion_label(top["label"]),
            top_score=float(top["score"]),
            confidence=float(top["score"]),
            source="audio"
        )
    except Exception as e:
        logger.error(f"Error en análisis de emoción audio: {e}")
        return EmotionResult({}, "neutral", 0.0, 0.0, "audio")


def fuse_emotions(
    text_result: EmotionResult,
    audio_result: EmotionResult,
    mode: str = FUSION_MODE
) -> EmotionResult:
    """
    Fusiona resultados de análisis de texto y audio.
    
    Args:
        text_result: Resultado de análisis de texto
        audio_result: Resultado de análisis de audio
        mode: Modo de fusión ('weighted_average', 'max_confidence', 'voting')
        
    Returns:
        EmotionResult fusionado
    """
    if mode == "weighted_average":
        return _fuse_weighted_average(text_result, audio_result)
    elif mode == "max_confidence":
        return _fuse_max_confidence(text_result, audio_result)
    elif mode == "voting":
        return _fuse_voting(text_result, audio_result)
    else:
        return _fuse_weighted_average(text_result, audio_result)


def _fuse_weighted_average(
    text_result: EmotionResult,
    audio_result: EmotionResult
) -> EmotionResult:
    """
    Fusión por promedio ponderado con AJUSTE DE SENSIBILIDAD.
    Reduce la predominancia de 'neutral' y potencia emociones activas.
    """
    all_emotions = set(text_result.emotions.keys()) | set(audio_result.emotions.keys())
    fused_raw = {}
    
    # 1. Promedio Ponderado Inicial
    for emotion in all_emotions:
        text_score = text_result.emotions.get(emotion, 0.0)
        audio_score = audio_result.emotions.get(emotion, 0.0)
        fused_raw[emotion] = (
            text_score * EMOTION_WEIGHT_TEXT +
            audio_score * EMOTION_WEIGHT_AUDIO
        )
        
        
    # 2. Ajuste de Sensibilidad (ULTRA-AGGRESSIVE Neutral Suppression)
    # Usuario requiere prácticamente eliminar neutral/other de los resultados
    BOOST_FACTOR = 1.6  # Potenciar emociones FUERTEMENTE (era 1.0, luego 1.3)
    NEUTRAL_DAMP = 0.25  # Castigar neutral SEVERAMENTE (era 1.0, luego 0.5)
    
    adjusted_scores = {}
    total_score = 0.0
    
    for emo, score in fused_raw.items():
        # Castigar tanto 'neutral' como 'other'
        if emo.lower() in ["neutral", "other", "others", "neu"]:
            new_score = score * NEUTRAL_DAMP
        else:
            new_score = score * BOOST_FACTOR
        
        adjusted_scores[emo] = new_score
        total_score += new_score
        
    # 3. Renormalización
    final_emotions = {}
    if total_score > 0:
        for emo, s in adjusted_scores.items():
            final_emotions[emo] = round(s / total_score, 4)
    else:
        final_emotions = fused_raw # Fallback

    if not final_emotions:
        return EmotionResult({}, "neutral", 0.0, 0.0, "multimodal")
    
    top_emotion = max(final_emotions, key=final_emotions.get)
    top_score = final_emotions[top_emotion]

    # --- LÓGICA "FORZAR EMOCIÓN" (ULTRA-AGGRESSIVE) ---
    # Si la ganadora es 'neutral' u 'other', buscamos la siguiente mejor opción.
    # UMBRAL MUY BAJO: Cualquier emoción con >8% será promovida sobre neutral
    if top_emotion.lower() in ["neutral", "other", "others", "neu"]:
        # Filtramos emociones que NO sean neutral/other y tengan mínimo 8% de presencia
        candidates = {
            k: v for k, v in final_emotions.items() 
            if k.lower() not in ["neutral", "other", "others", "neu"] and v > 0.08
        }
        
        if candidates:
            # Encontramos una alternativa válida - SIEMPRE la promovemos
            alt_emotion = max(candidates, key=candidates.get)
            alt_score = candidates[alt_emotion]
            
            # Promocionamos la alternativa
            top_emotion = alt_emotion
            top_score = alt_score
    # -----------------------------------------------
    
    # Confianza basada en concordancia entre modalidades
    confidence = _calculate_confidence(text_result, audio_result)
    
    return EmotionResult(
        emotions=final_emotions,
        top_emotion=top_emotion,
        top_score=top_score,
        confidence=confidence,
        source="multimodal"
    )


def _fuse_max_confidence(
    text_result: EmotionResult,
    audio_result: EmotionResult
) -> EmotionResult:
    """Selecciona el resultado con mayor confianza."""
    if text_result.confidence >= audio_result.confidence:
        return EmotionResult(
            emotions=text_result.emotions,
            top_emotion=text_result.top_emotion,
            top_score=text_result.top_score,
            confidence=text_result.confidence,
            source="multimodal"
        )
    else:
        return EmotionResult(
            emotions=audio_result.emotions,
            top_emotion=audio_result.top_emotion,
            top_score=audio_result.top_score,
            confidence=audio_result.confidence,
            source="multimodal"
        )


def _fuse_voting(
    text_result: EmotionResult,
    audio_result: EmotionResult
) -> EmotionResult:
    """Fusión por votación (si ambos coinciden, mayor confianza)."""
    if text_result.top_emotion == audio_result.top_emotion:
        # Ambos coinciden - alta confianza
        avg_score = (text_result.top_score + audio_result.top_score) / 2
        fused = _fuse_weighted_average(text_result, audio_result)
        fused.confidence = min(1.0, avg_score + 0.2)  # Bonus por concordancia
        return fused
    else:
        # No coinciden - usar promedio ponderado
        return _fuse_weighted_average(text_result, audio_result)


def _calculate_confidence(
    text_result: EmotionResult,
    audio_result: EmotionResult
) -> float:
    """Calcula confianza basada en concordancia entre modalidades."""
    if text_result.top_emotion == audio_result.top_emotion:
        # Alta concordancia
        return min(1.0, (text_result.confidence + audio_result.confidence) / 2 + 0.15)
    else:
        # Baja concordancia - tomar el promedio sin bonus
        return (text_result.confidence + audio_result.confidence) / 2


def analyze_segment_multimodal(
    text_es: str,
    text_en: str,
    audio_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analiza un segmento usando múltiples modalidades.
    
    Args:
        text_es: Texto original en español
        text_en: Texto traducido al inglés
        audio_path: Ruta opcional al fragmento de audio
        
    Returns:
        Diccionario con análisis completo
    """
    # Análisis de texto en español
    result_es = analyze_text_emotion_es(text_es)
    
    # Análisis de texto en inglés (traducido)
    result_en = analyze_text_emotion_en(text_en)
    
    # Combinar análisis de texto (promedio ES + EN)
    text_combined = _fuse_weighted_average(result_es, result_en)
    text_combined.source = "text"
    
    # Análisis de audio si está disponible
    if audio_path:
        result_audio = analyze_audio_emotion(audio_path)
        final_result = fuse_emotions(text_combined, result_audio)
    else:
        final_result = text_combined
    
    return {
        "text_es": {
            "top_emotion": result_es.top_emotion,
            "top_score": result_es.top_score,
            "emotions": result_es.emotions
        },
        "text_en": {
            "top_emotion": result_en.top_emotion,
            "top_score": result_en.top_score,
            "emotions": result_en.emotions
        },
        "audio": {
            "top_emotion": result_audio.top_emotion if audio_path else None,
            "top_score": result_audio.top_score if audio_path else 0.0,
            "emotions": result_audio.emotions if audio_path else {}
        } if audio_path else None,
        "multimodal": {
            "top_emotion": final_result.top_emotion,
            "top_score": final_result.top_score,
            "emotions": final_result.emotions,
            "confidence": final_result.confidence
        }
    }


def compute_weighted_emotion_score(
    segments: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calcula el score emocional ponderado por duración.
    
    Args:
        segments: Lista de segmentos con 'duration' y 'emotion' o 'multimodal'
        
    Returns:
        Diccionario con score ponderado por emoción
    """
    emotion_weights = {}
    total_duration = 0.0
    
    for seg in segments:
        duration = seg.get("end", 0) - seg.get("start", 0)
        if duration <= 0:
            continue
        
        total_duration += duration
        
        # Obtener emociones del segmento
        emotions = seg.get("multimodal", {}).get("emotions", {})
        if not emotions:
            emotions = seg.get("emotions", {})
        
        for emotion, score in emotions.items():
            if emotion not in emotion_weights:
                emotion_weights[emotion] = 0.0
            emotion_weights[emotion] += score * duration
    
    # Normalizar por duración total
    if total_duration > 0:
        for emotion in emotion_weights:
            emotion_weights[emotion] /= total_duration
    
    # Ordenar por score
    sorted_emotions = dict(
        sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
    )
    
    return sorted_emotions
