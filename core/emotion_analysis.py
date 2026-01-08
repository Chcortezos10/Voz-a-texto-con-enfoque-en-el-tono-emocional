"""
Módulo unificado de análisis emocional multi-modal.
Combina análisis de texto (español e inglés) y audio para mayor precisión.
"""
import logging
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import torch
import numpy as np
from transformers import pipeline

try:
    import librosa
    librosa_disponible = True
except ImportError:
    librosa_disponible = False
    logger.warning("Librosa no está instalado No se podr realizar analisis de emociones audio.")

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

class TemporalEmotionState:
    """Estado para suavizado temporal de emociones."""
    
    def __init__(self):
        self.history: List[Dict[str, float]] = []
    
    def add(self, emotions: Dict[str, float]) -> None:
        self.history.append(emotions.copy())
    
    def update(self, emotions: Dict[str, float]) -> None:
        """Alias for add method."""
        self.add(emotions)
    
    def reset(self) -> None:
        self.history = []
    
    def get_smoothed(self, current: Dict[str, float], alpha: float = 0.6) -> Dict[str, float]:
        """Retorna las emociones suavizadas con el historial."""
        if not self.history:
            return current
        
        all_emotions = set(current.keys())
        for hist in self.history:
            all_emotions.update(hist.keys())
        
        smoothed = {}
        for emotion in all_emotions:
            current_val = current.get(emotion, 0.0)
            hist_vals = [h.get(emotion, 0.0) for h in self.history]
            if hist_vals:
                hist_avg = sum(hist_vals) / len(hist_vals)
                smoothed[emotion] = alpha * current_val + (1 - alpha) * hist_avg
            else:
                smoothed[emotion] = current_val
        return smoothed

@dataclass
class ProsodyFeatures:
    """Características prosódicas del audio."""
    pitch_mean: float = 150.0
    pitch_std: float = 20.0
    pitch_range: float = 50.0
    energy_mean: float = 0.05
    energy_std: float = 0.02
    energy_range: float = 0.0
    speech_rate: float = 3.0
    pause_ratio: float = 0.2

class ProsodyAnalyzer:
# analiza las caracteristicas prosódicas del audio para la inferencia de las emociones
#complementa lo que ya se tenia en el analisis de texto y modelo de audio 
    PROSODY_RULES={"feliz": {"pitch": "high", "pitch_var": "high", "energy": "high", "rate": "fast"},
        "enojado": {"pitch": "high", "pitch_var": "low", "energy": "high", "rate": "fast"},
        "triste": {"pitch": "low", "pitch_var": "low", "energy": "low", "rate": "slow"},
        "neutral": {"pitch": "medium", "pitch_var": "low", "energy": "medium", "rate": "medium"}}
    
    def __init__(self,sr:int=16000):
        self.sr=sr
        self.pitch_thresholds ={"low":120,"high":240}
        self.energy_thresholds={"low":0.02,"high":0.08}
        self.rate_thresholds={"slow":2.0,"fast":4.5}
    
    def extract_features(self, audio: np.ndarray, sr: int = None) -> ProsodyFeatures:
        """Extrae características prosódicas del audio."""
        if sr is not None:
            self.sr = sr
        
        if audio is None or len(audio) < 1:
            return ProsodyFeatures()
        try:
            # Pitch usando pyin
            fo, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            fo_valid = fo[~np.isnan(fo)]
            if len(fo_valid) == 0:
                fo_valid = np.array([150.0])

            pitch_mean = float(np.mean(fo_valid))
            pitch_std = float(np.std(fo_valid))
            pitch_range = float(np.max(fo_valid) - np.min(fo_valid))

            # Energía del audio usando librosa 
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            energy_range = float(np.max(rms) - np.min(rms))

            # Velocidad de habla
            energy_diff = np.abs(np.diff(rms))
            threshold = np.mean(energy_diff) + np.std(energy_diff)
            speech_onsets = np.sum(energy_diff > threshold)
            duration = len(audio) / self.sr
            speech_rate = speech_onsets / max(duration, 0.1)

            # Ratio de pausas en el audio
            silence_threshold = energy_mean * 0.3
            silence_frames = np.sum(rms < silence_threshold)
            pause_ratio = silence_frames / max(len(rms), 1)
    
            return ProsodyFeatures(
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                pitch_range=pitch_range,
                energy_mean=energy_mean,
                energy_std=energy_std,
                energy_range=energy_range,
                speech_rate=speech_rate,
                pause_ratio=pause_ratio
            )
        except Exception as e:
            logger.error(f"Error al extraer características prosódicas: {e}")
            return ProsodyFeatures()

    def infer_emotion(self, features: ProsodyFeatures) -> tuple:
        """Infiere la emoción en base a las características prosódicas del audio."""
        scores = {}
        for emotion, rules in self.PROSODY_RULES.items():
            score = 0.0
            checks = 0
            
            # Evaluar pitch
            if rules["pitch"] == "high" and features.pitch_mean > self.pitch_thresholds["high"]:
                score += 1
            elif rules["pitch"] == "low" and features.pitch_mean < self.pitch_thresholds["low"]:
                score += 1
            elif rules["pitch"] == "medium" and self.pitch_thresholds["low"] <= features.pitch_mean <= self.pitch_thresholds["high"]:
                score += 1
            checks += 1

            # Evaluar variabilidad del pitch
            if rules["pitch_var"] == "high" and features.pitch_std > 30:
                score += 1
            elif rules["pitch_var"] == "low" and features.pitch_std < 30:
                score += 1
            checks += 1
            
            # Evaluar energía
            if rules["energy"] == "high" and features.energy_mean > self.energy_thresholds["high"]:
                score += 1
            elif rules["energy"] == "low" and features.energy_mean < self.energy_thresholds["low"]:
                score += 1
            elif rules["energy"] == "medium" and self.energy_thresholds["low"] <= features.energy_mean <= self.energy_thresholds["high"]:
                score += 1
            checks += 1

            # Evaluar velocidad de habla
            if rules["rate"] == "fast" and features.speech_rate > self.rate_thresholds["fast"]:
                score += 1
            elif rules["rate"] == "slow" and features.speech_rate < self.rate_thresholds["slow"]:
                score += 1
            checks += 1

            scores[emotion] = score / max(checks, 1)

        if not scores:
            return "neutral", 0.5
        top_emotion = max(scores, key=scores.get)
        return top_emotion, scores[top_emotion]

_prosody_analyzer:Optional[ProsodyAnalyzer]=None

def get_prosody_analyzer(sr:int = 16000)->ProsodyAnalyzer:
    #obtine o crea las instancias del analizador de prosodico
    global _prosody_analyzer
    if _prosody_analyzer is None:
        _prosody_analyzer=ProsodyAnalyzer(sr=sr)
    return _prosody_analyzer

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
    BOOST_FACTOR = 2.0  # Potenciar emociones FUERTEMENTE 
    NEUTRAL_DAMP = 0.1  # Castigar neutral SEVERAMENTE
    CONFIDENCE_THRESHOLD = 0.15  # Umbral de minimo de confianza
    
    adjusted_scores = {}
    total_score = 0.0
    
    for emo, score in fused_raw.items():
        emo_lower = emo.lower()
        # Castigar tanto 'neutral' como 'other'
        if emo_lower in ["neutral", "other", "others", "neu"]:
            new_score = score * NEUTRAL_DAMP
        else:
            #boost proporcional a la confianza original
            boost = BOOST_FACTOR *(1+score)
            new_score = score * boost

        adjusted_scores[emo] = new_score
        total_score += new_score
        
    # 3. Renormalización
    final_emotions = {}
    if total_score > 0:
        for emo, s in adjusted_scores.items():
            final_emotions[emo] = round(s / total_score, 4)
    else:
        final_emotions = {"neutral": 1.0} # Fallback

    if not final_emotions:
        return EmotionResult({}, "neutral", 0.0, 0.0, "multimodal")
    
    top_emotion = max(final_emotions, key=final_emotions.get)
    top_score = final_emotions[top_emotion]

    # Logica para forzar emociones
    # Si la ganadora es 'neutral' u 'other', buscamos la siguiente mejor opción.
    # UMBRAL MUY BAJO: Cualquier emoción con >8% será promovida sobre neutral
    if top_emotion.lower() in ["neutral", "other", "others", "neu"]:
        # Filtramos emociones que NO sean neutral/other y tengan mínimo 8% de presencia
        candidates = {
            k: v for k, v in final_emotions.items() 
            if k.lower() not in ["neutral", "other", "others", "neu"] and v > CONFIDENCE_THRESHOLD
        }
        
        if candidates:
            # Encontramos una alternativa válida - SIEMPRE la promovemos
            top_emotion = max(candidates, key=candidates.get)
            top_score = candidates[top_emotion]

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


class TemporalEmotionAnalyzer:

    #analizador temporal de emociones con suavizado en el analisis prosodico
    def __init__(self, use_prosody: bool = True):
        self.history:List[Dict[str,float]] = []
        self.segment_count = 0
        self.use_prosody = use_prosody
        self._prosody_analyzer = get_prosody_analyzer() if use_prosody else None
        self._temporal_state = TemporalEmotionState()
    
    def analyze_segment(
        self,
        text_es: str,
        text_en: str = "",
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr:int = 16000,
        audio_weight: float = 0.4,
        apply_smoothing: bool = True
    ) -> EmotionResult:

        results = []
        weights = []

        # Análisis de texto en español
        result_es = analyze_text_emotion_es(text_es)
        if result_es:
            results.append(result_es)
            weights.append(0.35)
        
        # Análisis de texto en inglés
        if text_en and text_en.strip():
            result_en = analyze_text_emotion_en(text_en)
            if result_en:
                results.append(result_en)
                weights.append(0.25)

        # Análisis de audio
        if audio_path and audio_weight > 0:
            try:
                audio_result = analyze_audio_emotion(audio_path)
                if audio_result:
                    results.append(audio_result)
                    weights.append(audio_weight * 0.6)
            except Exception as e:
                logger.error(f"Error al analizar el audio: {e}")

        # Análisis prosódico 
        if self.use_prosody and audio_array is not None and len(audio_array) > 0:
            try:
                prosody_features = self._prosody_analyzer.extract_features(audio_array, sr)
                prosody_emo, prosody_conf = self._prosody_analyzer.infer_emotion(prosody_features)
                prosody_emotions = {prosody_emo: prosody_conf}
                for emo in ["feliz", "enojado", "triste", "neutral"]:
                    if emo not in prosody_emotions:
                        prosody_emotions[emo] = (1 - prosody_conf) / 3

                prosody_result = EmotionResult(
                    emotions=prosody_emotions,
                    top_emotion=prosody_emo,
                    top_score=prosody_conf,
                    confidence=prosody_conf,
                    source="prosody"
                )
                results.append(prosody_result)
                weights.append(audio_weight * 0.4)
            except Exception as e:
                logger.error(f"Error al analizar el audio: {e}")

        #fusion de resultados
        if not results:
            return EmotionResult(
                emotions={"neutral": 1.0},
                top_emotion="neutral",
                top_score=0.5,
                confidence=0.0,
                source="fallback"
            )
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calcular emociones ponderadas
        all_emotions = set()
        for r in results:
            all_emotions.update(r.emotions.keys())

        fused_emotions = {}
        for emo in all_emotions:
            weight_sum = 0.0
            for res, weight in zip(results, weights):
                weight_sum += res.emotions.get(emo, 0.0) * weight
            fused_emotions[emo] = weight_sum
        
        # Aplicamos los ajustes de sensibilidad
        fused_emotions = self._apply_sensitivity_adjustments(fused_emotions)
        
        # Normalizar 
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: round(v/total, 4) for k, v in fused_emotions.items()}

        # Suavizado temporal 
        if apply_smoothing:
            fused_emotions = self._temporal_state.get_smoothed(fused_emotions, alpha=0.7)

        # Guardar en el historial
        self._temporal_state.update(fused_emotions)
        self.history.append(fused_emotions.copy())
        self.segment_count += 1

        # Determinamos el top de emociones
        top_emotion = max(fused_emotions, key=fused_emotions.get) if fused_emotions else "neutral"
        top_score = fused_emotions.get(top_emotion, 0.5)
        
        agreement = self._calculate_agreement(results)
        confidence = (top_score + agreement) / 2

        return EmotionResult(
            emotions=fused_emotions,
            top_emotion=top_emotion,
            top_score=top_score,
            confidence=confidence,
            source="temporal"
        )

    def _apply_sensitivity_adjustments(self, emotions: Dict[str, float]) -> Dict[str, float]:
        #aplica ajsutes para reducir el nutral y potenciar las emociones activas
        NEUTRAL_DAMP = 0.15
        ACTIVE_BOOST = 0.15
        MIN_THRESHOLD = 0.05

        adjusted = {}
        for emo,score in emotions.items():
            emo_lower = emo.lower()
            if emo_lower in ["neutral","other","others","neu"]:
                adjusted[emo] = score*NEUTRAL_DAMP
            else:
                adjusted[emo] = score*ACTIVE_BOOST
        return adjusted

        if adjusted:
            top=max(adjusted,key=adjusted.get)
            if top.lower() in ["neutral","other","others","neu"]:
                candidatos = {k: v for k, v in adjusted.items() 
                            if k.lower() not in ["neutral", "other", "others", "neu"] and v > MIN_THRESHOLD}
                if candidatos:
                    best_alt = max(candidatos, key=candidatos.get)
                    adjusted[best_alt] *= 1.5
        return adjusted

    def _calculate_agreement(self, results: List[EmotionResult]) -> float:
        """Calcula la concordancia entre los resultados."""
        if len(results) < 2:
            return 1.0
        
        top_emotions = [r.top_emotion for r in results]
        from collections import Counter
        counts = Counter(top_emotions)
        most_common_count = counts.most_common(1)[0][1]

        return most_common_count / len(results)
    
    def reset(self):
        """Resetea el estado del analizador."""
        self.history = []
        self.segment_count = 0
        self._temporal_state.reset()
    
    def get_emotion_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de las emociones."""
        if not self.history:
            return {"dominant": "neutral", "distribution": {}, "transitions": 0}

        # Acumular emociones
        accumulated = {}
        for emotions in self.history:
            for emo, score in emotions.items():
                accumulated[emo] = accumulated.get(emo, 0) + score

        # Normalizar
        total = sum(accumulated.values())
        if total > 0:
            accumulated = {k: round(v/total, 4) for k, v in accumulated.items()}

        # Calcular transiciones
        transitions = 0
        prev_top = None
        for emotions in self.history:
            if emotions:
                top = max(emotions.items(), key=lambda x: x[1])[0]
                if prev_top and prev_top != top:
                    transitions += 1
                prev_top = top
        
        return {
            "dominant": max(accumulated, key=accumulated.get) if accumulated else "neutral",
            "distribution": accumulated,
            "transitions": transitions,
            "segment_count": self.segment_count
        }

