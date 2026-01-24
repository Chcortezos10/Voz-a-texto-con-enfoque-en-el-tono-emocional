
"""
Módulo unificado de análisis emocional multi-modal.
Combina análisis de texto (español e inglés) y audio para mayor precisión.
"""
import logging
import hashlib
import threading
import time as time_module
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, OrderedDict, Counter
from core.model_manager import get_model_manager
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
from config import RESULT_CACHE_SIZE, RESULT_CACHE_TTL
import threading

# Configurar logger primero
logger = logging.getLogger(__name__)

# Import de librosa con manejo de errores
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa no está instalado. No se podrá realizar análisis de emociones de audio.")

import torch
import numpy as np
from transformers import pipeline

from config import (
    TEXT_EMOTION_MODEL,
    EMOTION_BOOST_FACTOR,
    EMOTION_NEUTRAL_DAMP,
    EMOTION_CONFIDENCE_THRESHOLD,
    AUDIO_EMOTION_MODEL,
    EMOTION_ES_MODEL,
    EMOTION_WEIGHT_TEXT,
    EMOTION_WEIGHT_AUDIO,
    MIN_EMOTION_CONFIDENCE,
    FUSION_MODE,
    DEVICE,
    EMOTION_MAPPING,
    TARGET_SR,
    SAD_BOOST_FACTOR,
    NEUTRAL_SUPPRESSION_FACTOR,
    ACTIVE_EMOTION_BOOST,
    HAPPY_SUPPRESSION_FACTOR
)

# Diccionario de palabras clave para reforzar detección emocional en español
# Estas palabras/frases sobrescriben la clasificación del modelo cuando hay confusión
LEXICON_TRISTE = {
    # Palabras EXTREMAS (peso 0.5) - Eventos traumáticos
    "muerte": 0.5, "muerto": 0.5, "murió": 0.5, "falleció": 0.5,
    "suicidio": 0.5, "suicida": 0.5, "luto": 0.5, "duelo": 0.5,
    "agonía": 0.5, "tragedia": 0.5, "fatal": 0.5,
    
    # Palabras FUERTES (peso 0.4) - Sufrimiento intenso
    "llorar": 0.5, "lloro": 0.5, "llorando": 0.5,  # ← Aumentar peso
    "lágrimas": 0.5, "lágrima": 0.5,  # ← Aumentar peso
    "llora": 0.5, "lloran": 0.5, "lloraba": 0.5,
    "llore": 0.45, "llores": 0.45,
    "sollozar": 0.5, "sollozos": 0.5, "sollozo": 0.5,
    "gimiendo": 0.45, "gimotear": 0.45, "lamento": 0.45,
    "me pongo a llorar": 0.5, "me hace llorar": 0.5,
    "estoy llorando": 0.5, "voy a llorar": 0.5,
    "quiero llorar": 0.45, "ganas de llorar": 0.45,
    "no puedo parar de llorar": 0.5,
    
    "pobreza": 0.4, "hambre": 0.4, "deuda": 0.4, "miseria": 0.4,
    "dolor": 0.4, "doloroso": 0.4, "sufro": 0.4, "sufriendo": 0.4,
    "desesperado": 0.4, "desesperación": 0.4, "angustia": 0.4,
    "deprimido": 0.4, "depresión": 0.4, "devastado": 0.4,
    
    # Palabras MODERADAS (peso 0.3) - Malestar general
    "triste": 0.3, "tristeza": 0.3, "pena": 0.3, "lástima": 0.3,
    "me pone mal": 0.3, "pone mal": 0.3, "me duele": 0.3, 
    "me afecta": 0.3, "me entristece": 0.3, "qué pena": 0.3,
    "me da pena": 0.3, "me da tristeza": 0.3, "me da lástima": 0.3,
    "perdón": 0.3, "lo siento": 0.3, "disculpa": 0.3, "perdona": 0.3,
    "lamentablemente": 0.3, "desafortunadamente": 0.3, "por desgracia": 0.3,
    "pobres": 0.3, "soledad": 0.3, "solo": 0.3, "sola": 0.3,
    "melancolía": 0.3, "melancólico": 0.3, "nostálgico": 0.3,
    
    # Palabras LEVES (peso 0.2) - Dificultades
    "difícil": 0.2, "complicado": 0.2, "terrible": 0.2, "horrible": 0.2,
    "mal": 0.2, "peor": 0.2, "desgracia": 0.2,
    "no puedo": 0.2, "no soporto": 0.2, "no aguanto": 0.2, "me cuesta": 0.2,
}

LEXICON_ENOJADO = {
    # Palabras EXTREMAS (peso 0.5) - Furia e indignación
    "odio": 0.5, "detesto": 0.5, "asco": 0.5, "repugnante": 0.5,
    "maldito": 0.5, "maldita": 0.5, "maldición": 0.5,
    "furia": 0.5, "furioso": 0.5, "furiosa": 0.5,
    "rabia": 0.5, "ira": 0.5, "iracundo": 0.5,
    "me cago en": 0.5, "vete a la mierda": 0.5, "vete al carajo": 0.5,
    "hijo de puta": 0.5, "hijos de puta": 0.5,
    "mierda": 0.5, "carajo": 0.5, "joder": 0.5,
    "puto": 0.5, "puta": 0.5, "puñeta": 0.5,
    
    # Palabras FUERTES (peso 0.4) - Enojo intenso
    "enojado": 0.4, "enojada": 0.4, "enojo": 0.4,
    "molesto": 0.4, "molesta": 0.4, "molestia": 0.4,
    "indignado": 0.4, "indignada": 0.4, "indignación": 0.4,
    "injusto": 0.4, "injusticia": 0.4, "injusta": 0.4,
    "me irrita": 0.4, "irritante": 0.4, "irritado": 0.4,
    "harto": 0.4, "harta": 0.4, "hartazgo": 0.4,
    "cabrón": 0.4, "cabrona": 0.4, "cabrones": 0.4,
    "estúpido": 0.4, "estúpida": 0.4, "idiota": 0.4,
    "imbécil": 0.4, "pendejo": 0.4, "pendeja": 0.4,
    "me revienta": 0.4, "me cansa": 0.4, "me saca": 0.4,
    "no aguanto": 0.4, "no soporto": 0.4, "insoportable": 0.4,
    
    # Palabras MODERADAS (peso 0.3) - Molestia general
    "qué asco": 0.3, "asqueroso": 0.3, "asquerosa": 0.3,
    "corrupto": 0.3, "corrupción": 0.3, "corrupta": 0.3,
    "ladrones": 0.3, "ladrón": 0.3, "roban": 0.3, "robo": 0.3,
    "mentira": 0.3, "mentiroso": 0.3, "mentirosa": 0.3,
    "engaño": 0.3, "trampa": 0.3, "tramposo": 0.3,
    "pelea": 0.3, "gritos": 0.3, "gritando": 0.3, "grito": 0.3,
    "discutir": 0.3, "discusión": 0.3, "peleando": 0.3,
    "bronca": 0.3, "enemistad": 0.3, "rencor": 0.3,
    "venganza": 0.3, "vengarme": 0.3, "vengativo": 0.3,
    "hostil": 0.3, "hostilidad": 0.3, "agresivo": 0.3,
    "violento": 0.3, "violencia": 0.3, "agredir": 0.3,
    "abuso": 0.3, "abusivo": 0.3, "abusiva": 0.3,
    "desprecio": 0.3, "despreciable": 0.3, "despreciar": 0.3,
    "humillar": 0.3, "humillación": 0.3, "humillante": 0.3,
    "frustrado": 0.3, "frustración": 0.3, "frustrante": 0.3,
    "intolerante": 0.3, "intolerancia": 0.3,
    
    # Palabras LEVES (peso 0.2) - Molestia leve
    "fastidioso": 0.2, "fastidio": 0.2, "fastidiado": 0.2,
    "molestia": 0.2, "problema": 0.2, "problemático": 0.2,
    "incómodo": 0.2, "incomodidad": 0.2, "desagradable": 0.2,
    "disgusto": 0.2, "disgustado": 0.2, "disgustada": 0.2,
    "enfadado": 0.2, "enfadada": 0.2, "enfado": 0.2,
    "queja": 0.2, "quejar": 0.2, "quejoso": 0.2,
    "crítica": 0.2, "criticar": 0.2, "crítico": 0.2,
    "protesta": 0.2, "protestar": 0.2, "reclamar": 0.2,
}

LEXICON_FELIZ = {
    # Palabras EXTREMAS (peso 0.5) - Euforia
    "amo": 0.5, "adoro": 0.5, "me encanta": 0.5, "encantado": 0.5,
    "increíble": 0.5, "espectacular": 0.5, "maravilloso": 0.5,
    "perfecta": 0.5, "perfecto": 0.5, "buenísimo": 0.5,
    
    # Palabras FUERTES (peso 0.4) - Alegría intensa
    "feliz": 0.4, "felicidad": 0.4, "alegría": 0.4, "alegre": 0.4,
    "genial": 0.4, "fantástico": 0.4, "excelente": 0.4,
    
    # ← AGREGAR ESTAS (detección de risa):
    "jajaja": 0.5, "jajajaja": 0.5, "jajajajaja": 0.5,
    "jajajajajaja": 0.5, "jajajajajajaja": 0.5,
    "jejeje": 0.45, "jejejeje": 0.45, "jijiji": 0.45,
    "jojojo": 0.45, "juajua": 0.45,
    "hahaha": 0.5, "hahahaha": 0.5, "hehehe": 0.45,
    "jaja": 0.4, "jeje": 0.4, "jiji": 0.4, "jojo": 0.4,
    "carcajada": 0.5, "carcajadas": 0.5,
    "riendo": 0.45, "riéndose": 0.45, "reír": 0.4,
    "risa": 0.4, "risas": 0.4, "risota": 0.45, "risotas": 0.45,
    "muerto de risa": 0.5, "me muero de risa": 0.5,
    "me río": 0.4, "me estoy riendo": 0.45,
    "qué risa": 0.4, "qué risas": 0.4, "me da risa": 0.4,
    "ja": 0.3, "je": 0.3, "ji": 0.3,  # Risa corta
    
    "celebrar": 0.4, "celebración": 0.4, "éxito": 0.4, "victoria": 0.4,
    
    # Palabras MODERADAS (peso 0.3) - Satisfacción
    "contento": 0.3, "contenta": 0.3, "qué bien": 0.3, "qué bueno": 0.3,
    "hermoso": 0.3, "hermosa": 0.3, "lindo": 0.3, "linda": 0.3,
    "gracioso": 0.3, "divertido": 0.3, "divertida": 0.3,
    "qué chistoso": 0.3, "chistoso": 0.3, "cómico": 0.3,
    "humor": 0.3, "humorístico": 0.3,
    
    # Palabras LEVES (peso 0.2)
    "bien": 0.2, "mejor": 0.2, "super": 0.2, "súper": 0.2, 
    "guau": 0.2, "wow": 0.2, "fiesta": 0.2, "sonrisa": 0.2,
}

def analyze_lexicon(text: str) -> Dict[str, float]:
    """
    Analiza el texto buscando palabras clave emocionales.
    Retorna un diccionario con scores basados en coincidencias lexicográficas.
    """
    text_lower = text.lower()
    scores = {"triste": 0.0, "enojado": 0.0, "feliz": 0.0, "neutral": 0.0}

    # Buscar coincidencias en cada lexicón
    for palabra in LEXICON_TRISTE:
        if palabra in text_lower:
            scores["triste"] += 0.3  # Cada coincidencia suma 0.3

    for palabra in LEXICON_ENOJADO:
        if palabra in text_lower:
            scores["enojado"] += 0.3

    for palabra in LEXICON_FELIZ:
        if palabra in text_lower:
            scores["feliz"] += 0.3

    # Normalizar si hay coincidencias
    total = sum(scores.values())
    if total > 0:
        scores = {k: min(1.0, v) for k, v in scores.items()}  # Cap at 1.0
    else:
        scores["neutral"] = 0.1  # Sin coincidencias = ligeramente neutral

    return scores

def apply_lexicon_correction(emotions: Dict[str, float], text: str) -> Dict[str, float]:
    lexicon_scores = analyze_lexicon(text)
    text_lower = text.lower()

    has_laugh = any(x in text_lower for x in ["jaja", "jeje", "risa", "risas", "jajaja", "jejeje", "riendo", "carcajada"])

    if lexicon_scores["triste"] >= 0.15 and lexicon_scores["feliz"] < 0.15:
        feliz_score = emotions.get("feliz", 0.0)
        if feliz_score > 0.15:
            transfer = feliz_score * 0.85
            emotions["feliz"] = feliz_score - transfer
            emotions["triste"] = emotions.get("triste", 0.0) + transfer + 0.15
        else:
            emotions["triste"] = emotions.get("triste", 0.0) + 0.25

    if lexicon_scores["enojado"] >= 0.15 and lexicon_scores["feliz"] < 0.15:
        feliz_score = emotions.get("feliz", 0.0)
        if feliz_score > 0.15:
            transfer = feliz_score * 0.85
            emotions["feliz"] = feliz_score - transfer
            emotions["enojado"] = emotions.get("enojado", 0.0) + transfer + 0.15
        else:
            emotions["enojado"] = emotions.get("enojado", 0.0) + 0.25

    if lexicon_scores["feliz"] >= 0.15 or has_laugh:
        if lexicon_scores["triste"] < 0.15 and lexicon_scores["enojado"] < 0.15:
            emotions["feliz"] = emotions.get("feliz", 0.0) + 0.35
            if has_laugh:
                emotions["feliz"] = emotions.get("feliz", 0.0) + 0.25

    return emotions

# MAPEO_EMOCIONAL_EXTENDIDO eliminado en favor de config.EMOTION_MAPPING

# Constantes
GO_EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
PROSODY_WEIGHT = 0.15

@dataclass
class EmotionResult:
    """Resultado de análisis emocional."""
    emotions: Dict[str, float]  # Mapa de emoción -> score
    top_emotion: str
    top_score: float
    confidence: float
    source: str  # 'text', 'audio', 'text_es', 'text_en', 'multimodal'

class SmartResultCache:
    def __init__(self, maxsize: int = 200,ttl: int = 1800):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[EmotionResult]:
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def add(self, key: str, value: EmotionResult) -> None:
        with self.lock:
            self.cache[key] = value

    def get_stats(self):
        with self.lock:
            return {
                "size": len(self.cache),
                "maxsize": self.cache.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / max(1, self.hits + self.misses) * 100, 2)
            }

    @staticmethod
    def make_key(text: str, model_name: str) -> str:
        content = f"{model_name}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class ProsodyFeatures:
    #características prosódicas extraidas del audio
    pitch_mean: float = 150.0
    pitch_std: float = 20.0
    energy_mean: float = 0.05
    energy_std: float = 0.02
    pitch_range: float = 50.0
    energy_max: float = 0.1
    speech_rate: float = 3.0
    pause_ratio: float = 0.2
    voiced_ratio: float = 0.7

class ProsodyAnalyzer:
    #analiza las características prosódicas del audio para poder inferir emociones

    PROSODY_RULES = {
        "feliz": {
            "pitch": "high", 
            "pitch_var": "high", 
            "energy": "high", 
            "rate": "fast",
            "weights": {"pitch": 0.3, "pitch_var": 0.2, "energy": 0.3, "rate": 0.2}
        },
        "enojado": {
            "pitch": "high", 
            "pitch_var": "medium", 
            "energy": "very_high", 
            "rate": "fast",
            "weights": {"pitch": 0.2, "pitch_var": 0.15, "energy": 0.4, "rate": 0.25}
        },
        "triste": {
            "pitch": "low", 
            "pitch_var": "low", 
            "energy": "low", 
            "rate": "slow",
            "weights": {"pitch": 0.3, "pitch_var": 0.2, "energy": 0.3, "rate": 0.2}
        },
        "neutral": {
            "pitch": "medium", 
            "pitch_var": "low", 
            "energy": "medium", 
            "rate": "medium",
            "weights": {"pitch": 0.25, "pitch_var": 0.25, "energy": 0.25, "rate": 0.25}
        }
    }

    def __init__(self, sr: int = TARGET_SR):
        """Inicializa el analizador prosódico con la tasa de muestreo."""
        self.sr = sr
        self.pitch_thresholds = {"low": 110, "medium_low": 140, "medium_high": 180, "high": 210}
        self.energy_thresholds = {"low": 0.015, "medium": 0.05, "high": 0.08, "very_high": 0.12}
        self.rate_thresholds = {"slow": 1.8, "medium": 3.5, "fast": 5.0}

    def extract_features(self, audio: np.ndarray) -> ProsodyFeatures:
        """Extrae las características prosódicas del audio."""
        if not LIBROSA_AVAILABLE:
            return ProsodyFeatures()

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )

            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                f0_valid = np.array([150.0])

            # Energía RMS
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]

            # Tasa de habla estimada
            energy_diff = np.abs(np.diff(rms))
            threshold = np.mean(energy_diff) + 0.5 * np.std(energy_diff)
            onsets = np.sum(energy_diff > threshold)
            duration = len(audio) / self.sr

            # Ratio de pausas
            silence_threshold = np.mean(rms) * 0.25
            pause_frames = np.sum(rms < silence_threshold)

            return ProsodyFeatures(
                pitch_mean=float(np.mean(f0_valid)),
                pitch_std=float(np.std(f0_valid)),
                pitch_range=float(np.ptp(f0_valid)),
                energy_mean=float(np.mean(rms)),
                energy_std=float(np.std(rms)),
                energy_max=float(np.max(rms)),
                speech_rate=onsets / max(duration, 0.1),
                pause_ratio=pause_frames / max(len(rms), 1),
                voiced_ratio=float(np.mean(voiced_flag)) if len(voiced_flag) > 0 else 0.5
            )
        except Exception as e:
            print(f"Error al extraer características prosódicas: {e}")
            return ProsodyFeatures()

    def infer_emotion(self, features: ProsodyFeatures) -> Tuple[str, float, Dict[str, float]]:
        """Infiere la emoción a partir de las características prosódicas."""
        scores = {}
        for emotion, rule in self.PROSODY_RULES.items():
            score = 0.0
            weights = rule.get("weights", {"pitch": 0.25, "pitch_var": 0.25, "energy": 0.25, "rate": 0.25})

            # Evaluar pitch
            pitch_score = self._evaluate_pitch(features.pitch_mean, rule["pitch"])
            score += pitch_score * weights["pitch"]

            # Evaluar variabilidad de pitch
            var_score = self._evaluate_pitch_var(features.pitch_std, rule["pitch_var"])
            score += var_score * weights["pitch_var"]

            # Evaluar energía
            energy_score = self._evaluate_energy(features.energy_mean, rule["energy"])
            score += energy_score * weights["energy"]

            # Evaluar velocidad
            rate_score = self._evaluate_rate(features.speech_rate, rule["rate"])
            score += rate_score * weights["rate"]

            scores[emotion] = score

        # Normalizar scores y calcular top_emotion
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
            top_emotion = max(scores, key=scores.get)
            confidence = scores[top_emotion]
        else:
            top_emotion = "neutral"
            confidence = 0.0
            scores = {"neutral": 1.0}

        return top_emotion, confidence, scores

    def _evaluate_pitch(self, pitch: float, expected: str) -> float:
        """Evalúa el pitch contra la expectativa."""
        if expected == "high":
            return min(1.0, max(0.0, (pitch - 150) / 100))
        elif expected == "low":
            return min(1.0, max(0.0, (180 - pitch) / 100))
        else:  # medium
            return 1.0 - abs(pitch - 160) / 100

    def _evaluate_pitch_var(self, std: float, expected: str) -> float:
        """Evalúa la variabilidad del pitch."""
        if expected == "high":
            return min(1.0, std / 40)
        elif expected == "low":
            return min(1.0, max(0.0, (30 - std) / 30))
        else:
            return 1.0 - abs(std - 25) / 30

    def _evaluate_energy(self, energy: float, expected: str) -> float:
        """Evalúa la energía."""
        thresholds = self.energy_thresholds
        if expected == "very_high":
            return min(1.0, energy / thresholds["very_high"])
        elif expected == "high":
            return min(1.0, energy / thresholds["high"])
        elif expected == "low":
            return min(1.0, max(0.0, (thresholds["medium"] - energy) / thresholds["medium"]))
        else:
            return 1.0 - abs(energy - thresholds["medium"]) / thresholds["medium"]

    def _evaluate_rate(self, rate: float, expected: str) -> float:
        """Evalúa la velocidad del habla."""
        thresholds = self.rate_thresholds
        if expected == "fast":
            return min(1.0, rate / thresholds["fast"])
        elif expected == "slow":
            return min(1.0, max(0.0, (thresholds["medium"] - rate) / thresholds["medium"]))
        else:
            return 1.0 - abs(rate - thresholds["medium"]) / thresholds["medium"]


_prosody_analyzer: Optional[ProsodyAnalyzer] = None

def get_prosody_analyzer() -> ProsodyAnalyzer:
    """Obtiene o crea instancia del analizador prosódico."""
    global _prosody_analyzer
    if _prosody_analyzer is None:
        _prosody_analyzer = ProsodyAnalyzer()
    return _prosody_analyzer


class TemporalEmotionState:
    """Estado para suavizado temporal de emociones."""

    def __init__(self, max_history: int = 10):
        """Inicializa el estado temporal con historial limitado."""
        self.history = deque(maxlen=max_history)

    def add(self, emotions: Dict[str, float]) -> None:
        """Agrega un nuevo resultado de emociones al historial."""
        self.history.append(emotions.copy())

    def update(self, emotions: Dict[str, float]) -> None:
        """Alias for add method."""
        self.add(emotions)

    def reset(self) -> None:
        self.history.clear()

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
        top_k=None,
        device=device_num
    )


@lru_cache(maxsize=1)
def load_text_emotion_es():
    from config import MAX_EMOTION_MODELS_LOADED
    device_num = 0 if get_device() == "cuda" else -1

    def _loader():
        logger.info(f"Cargando modelo de emociones en español...")
        return pipeline(
            "text-classification",
            model=EMOTION_ES_MODEL,
            top_k=None,
            device=device_num
        )

    manager = get_model_manager(max_models=MAX_EMOTION_MODELS_LOADED)
    return manager.get_model("text_emotion_es", _loader)    

@lru_cache(maxsize=1)
def load_audio_emotion():
    from config import MAX_EMOTION_MODELS_LOADED
    device_num = 0 if get_device() == "cuda" else -1

    def _loader():
        logger.info(f"Cargando modelo de emociones de audio...")
        return pipeline(
            "audio-classification",
            model=AUDIO_EMOTION_MODEL,
            device=device_num
        )

    manager = get_model_manager(max_models=MAX_EMOTION_MODELS_LOADED)
    return manager.get_model("audio_emotion", _loader)


@lru_cache(maxsize=1)
def load_go_emotion():
    from config import ENABLE_GO_EMOTIONS, MAX_EMOTION_MODELS_LOADED
    if not ENABLE_GO_EMOTIONS:
        logger.info("GoEmotions deshabilitado")
        return None

    device_num = 0 if get_device() == "cuda" else -1

    def _loader():
        logger.info(f"Cargando modelo de emociones de GoEmotions...")
        return pipeline(
            "text-classification",
            model=GO_EMOTION_MODEL,
            top_k=None,
            device=device_num
        )

    try:
        manager = get_model_manager(max_models=MAX_EMOTION_MODELS_LOADED)
        return manager.get_model("go_emotion", _loader)
    except OSError as e:
        logger.error(f"GoEmotions no disponible: {str(e)[:100]}")
        return None
    except Exception as e:
        logger.error(f"GoEmotions deshabilitado: {str(e)[:100]}")
        return None

def analyze_text_go_emotions(text: str) -> EmotionResult:
    """Analiza emoción usando GoEmotions (28 emociones)."""
    if not text or not text.strip():
        return EmotionResult({}, "neutral", 0.0, 0.0, "go_emotions")

    try:
        classifier = load_go_emotion()
        if classifier is None:
            return EmotionResult({"neutral": 1.0}, "neutral", 1.0, 0.5, "go_emotions")

        results = classifier(text[:512])[0]
        emotions = {"feliz": 0.0, "enojado": 0.0, "triste": 0.0, "neutral": 0.0}

        for result in results:
            label = result["label"].lower()
            score = float(result["score"])
            mapped = EMOTION_MAPPING.get(label, "neutral")
            emotions[mapped] = emotions.get(mapped, 0.0) + score

        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}

        top_emotion = max(emotions, key=emotions.get)
        return EmotionResult(
            emotions=emotions,
            top_emotion=top_emotion,
            top_score=emotions[top_emotion],
            confidence=emotions[top_emotion],
            source="go_emotions"
        )
    except Exception as e:
        logger.error(f"Error en GoEmotions: {e}")
        return EmotionResult({}, "neutral", 0.0, 0.0, "go_emotions")


@dataclass
class EnsembleResult:
    """Resultado del ensemble completo."""
    emotions: Dict[str, float]
    top_emotion: str
    top_score: float
    confidence: float
    source: str
    individual_results: Optional[Dict[str, Any]] = None
    agreement_score: float = 0.0
    prosody_emotion: Optional[str] = None
    prosody_confidence: float = 0.0
    processing_time: float = 0.0

class EmotionEnsemble:
    """Ensemble de múltiples modelos para análisis emocional."""

    def __init__(
        self,
        use_go_emotions: bool = False,  # Deshabilitado por defecto (ahorra RAM)
        use_prosody: bool = True,
        parallel: bool = False,  # Secuencial por defecto (más estable)
        neutral_suppression: float = 0.12,
        active_boost: float = 2.0
    ):
        self.use_go_emotions = use_go_emotions
        self.use_prosody = use_prosody
        self.parallel = parallel
        self.neutral_suppression = neutral_suppression
        self.active_boost = active_boost
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel else None
        self._prosody = get_prosody_analyzer() if use_prosody else None

    def analyze(
        self,
        text_es: str,
        text_en: str = "",
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16000
    ) -> EnsembleResult:
        """Análisis completo con ensemble de modelos."""
        start_time = time_module.time()
        results = {}
        weights = {}

        if self.parallel and self._executor:
            futures = {}

            if text_es.strip():
                futures['text_es'] = self._executor.submit(analyze_text_emotion_es, text_es)
                weights['text_es'] = 0.30

            if text_en.strip():
                futures['text_en'] = self._executor.submit(analyze_text_emotion_en, text_en)
                weights['text_en'] = 0.25

            if self.use_go_emotions and text_en.strip():
                futures['go_emotions'] = self._executor.submit(analyze_text_go_emotions, text_en)
                weights['go_emotions'] = 0.15

            if audio_path:
                futures['audio'] = self._executor.submit(analyze_audio_emotion, audio_path)
                weights['audio'] = 0.15

            for key, future in futures.items():
                try:
                    result = future.result(timeout=60)
                    if result and result.emotions:
                        results[key] = result
                except TimeoutError:
                    logger.warning(f"Timeout en {key}, continuando...")
                except Exception as e:
                    logger.warning(f"Error en {key}: {str(e)[:80]}")
        else:
            if text_es.strip():
                results['text_es'] = analyze_text_emotion_es(text_es)
                weights['text_es'] = 0.30

            if text_en.strip():
                results['text_en'] = analyze_text_emotion_en(text_en)
                weights['text_en'] = 0.25

            if self.use_go_emotions and text_en.strip():
                result = analyze_text_go_emotions(text_en)
                if result.emotions:
                    results['go_emotions'] = result
                    weights['go_emotions'] = 0.15

            if audio_path:
                results['audio'] = analyze_audio_emotion(audio_path)
                weights['audio'] = 0.15

        prosody_emotion = None
        prosody_confidence = 0.0

        if self.use_prosody and self._prosody and audio_array is not None and len(audio_array) > 0:
            try:
                features = self._prosody.extract_features(audio_array)
                prosody_emotion, prosody_confidence, prosody_scores = self._prosody.infer_emotion(features)

                results['prosody'] = EmotionResult(
                    emotions=prosody_scores,
                    top_emotion=prosody_emotion,
                    top_score=prosody_confidence,
                    confidence=prosody_confidence,
                    source="prosody"
                )
                weights['prosody'] = PROSODY_WEIGHT
            except Exception as e:
                logger.warning(f"Error en prosodia: {e}")

        if not results:
            return EnsembleResult(
                emotions={"neutral": 1.0},
                top_emotion="neutral",
                top_score=0.5,
                confidence=0.0,
                source="fallback",
                processing_time=time_module.time() - start_time
            )

        fused = self._fuse_results(results, weights)
        agreement = self._calculate_agreement(results)
        confidence = (fused["top_score"] * 0.6) + (agreement * 0.4)

        return EnsembleResult(
            emotions=fused["emotions"],
            top_emotion=fused["top_emotion"],
            top_score=round(fused["top_score"], 4),
            confidence=round(confidence, 4),
            source="ensemble",
            individual_results=results,
            agreement_score=round(agreement, 4),
            prosody_emotion=prosody_emotion,
            prosody_confidence=round(prosody_confidence, 4),
            processing_time=round(time_module.time() - start_time, 3)
        )

    def _fuse_results(self, results: Dict[str, EmotionResult], weights: Dict[str, float]) -> Dict[str, Any]:
        """Fusiona resultados con votación ponderada."""
        total_weight = sum(weights.get(k, 0) for k in results.keys())
        if total_weight == 0:
            total_weight = 1

        norm_weights = {k: weights.get(k, 0) / total_weight for k in results.keys()}

        all_emotions = set()
        for result in results.values():
            all_emotions.update(result.emotions.keys())

        fused = {}
        for emotion in all_emotions:
            weighted_sum = 0.0
            for key, result in results.items():
                weighted_sum += result.emotions.get(emotion, 0.0) * norm_weights.get(key, 0)
            fused[emotion] = weighted_sum

        fused = self._apply_sensitivity(fused)

        total = sum(fused.values())
        if total > 0:
            fused = {k: round(v / total, 4) for k, v in fused.items()}

        top_emotion = max(fused, key=fused.get)

        return {"emotions": fused, "top_emotion": top_emotion, "top_score": fused[top_emotion]}

    def _apply_sensitivity(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Aplica ajustes de sensibilidad balanceados."""
        adjusted = {}

        for emo, score in emotions.items():
            emo_lower = emo.lower()
            if emo_lower in ["neutral", "other", "others", "neu"]:
                adjusted[emo] = score * NEUTRAL_SUPPRESSION_FACTOR
            elif emo_lower in ["triste", "sad", "sadness", "tristeza"]:
                adjusted[emo] = score * ACTIVE_EMOTION_BOOST * SAD_BOOST_FACTOR
            elif emo_lower in ["feliz", "happy", "joy", "happiness"]:
                adjusted[emo] = score * HAPPY_SUPPRESSION_FACTOR
            else:
                adjusted[emo] = score * ACTIVE_EMOTION_BOOST

        if adjusted:
            top = max(adjusted, key=adjusted.get)
            if top.lower() in ["neutral", "other", "others"]:
                candidates = {k: v for k, v in adjusted.items() if k.lower() not in ["neutral", "other", "others"] and v > 0.12}
                if candidates:
                    best = max(candidates, key=candidates.get)
                    adjusted[best] *= 1.15

        return adjusted

    def _calculate_agreement(self, results: Dict[str, EmotionResult]) -> float:
        """Calcula acuerdo entre modelos."""
        if len(results) < 2:
            return 1.0

        top_emotions = [r.top_emotion for r in results.values()]
        counts = Counter(top_emotions)
        most_common_count = counts.most_common(1)[0][1]

        return most_common_count / len(results)


_emotion_ensemble: Optional[EmotionEnsemble] = None


def get_emotion_ensemble(use_go_emotions: bool = False, use_prosody: bool = True, parallel: bool = False) -> EmotionEnsemble:
    """Obtiene o crea instancia del ensemble (modo bajo RAM por defecto)."""
    global _emotion_ensemble
    if _emotion_ensemble is None:
        _emotion_ensemble = EmotionEnsemble(use_go_emotions=use_go_emotions, use_prosody=use_prosody, parallel=parallel)
    return _emotion_ensemble


def reset_emotion_ensemble():
    """Reinicia el ensemble."""
    global _emotion_ensemble
    _emotion_ensemble = None


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
            source="text_en"
        )
    except OSError as e:
        logger.warning(f"Modelo EN no disponible (RAM): {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "text_en")
    except Exception as e:
        logger.warning(f"Error texto EN: {type(e).__name__}: {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "text_en")


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

        # APLICAR CORRECCIÓN LEXICOGRÁFICA para evitar confusión feliz/triste
        emotions = apply_lexicon_correction(emotions, text)

        # Recalcular top_emotion después de la corrección
        if emotions:
            top_emotion = max(emotions, key=emotions.get)
            top_score = emotions[top_emotion]
        else:
            top_emotion = normalize_emotion_label(max(results, key=lambda x: x["score"])["label"])
            top_score = float(max(results, key=lambda x: x["score"])["score"])

        return EmotionResult(
            emotions=emotions,
            top_emotion=top_emotion,
            top_score=top_score,
            confidence=top_score,
            source="text_es"
        )
    except OSError as e:
        logger.warning(f"Modelo ES no disponible (RAM): {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "text_es")
    except Exception as e:
        logger.warning(f"Error texto ES: {type(e).__name__}: {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "text_es")


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
    except OSError as e:
        logger.warning(f"Modelo audio no disponible (RAM): {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "audio")
    except Exception as e:
        logger.warning(f"Error audio: {type(e).__name__}: {str(e)[:60]}")
        return EmotionResult({"neutral": 1.0}, "neutral", 0.5, 0.3, "audio")


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
    BOOST_FACTOR = EMOTION_BOOST_FACTOR
    NEUTRAL_DAMP = EMOTION_NEUTRAL_DAMP
    CONFIDENCE_THRESHOLD = EMOTION_CONFIDENCE_THRESHOLD

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
    """Analizador con suavizado temporal y soporte para ensemble."""

    def __init__(self, use_prosody: bool = True):
        self.history = []
        self.segment_count = 0
        self.use_prosody = use_prosody
        self._temporal_state = TemporalEmotionState()

    def analyze_segment(
        self,
        text_es: str,
        text_en: str = "",
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16000,
        audio_weight: float = 0.4,
        apply_smoothing: bool = True,
        use_ensemble: bool = True
    ) -> EmotionResult:
        """Analiza un segmento con múltiples fuentes y suavizado temporal."""

        if use_ensemble:
            ensemble = get_emotion_ensemble(use_go_emotions=False, use_prosody=self.use_prosody)

            ensemble_result = ensemble.analyze(
                text_es=text_es,
                text_en=text_en,
                audio_path=audio_path,
                audio_array=audio_array,
                sr=sr
            )

            fused_emotions = ensemble_result.emotions
            top_emotion = ensemble_result.top_emotion
            top_score = ensemble_result.top_score
            confidence = ensemble_result.confidence
        else:
            results = []
            weights = []

            result_es = analyze_text_emotion_es(text_es)
            if result_es.emotions:
                results.append(result_es)
                weights.append(0.35)

            if text_en and text_en.strip():
                result_en = analyze_text_emotion_en(text_en)
                if result_en.emotions:
                    results.append(result_en)
                    weights.append(0.25)

            if audio_path and audio_weight > 0:
                try:
                    result_audio = analyze_audio_emotion(audio_path)
                    if result_audio.emotions:
                        results.append(result_audio)
                        weights.append(audio_weight)
                except Exception as e:
                    logger.warning(f"Error análisis audio: {e}")

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

            all_emotions = set()
            for r in results:
                all_emotions.update(r.emotions.keys())

            fused_emotions = {}
            for emotion in all_emotions:
                weighted_sum = sum(r.emotions.get(emotion, 0.0) * w for r, w in zip(results, weights))
                fused_emotions[emotion] = weighted_sum

            fused_emotions = self._apply_sensitivity_adjustments(fused_emotions)

            total = sum(fused_emotions.values())
            if total > 0:
                fused_emotions = {k: round(v / total, 4) for k, v in fused_emotions.items()}

            top_emotion = max(fused_emotions, key=fused_emotions.get)
            top_score = fused_emotions[top_emotion]
            confidence = top_score

        if apply_smoothing and self.history:
            fused_emotions = self._temporal_state.get_smoothed(fused_emotions, alpha=0.7)
            top_emotion = max(fused_emotions, key=fused_emotions.get)
            top_score = fused_emotions[top_emotion]

        self._temporal_state.add(fused_emotions)
        self.history.append(fused_emotions.copy())
        self.segment_count += 1

        return EmotionResult(
            emotions=fused_emotions,
            top_emotion=top_emotion,
            top_score=round(top_score, 4),
            confidence=round(confidence, 4),
            source="ensemble" if use_ensemble else "multimodal"
        )

    def _apply_sensitivity_adjustments(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Aplica ajustes de sensibilidad (BALANCEADO - permite neutral genuino)."""
        adjusted = {}
        for emo, score in emotions.items():
            emo_lower = emo.lower()
            if emo_lower in ["neutral", "other", "others"]:
                # Menos supresión de neutral (subido de 0.15)
                adjusted[emo] = score * NEUTRAL_SUPPRESSION_FACTOR
            elif emo_lower in ["triste", "sad", "sadness", "tristeza"]:
                # Boost específico para tristeza
                adjusted[emo] = score * ACTIVE_EMOTION_BOOST * SAD_BOOST_FACTOR
            elif emo_lower in ["feliz", "happy", "joy", "happiness"]:
                # PENALIZACIÓN para feliz (evitar falsos positivos)
                adjusted[emo] = score * HAPPY_SUPPRESSION_FACTOR
            else:
                adjusted[emo] = score * ACTIVE_EMOTION_BOOST
        return adjusted

    def reset(self):
        """Reinicia el estado del analizador."""
        self.history = []
        self.segment_count = 0