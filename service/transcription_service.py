"""
servicio de transcripción centralizado
extrae la logica de la transcripción de app_fastapi
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.models import load_whisper_models
from core.audio_processing import load_audio
from core.translation import translate_batch
from core.emotion_analysis import TemporalEmotionAnalyzer
from Resilience import WHISPER_BREAKER

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionConfig:
    #configuracion para la transcripción
    
    lite_mode:bool=False
    audio_weight:float=0.4
    enable_translation:bool=False
    num_speakers:Optional[int]=None
    apply_speaker_labels:bool=True 

@dataclass
class TranscriptionResult:
    #resultado de la transcripción
    text:str
    segments:List[Dict[str,Any]]
    duration:float
    processing_time:float
    warnings:List[str]

class TranscriptionService:
    def __init__(self):
        self._models_cache =None
        self._emotion_analysis = None
    
    def get_models(self):
        if self._models_cache is None:
            self._models_cache = load_whisper_models()
        return self._models_cache

    def get_emotion_analysis(self):
        if self._emotion_analysis is None:
            self._emotion_analysis = TemporalEmotionAnalyzer(use_prosody=ENABLE_PROSODY)
        return self._emotion_analysis

    def transcribe(self,
        audio_path: str,
        config: TranscriptionConfig
    ) -> Dict[str, Any]:

        models = self.get_models()
        whisper_model = models["whisper"]
        
        def _transcribe():
            return WHISPER_BREAKER.call(
                whisper_model.transcribe,
                audio_path,
                language="es",
                fallback=lambda *a, **k: {"text": "", "segments": []}
            )
        
        return _transcribe()

    def analyze_emotion(self,
        segments: List[Dict[str, Any]],
        audio_path: str,
        audio_array: np.ndarray,
        sr: int,
        config: TranscriptionConfig
    ) -> List[Dict[str, Any]]:

        if config.lite_mode:
            logger.info ("modo lite activado, omitiendo análisis emocional")
            return segments

        analyzer = self.get_emotion_analysis()
        enriched_segments = []

        for seg in segments:
            text_es = seg.get("text_es","")
            text_en = seg.get("start",0)
            end = seg.get("end",0)
            
            audio_segment = None
            if audio_array is not None and config.audio_weight > 0:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                audio_segment = audio_array[start_sample:end_sample]

            text_en =""
            from config import ENABLE_TRANSLATION
            if ENABLE_TRANSLATION:
                text_en = translate_batch([text_es])[0]

            emotion_result = analyzer.analyze(
                text_es=text_es,
                text_en=text_en,
                audio_array=audio_segment,
                sr=sr,
                audio_weight=config.audio_weight,
                apply_smoothing=config.apply_smoothing,
                use_ensemble=True
            )

            #enriquecer el segmento con el resultado del análisis emocional

            seg_enriched = seg.copy()
            seg_enriched.update({
                "text_en": text_en,
                "emotion": emotion_result.top_emotion,
                "emotion_score": emotion_result.top_score,
                "confidence": emotion_result.confidence,
                "emotions": emotion_result.emotions
            })
            enriched_segments.append(seg_enriched)

        return enriched_segments
    
#instancia global 

def get_transcription_service():
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service