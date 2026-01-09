"""
Modulo para la transcripcion en tiempo real
"""

import logging
import asyncio
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)

#VAD simple basado en la energia del audio
class SimpleVAD:
    """
    Voice Activity Detection simple basado en energia
    Detecta cuando hay voz activa en el audio
    """
    
    def __init__(
        self,
        sr: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        speech_pad_ms: int = 300
    ):
        self.sr = sr
        self.frame_size = int(sr * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        self.speech_pad_samples = int(sr * speech_pad_ms / 1000)
        
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._min_speech_frames = 3  
        self._min_silence_frames = 10  
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Procesa un frame de audio y detecta actividad de voz.
        
        Returns:
            Dict con estado de VAD
        """
        energy = np.sqrt(np.mean(frame ** 2))
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            
            if not self._is_speaking and self._speech_frames >= self._min_speech_frames:
                self._is_speaking = True
                return {"event": "speech_start", "energy": float(energy)}
        else:
            self._silence_frames += 1
            
            if self._is_speaking and self._silence_frames >= self._min_silence_frames:
                self._is_speaking = False
                self._speech_frames = 0
                return {"event": "speech_end", "energy": float(energy)}
        
        return {
            "event": "continue",
            "is_speaking": self._is_speaking,
            "energy": float(energy)
        }
    
    def reset(self):
    #reinicia el estado del VAD
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

"""
buffer para acumular la energia del audio
"""

"""
Clase para el analisis de emociones en tiempo real
"""

"""
procesa un chunk de audio del stream
"""

"""
analsiis parcial y rapido de emociones
"""