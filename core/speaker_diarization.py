"""
Modulos de diarización de hablantes
detecta y separa las diferentes voces en un audio
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
#segmento de audio con hablante identificado
    start: float
    end: float
    speaker_id: int
    speaker_label: str
    confidence: float

#resultado de la diarización
@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_embeddings: Dict[int, np.ndarray]

#diarizador de hablantes usando los embeddings de un modelo de audio


#cargar lazy encoder de voz

#extraer embeddings de voz para cada segmento del audio

#clusterizar los embeddings

#realizar la diarización

#fusionar segmentos consecutivos del mismo hablante

#instancia global 