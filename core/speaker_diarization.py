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

# Instancia global del diarizador
diarizer = None

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


class SpeakerDiarizer:
#diarizador de hablantes usando los embeddings de un modelo de audio
    def __init__(self,
        min_segment_duration: float = 0.5,
        max_speakers: int = 10,
        use_gpu: bool = True):

        self.min_segment_duration = min_segment_duration
        self.max_speakers = max_speakers
        self.use_gpu = use_gpu
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                from resemblyzer import VoiceEncoder
                device = "cuda" if self.use_gpu else "cpu"
                try:
                    self._encoder = VoiceEncoder(device=device)
                except:
                    self._encoder = VoiceEncoder(device="cpu")
                logger.info("VoiceEncoder cargado")
            except ImportError:
                logger.error("resemblyzer no instalado. Ejecuta: pip install resemblyzer")
                raise
        return self._encoder

    def extract_embeddings(self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], np.ndarray]]:
        #extraer embeddings de voz para cada segmento del audio
        results = []
        
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            duration = end - start
            
            if duration < self.min_segment_duration:
                continue
            
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            start_idx = max(0, min(start_idx, len(audio)))
            end_idx = max(start_idx + int(sr * 0.1), min(end_idx, len(audio)))
            
            chunk = audio[start_idx:end_idx]
            
            if len(chunk) < sr * 0.1:
                continue
            
            try:
                embedding = self.encoder.embed_utterance(chunk)
                results.append((seg, embedding))
            except Exception as e:
                logger.warning(f"Error embedding segmento {start}-{end}: {e}")
        
        return results

    def cluster_speakers(self,
        embeddings: List[np.ndarray],
        num_speakers: Optional[int] = None
    ) -> np.ndarray:
        #clusterizar los embeddings
        if len(embeddings) == 0:
            return np.array([])
        
        if len(embeddings) == 1:
            return np.array([0])
        
        X = np.vstack(embeddings)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if num_speakers is not None:
            n_clusters = min(num_speakers, len(embeddings))
        else:
            # Auto-detectar usando silhouette score
            best_labels = None
            best_score = -1
            
            for n in range(1, min(self.max_speakers + 1, len(embeddings))):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(X_scaled)
                    
                    from sklearn.metrics import silhouette_score
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_scaled, labels, metric='cosine')
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                except:
                    continue
            
            if best_labels is not None:
                return best_labels
            
            n_clusters = 1
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        return clustering.fit_predict(X_scaled)
#realizar la diarización
    def diarize(self,audio:np.ndarray,sr:int,segments:List[Dict[str,Any]],
    num_speakers:Optional[int]=None)->DiarizationResult:
        #realizar la diarización
        segments_embeddings = self.extract_embeddings(audio,sr,segments)
        if not segments_embeddings:
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                speaker_embeddings={}
            )
        segments_list, embeddings_list = zip(*segments_embeddings)
        embeddings_array = list(embeddings_list)

        labels = self.cluster_speakers(embeddings_array,num_speakers=num_speakers)

        unique_labels = set(labels)
        speaker_embeddings = {}

        for label in unique_labels:
            labels_embs = [emb for emb, lbl in zip(embeddings_array, labels) if lbl == label]
            speaker_embeddings[int(label)] = np.mean(labels_embs, axis=0)

        #se crea segmentos diarizados 
        diarized_segments = []
        for i, (seg, label) in enumerate(zip(segments_list, labels)):
            confidence = 1 - cosine(embeddings_array[i], speaker_embeddings[int(label)])
            diarized_segments.append(
                SpeakerSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                speaker_id=int(label),
                speaker_label=f"Hablante {int(label) + 1}",
                confidence=float(max(0, min(1, confidence)))
                )
            )
        return DiarizationResult(
            segments=diarized_segments,
            num_speakers=len(unique_labels),
            speaker_embeddings=speaker_embeddings
        )
        
#fusionar segmentos consecutivos del mismo hablante
    def merge_consecutive_segments(      self,
        segments: List[SpeakerSegment],
        gap_threshold: float = 0.5
    ) -> List[SpeakerSegment]:
        #fusionar segmentos consecutivos del mismo hablante
        if not segments:
            return[]
        
        sorted_segments = sorted(segments,key=lambda x:x.start)
        merged_segments = [sorted_segments[0]]
        
        for seg in sorted_segments[1:]:
            last_merged = merged_segments[-1]
            if seg.speaker_id == last_merged.speaker_id and seg.start - last_merged.end <= gap_threshold:
                merged_segments[-1] = SpeakerSegment(
                    start=last_merged.start,
                    end=seg.end,
                    speaker_id=last_merged.speaker_id,
                    speaker_label=last_merged.speaker_label,
                    confidence=last_merged.confidence
                )
            else:
                merged_segments.append(seg)
        
        return merged_segments  
        
#instancia global 
def get_diarizer(use_gpu:bool = True):
    #se obtiene la instancia global del diarizador
    global diarizer
    if diarizer is None:
        diarizer = SpeakerDiarizer(use_gpu=use_gpu)
    return diarizer

def diarize_audio(audio:np.ndarray,sr:int,
segments:List[Dict[str,Any]],num_speakers:Optional[int]=None)->DiarizationResult:
    #se diariza el audio
    return get_diarizer().diarize(audio,sr,segments,num_speakers)
