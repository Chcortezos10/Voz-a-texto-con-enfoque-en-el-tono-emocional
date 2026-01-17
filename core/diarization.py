"""
Lógica de diarización (detección de hablantes) usando embeddings y change-point detection.
"""
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize

from config import WINDOW_SEC, CHANGE_SIM_THRESHOLD, MIN_SEG_SEC
import logging
import webrtcvad

logger = logging.getLogger(__name__)


def compute_embeddings_with_vad(
    audio: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    encoder: VoiceEncoder,
    window_sec: float = WINDOW_SEC,
    vad_aggressiveness: int = 2  # 0-3 (3 más agresivo)
) -> Tuple[np.ndarray, List[float]]:
    """
    Version mejorada con WebRTC VAD para mejor deteccion de voz
    """
    embeddings = []
    starts = []
    
    # Inicializar VAD
    vad = webrtcvad.Vad(vad_aggressiveness)
    
    # WebRTC VAD requiere 16kHz y frames de 10/20/30ms
    frame_duration_ms = 30
    frame_size = int(sr * frame_duration_ms / 1000)
    
    for (s, e) in windows:
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        seg = audio[start_idx:end_idx]
        
        # Convertir a 16-bit PCM para VAD
        seg_int16 = (seg * 32767).astype(np.int16)
        
        # Verificar actividad de voz en múltiples frames
        num_frames = len(seg_int16) // frame_size
        voiced_frames = 0
        
        for i in range(num_frames):
            frame = seg_int16[i * frame_size:(i + 1) * frame_size]
            if len(frame) == frame_size:
                try:
                    is_speech = vad.is_speech(frame.tobytes(), sr)
                    if is_speech:
                        voiced_frames += 1
                except:
                    pass
        
        # Requerir al menos 50% de frames con voz
        voice_ratio = voiced_frames / max(1, num_frames)
        if voice_ratio < 0.5:
            continue
        
        # Calcular embedding
        try:
            emb = encoder.embed_utterance(seg)
            embeddings.append(emb)
            starts.append(s)
        except Exception as e:
            logger.warning(f"Error embedding segmento {s}-{e}: {e}")
    
    if not embeddings:
        return np.array([]), []
    
    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings)
    
    return embeddings, starts


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def cluster_speakers(
    embeddings: np.ndarray,
    num_speakers: int = None,
    min_speakers: int = 1,
    max_speakers: int = 7
) -> np.ndarray:
    """
    Agrupa embeddings por hablante usando Agglomerative Clustering.
    Usa Silhouette Score para determinar automáticamente el número óptimo de hablantes.
    
    Args:
        embeddings: Array de embeddings (N, 256)
        num_speakers: Número forzado de hablantes (opcional)
        
    Returns:
        Array de etiquetas (N,)
    """
    N = len(embeddings)
    logger.debug(f"[DBG cluster_speakers] N={N}, num_speakers={num_speakers}, type={type(num_speakers)}")
    
    # Si tenemos pocos embeddings, devolvemos todo 0
    if N < 2:
        logger.debug(f"[DBG] N < 2, retornando todos 0")
        labels = np.zeros(N, dtype=int)
        labels = smooth_speaker_labels(labels, window_size=5, min_segment_length=3)
        return labels
        
    try:
        # Modo Manual: K fijo (verifica que num_speakers sea un int > 0)
        if num_speakers is not None and num_speakers > 0:
            k = min(num_speakers, N)
            logger.debug(f"[DBG] Modo MANUAL: Forzando k={k}")
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            labels = smooth_speaker_labels(labels, window_size=5, min_segment_length=3)
            return labels
            
        # Modo Auto: Iterar K y buscar mejor Silhouette Score
        best_k = 1
        best_score = -1.0 # Silhouette va de -1 a 1
        best_labels = np.zeros(N, dtype=int)
        
        # Limitar rango de búsqueda
        search_max = min(max_speakers, N)
        if search_max < 2:
             labels = np.zeros(N, dtype=int)
             labels = smooth_speaker_labels(labels, window_size=5, min_segment_length=3)
             return labels

        # Probamos K desde min_speakers (o 2) hasta search_max
        # Si el usuario no especificó nada, asumimos que intenta buscar diferencias
        start_k = max(2, min_speakers) if min_speakers else 2
        
        logger.debug(f"Iniciando auto-clustering (búsqueda K={start_k}..{search_max})...")
        
        for k in range(start_k, search_max + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'
            )
            labels = model.fit_predict(embeddings)
            
            # Calcular score (métrica coseno para coincidir con el clustering)
            try:
                score = silhouette_score(embeddings, labels, metric='cosine')
            except:
                score = -1.0
                
            logger.debug(f"  K={k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        
        # Umbral mínimo de calidad para aceptar múltiples hablantes1
        MIN_SCORE_THRESHOLD = 0.05 
        
        if best_score > MIN_SCORE_THRESHOLD:
            logger.info(f"✔ Auto-clustering seleccionado: {best_k} hablantes (Score: {best_score:.4f})")
            best_labels = smooth_speaker_labels(best_labels, window_size=5, min_segment_length=3)
            return best_labels
        else:
            logger.warning(f"⚠ Score máximo ({best_score:.4f}) bajo. Fallback: intentando separar al menos 2 si score > 0.")
            # Fallback agresivo: si score positivo y detectó >1, úsalos
            if best_score > 0 and best_k > 1:
                 logger.info(f"  -> Forzando {best_k} hablantes por fallback (Score positivo).")
                 best_labels = smooth_speaker_labels(best_labels, window_size=5, min_segment_length=3)
                 return best_labels
            
            labels = np.zeros(N, dtype=int)
            labels = smooth_speaker_labels(labels, window_size=5, min_segment_length=3)
            return labels

    except Exception as e:
        logger.error(f"Error en clustering: {e}. Fallback a todos 0.", exc_info=True)
        # import traceback; traceback.print_exc() -> handled by exc_info=True
        labels = np.zeros(N, dtype=int)
        labels = smooth_speaker_labels(labels, window_size=5, min_segment_length=3)
        return labels


def detect_speaker_changes(
    embeddings: np.ndarray,
    similarity_threshold: float = CHANGE_SIM_THRESHOLD,
    num_speakers: int = None
) -> Tuple[Any, np.ndarray]:
    """
    Wrapper para mantener la firma, pero ahora usa clustering real.
    """
    labels = cluster_speakers(embeddings, num_speakers=num_speakers)
    return None, labels


def create_segments_from_labels(
    windows: List[Tuple[float, float]], 
    labels: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Convierte etiquetas frame-level a segmentos continuos.
    """
    segments = []
    if len(windows) == 0: return segments
    
    current_label = labels[0]
    start_time = windows[0][0]
    end_time = windows[0][1]
    
    for i in range(1, len(windows)):
        label = labels[i]
        w_start, w_end = windows[i]
        
        if label == current_label:
            # Extender segmento actual
            end_time = w_end
        else:
            # Guardar segmento anterior
            segments.append({
                "start": start_time, 
                "end": end_time, 
                "speaker": int(current_label)
            })
            # Iniciar nuevo
            current_label = label
            start_time = w_start
            end_time = w_end
            
    # Último segmento
    segments.append({
        "start": start_time, 
        "end": end_time, 
        "speaker": int(current_label)
    })
    
    return segments


def merge_consecutive_same_speaker(
    segments_with_text: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Fusiona segmentos consecutivos del mismo hablante.
    """
    if not segments_with_text: return []
    merged = []
    current = segments_with_text[0].copy()
    
    for nxt in segments_with_text[1:]:
        # Comparar speaker (int o str)
        s1 = current.get("speaker", 0)
        s2 = nxt.get("speaker", 0)
        # Normalizar a int si es posible para comparar
        try:
             if isinstance(s1, str) and "_" in s1: s1 = int(s1.split("_")[1])
        except: pass
        try:
             if isinstance(s2, str) and "_" in s2: s2 = int(s2.split("_")[1])
        except: pass

        if s1 == s2:
            # Merge
            current["end"] = max(current["end"], nxt["end"])
            txt1 = current.get("text_es") or current.get("text", "")
            txt2 = nxt.get("text_es") or nxt.get("text", "")
            full_txt = (txt1 + " " + txt2).strip()
            
            current["text_es"] = full_txt
            current["text"] = full_txt
        else:
            merged.append(current)
            current = nxt.copy()
            
    merged.append(current)
    return merged


def format_labeled_transcription(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        spk = s.get("speaker", 0)
        # Parse speaker if string
        if isinstance(spk, str) and spk.startswith("speaker_"):
            try: spk = int(spk.split("_")[1])
            except: spk = 0
            
        text = s.get("text_es") or s.get("text", "")
        if text:
            lines.append(f"[Hablante {int(spk)+1}]: {text}")
    return " ".join(lines)

def smooth_speaker_labels(labels: np.ndarray,
    window_size: int = 5,
    min_segment_length: int = 3
) -> np.ndarray:
     
     if len(labels) < window_size:
         return labels
         
     smoothed = labels.copy()
     
     #filtro de mediana para eliminar los cambios puntuales 
     from scipy.signal import medfilt
     smoothed = medfilt(smoothed.astype(float), kernel_size=window_size).astype(int)

     i=0
     while i < len(smoothed):
        current_label = smoothed[i]
        start = i
        while i+1 < len(smoothed) and smoothed[i+1] == current_label:
            i+=1

        segment_length = i - start + 1
        if segment_length < min_segment_length:
            prev_label = smoothed[start - 1] if start > 0 else current_label
            next_label = smoothed[i] if i < len(smoothed) else current_label
            majority_label = prev_label if prev_label == next_label else current_label
            smoothed[start:i] = majority_label

        return smoothed
