"""
Lógica de diarización (detección de hablantes) usando embeddings y change-point detection.
"""
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize

from config import WINDOW_SEC, CHANGE_SIM_THRESHOLD, MIN_SEG_SEC
import logging

logger = logging.getLogger(__name__)


def compute_embeddings(
    audio: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    encoder: VoiceEncoder,
    window_sec: float = WINDOW_SEC
) -> Tuple[np.ndarray, List[float]]:
    """
    Calcula embeddings de voz para ventanas de audio, ignorando silencio.

    Args:
        audio: Array de audio
        sr: Sample rate
        windows: Lista de tuplas (start, end) de ventanas
        encoder: VoiceEncoder de Resemblyzer
        window_sec: Tamaño de ventana en segundos

    Returns:
        Tuple de (embeddings normalizados, lista de starts validos)
    """
    embeddings = []
    starts = []
    win_len_samples = int(round(window_sec * sr))
    
    # Umbral de energía RMS para considerar que hay voz
    # Valores típicos: 0.005 a 0.02 dependiendo de normalización
    RMS_THRESHOLD = 0.005 

    for (s, e) in windows:
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        seg = audio[start_idx:end_idx]

        # Si segmento más corto que ventana, pad con ceros
        if len(seg) < win_len_samples:
            pad = np.zeros(win_len_samples - len(seg), dtype=seg.dtype)
            seg = np.concatenate([seg, pad])
            
        # Chequeo de energía (RMS)
        rms = np.sqrt(np.mean(seg**2))
        if rms < RMS_THRESHOLD:
            continue # Saltar silencio

        emb = encoder.embed_utterance(seg)
        embeddings.append(emb)
        starts.append(s)

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
    min_speakers: int = 2,
    max_speakers: int = 6
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
        return np.zeros(N, dtype=int)
        
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
            return clustering.fit_predict(embeddings)
            
        # Modo Auto: Iterar K y buscar mejor Silhouette Score
        best_k = 1
        best_score = -1.0 # Silhouette va de -1 a 1
        best_labels = np.zeros(N, dtype=int)
        
        # Limitar rango de búsqueda
        search_max = min(max_speakers, N)
        if search_max < 2:
             return np.zeros(N, dtype=int)

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
        
        # Umbral mínimo de calidad para aceptar múltiples hablantes
        # Si el score es muy bajo, probablemente sea un solo hablante con ruido
        # Reducido a 0.01 (en v4.1) para ser mas permisivo con voces similares
        MIN_SCORE_THRESHOLD = 0.01 
        
        if best_score > MIN_SCORE_THRESHOLD:
            logger.info(f"✔ Auto-clustering seleccionado: {best_k} hablantes (Score: {best_score:.4f})")
            return best_labels
        else:
            logger.warning(f"⚠ Score máximo ({best_score:.4f}) bajo. Fallback: intentando separar al menos 2 si score > 0.")
            # Fallback agresivo: si score positivo y detectó >1, úsalos
            if best_score > 0 and best_k > 1:
                 logger.info(f"  -> Forzando {best_k} hablantes por fallback (Score positivo).")
                 return best_labels
            
            return np.zeros(N, dtype=int)

    except Exception as e:
        logger.error(f"Error en clustering: {e}. Fallback a todos 0.", exc_info=True)
        # import traceback; traceback.print_exc() -> handled by exc_info=True
        return np.zeros(N, dtype=int)


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
