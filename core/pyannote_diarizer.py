"""
Diarización usando Resemblyzer VoiceEncoder + WebRTC VAD + Agglomerative Clustering.
Restaurado del sistema original del proyecto.

Pipeline:
1. Crear ventanas deslizantes sobre TODO el audio (0.9s, hop 0.25s)
2. Filtrar ventanas sin voz (WebRTC VAD)
3. Calcular embeddings con Resemblyzer VoiceEncoder
4. Clustering aglomerativo con Silhouette Score automático
5. Suavizar labels con filtro de mediana
6. Asignar speakers a segmentos de WhisperX por overlap temporal
"""
import logging
import gc
import numpy as np
import webrtcvad
from typing import List, Dict, Any, Optional, Tuple

from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.signal import medfilt

from config import (
    WINDOW_SEC,
    CHANGE_SIM_THRESHOLD,
    MIN_SEG_SEC,
    VOICE_RATIO_THRESHOLD,
    DIARIZATION_SMOOTH_WINDOW,
    DIARIZATION_MIN_SEGMENT,
    HOP_SEC,
)

logger = logging.getLogger(__name__)

# =============================================
# Singleton del VoiceEncoder
# =============================================
_encoder: Optional[VoiceEncoder] = None


def _get_encoder() -> VoiceEncoder:
    """Retorna el VoiceEncoder singleton (lazy load)."""
    global _encoder
    if _encoder is None:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cargando Resemblyzer VoiceEncoder en {device_str}...")
        _encoder = VoiceEncoder(device_str)
        logger.info("VoiceEncoder cargado.")
    return _encoder


# =============================================
# Crear ventanas deslizantes sobre TODO el audio
# =============================================
def _create_sliding_windows(
    audio_duration: float,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC
) -> List[Tuple[float, float]]:
    """
    Crea ventanas deslizantes que cubren TODO el audio.
    Esto es clave para la resolución de la diarización.
    
    Con window_sec=0.9 y hop_sec=0.25:
    - Un audio de 60s genera ~236 ventanas
    - Cada ventana tiene 0.9s de audio
    - Las ventanas se superponen 0.65s
    """
    windows = []
    t = 0.0
    while t + window_sec <= audio_duration:
        windows.append((t, t + window_sec))
        t += hop_sec
    # Última ventana parcial si queda audio
    if t < audio_duration and audio_duration - t > 0.2:
        windows.append((t, audio_duration))
    return windows


# =============================================
# Embeddings con VAD
# =============================================
def compute_embeddings_with_vad(
    audio: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    encoder: VoiceEncoder,
    vad_aggressiveness: int = 2
) -> Tuple[np.ndarray, List[int]]:
    """
    Calcula embeddings de voz con filtro VAD (WebRTC).
    Solo procesa ventanas con suficiente actividad de voz.
    """
    embeddings = []
    valid_indices = []

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_duration_ms = 30
    frame_size = int(sr * frame_duration_ms / 1000)

    for idx, (s, e) in enumerate(windows):
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        seg = audio[start_idx:end_idx]

        if len(seg) < frame_size:
            continue

        # VAD: verificar actividad de voz
        seg_int16 = (seg * 32767).astype(np.int16)
        num_frames = len(seg_int16) // frame_size
        voiced_frames = 0

        for i in range(num_frames):
            frame = seg_int16[i * frame_size:(i + 1) * frame_size]
            if len(frame) == frame_size:
                try:
                    if vad.is_speech(frame.tobytes(), sr):
                        voiced_frames += 1
                except:
                    pass

        voice_ratio = voiced_frames / max(1, num_frames)
        if voice_ratio < VOICE_RATIO_THRESHOLD:
            continue

        # Calcular embedding
        try:
            emb = encoder.embed_utterance(seg)
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as ex:
            logger.warning(f"Error embedding ventana {s:.2f}-{e:.2f}: {ex}")

    if not embeddings:
        return np.array([]), []

    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings)

    return embeddings, valid_indices


# =============================================
# Suavizado de etiquetas
# =============================================
def smooth_speaker_labels(
    labels: np.ndarray,
    window_size: int = DIARIZATION_SMOOTH_WINDOW,
    min_segment_length: int = DIARIZATION_MIN_SEGMENT
) -> np.ndarray:
    """Suaviza cambios de speaker usando filtro de mediana."""
    if len(labels) < 3:
        return labels

    smoothed = labels.copy()

    # Filtro de mediana para eliminar cambios puntuales
    kernel = window_size if window_size % 2 == 1 else window_size + 1
    kernel = max(3, kernel)  # mínimo 3
    smoothed = medfilt(smoothed.astype(float), kernel_size=kernel).astype(int)

    # Eliminar segmentos muy cortos
    i = 0
    while i < len(smoothed):
        current_label = smoothed[i]
        start = i
        while i + 1 < len(smoothed) and smoothed[i + 1] == current_label:
            i += 1

        segment_length = i - start + 1
        if segment_length < min_segment_length:
            prev_label = smoothed[start - 1] if start > 0 else current_label
            next_label = smoothed[i + 1] if i + 1 < len(smoothed) else current_label
            majority_label = prev_label if prev_label == next_label else current_label
            smoothed[start:i + 1] = majority_label

        i += 1  # Avanzar al siguiente segmento

    return smoothed  # Retornar DESPUÉS de procesar todos los segmentos


# =============================================
# Clustering de hablantes
# =============================================
def cluster_speakers(
    embeddings: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 7
) -> np.ndarray:
    """
    Agrupa embeddings por hablante usando Agglomerative Clustering.
    Si num_speakers es None, busca el K óptimo con Silhouette Score.
    """
    N = len(embeddings)
    logger.debug(f"[cluster_speakers] N={N}, num_speakers={num_speakers}")

    if N < 2:
        labels = np.zeros(N, dtype=int)
        return smooth_speaker_labels(labels)

    try:
        # Modo manual: K fijo
        if num_speakers is not None and num_speakers > 0:
            k = min(num_speakers, N)
            logger.info(f"Clustering con K fijo = {k}")
            clustering = AgglomerativeClustering(
                n_clusters=k, metric='cosine', linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            return smooth_speaker_labels(labels, window_size=5, min_segment_length=3)

        # Modo automático: iterar K y buscar mejor Silhouette Score
        best_k = 1
        best_score = -1.0
        best_labels = np.zeros(N, dtype=int)

        search_max = min(max_speakers, N)
        if search_max < 2:
            return smooth_speaker_labels(np.zeros(N, dtype=int))

        start_k = max(2, min_speakers) if min_speakers else 2
        logger.info(f"Auto-clustering: buscando K óptimo ({start_k}..{search_max})...")

        for k in range(start_k, search_max + 1):
            model = AgglomerativeClustering(
                n_clusters=k, metric='cosine', linkage='average'
            )
            labels = model.fit_predict(embeddings)

            try:
                score = silhouette_score(embeddings, labels, metric='cosine')
            except:
                score = -1.0

            logger.debug(f"  K={k}: Silhouette = {score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        MIN_SCORE_THRESHOLD = 0.005

        if best_score > MIN_SCORE_THRESHOLD:
            logger.info(f"✔ Auto-clustering: {best_k} hablantes (Score: {best_score:.4f})")
            return smooth_speaker_labels(best_labels, window_size=5, min_segment_length=3)
        else:
            # Fallback: si detectó >1 cluster con score positivo, usarlos
            if best_score > 0 and best_k > 1:
                logger.info(f"Fallback: {best_k} hablantes (Score positivo: {best_score:.4f})")
                return smooth_speaker_labels(best_labels, window_size=5, min_segment_length=3)
            # Si score negativo con >1 cluster, intentar con 2 forzado
            if best_k > 1 and N >= 4:
                logger.info(f"Fallback forzado: intentando 2 hablantes")
                model = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
                labels_2 = model.fit_predict(embeddings)
                return smooth_speaker_labels(labels_2, window_size=5, min_segment_length=3)

            logger.warning(f"Score bajo ({best_score:.4f}), asignando 1 hablante")
            return smooth_speaker_labels(np.zeros(N, dtype=int))

    except Exception as e:
        logger.error(f"Error en clustering: {e}", exc_info=True)
        return smooth_speaker_labels(np.zeros(N, dtype=int))


# =============================================
# Función principal de diarización
# =============================================
def diarize_segments(
    audio_file: str,
    transcription_segments: List[Dict[str, Any]],
    num_speakers: Optional[int] = None,
    sr: int = 16000
) -> List[Dict[str, Any]]:
    """
    Diariza usando ventanas deslizantes sobre TODO el audio (como el sistema original).
    
    Pipeline:
    1. Cargar audio
    2. Crear ventanas deslizantes (0.9s, hop 0.25s) sobre TODO el audio
    3. Filtrar con VAD, calcular embeddings con Resemblyzer  
    4. Clustering para determinar speakers
    5. Mapear labels de ventanas a segmentos de transcripción por overlap
    """
    import librosa

    if not transcription_segments:
        return transcription_segments

    try:
        # 1. Cargar audio
        logger.info(f"Diarización Resemblyzer: cargando audio...")
        y, sr = librosa.load(audio_file, sr=sr)
        duration = len(y) / sr
        logger.info(f"Audio: {duration:.1f}s, {len(transcription_segments)} segmentos de transcripción")

        # 2. Crear ventanas deslizantes sobre TODO el audio
        #    Esto es la clave: ventanas finas (0.9s) con paso corto (0.25s)
        #    dan mucha mejor resolución que usar los segmentos de WhisperX directamente
        windows = _create_sliding_windows(duration)
        logger.info(f"Ventanas deslizantes: {len(windows)} ventanas de {WINDOW_SEC}s con hop {HOP_SEC}s")

        if not windows:
            logger.warning("No se generaron ventanas")
            for seg in transcription_segments:
                seg["speaker"] = "SPEAKER_00"
            return transcription_segments

        # 3. Calcular embeddings con VAD
        encoder = _get_encoder()
        embeddings, valid_indices = compute_embeddings_with_vad(y, sr, windows, encoder)
        logger.info(f"Embeddings VAD válidos: {len(embeddings)}/{len(windows)}")

        if len(embeddings) < 2:
            logger.warning(f"Solo {len(embeddings)} embeddings válidos, asignando speaker único")
            for seg in transcription_segments:
                seg["speaker"] = "SPEAKER_00"
            return transcription_segments

        # 4. Clustering
        labels = cluster_speakers(embeddings, num_speakers=num_speakers)
        unique_speakers = len(set(labels))
        logger.info(f"Clustering: {unique_speakers} hablantes en {len(labels)} ventanas válidas")

        # 5. Crear mapeo de ventanas válidas → speaker  
        window_speaker_map = []
        for i, window_idx in enumerate(valid_indices):
            ws, we = windows[window_idx]
            window_speaker_map.append({
                "start": ws,
                "end": we,
                "speaker": int(labels[i])
            })

        # 6. Asignar speaker a cada segmento de transcripción por overlap
        #    Para cada segmento, votamos entre las ventanas que se superponen
        for seg in transcription_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            speaker_votes = {}
            for wm in window_speaker_map:
                overlap_start = max(seg_start, wm["start"])
                overlap_end = min(seg_end, wm["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0:
                    spk = wm["speaker"]
                    speaker_votes[spk] = speaker_votes.get(spk, 0) + overlap

            if speaker_votes:
                best_speaker = max(speaker_votes, key=speaker_votes.get)
                seg["speaker"] = f"SPEAKER_{best_speaker:02d}"
            else:
                seg["speaker"] = "SPEAKER_00"

        final_speakers = len(set(s.get("speaker", "SPEAKER_00") for s in transcription_segments))
        logger.info(f"✔ Diarización completada: {final_speakers} hablantes en {len(transcription_segments)} segmentos")

        return transcription_segments

    except Exception as e:
        logger.error(f"Error en diarización Resemblyzer: {e}", exc_info=True)
        for seg in transcription_segments:
            if "speaker" not in seg:
                seg["speaker"] = "SPEAKER_00"
        return transcription_segments


# =============================================
# API para app_fastapi.py
# =============================================
def get_local_diarizer():
    """Compatibilidad con la interfaz existente en app_fastapi.py."""
    return _DiarizationWrapper()


class _DiarizationWrapper:
    """Wrapper para mantener la interfaz esperada por app_fastapi.py."""
    
    def diarize_from_segments(
        self,
        audio_file: str,
        transcription_segments: List[Dict[str, Any]],
        num_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return diarize_segments(audio_file, transcription_segments, num_speakers)
