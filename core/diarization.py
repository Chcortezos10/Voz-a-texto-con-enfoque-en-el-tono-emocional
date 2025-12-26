"""
Lógica de diarización (detección de hablantes) usando embeddings y change-point detection.
"""
from typing import List, Tuple, Dict, Any

import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.preprocessing import normalize

from config import WINDOW_SEC, CHANGE_SIM_THRESHOLD, MIN_SEG_SEC


def compute_embeddings(
    audio: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    encoder: VoiceEncoder,
    window_sec: float = WINDOW_SEC
) -> Tuple[np.ndarray, List[float]]:
    """
    Calcula embeddings de voz para ventanas de audio.

    Args:
        audio: Array de audio
        sr: Sample rate
        windows: Lista de tuplas (start, end) de ventanas
        encoder: VoiceEncoder de Resemblyzer
        window_sec: Tamaño de ventana en segundos

    Returns:
        Tuple de (embeddings normalizados, lista de starts)
    """
    embeddings = []
    starts = []
    win_len_samples = int(round(window_sec * sr))

    for (s, e) in windows:
        start_idx = int(round(s * sr))
        end_idx = int(round(e * sr))
        seg = audio[start_idx:end_idx]

        # Si segmento más corto que ventana, pad con ceros
        if len(seg) < win_len_samples:
            pad = np.zeros(win_len_samples - len(seg), dtype=seg.dtype)
            seg = np.concatenate([seg, pad])

        emb = encoder.embed_utterance(seg)
        embeddings.append(emb)
        starts.append(s)

    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings)

    return embeddings, starts


def detect_speaker_changes(
    embeddings: np.ndarray,
    threshold: float = CHANGE_SIM_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecta cambios de hablante usando similitud coseno entre embeddings consecutivos.

    Args:
        embeddings: Array de embeddings normalizados
        threshold: Umbral de similitud para considerar cambio de hablante

    Returns:
        Tuple de (similitudes, etiquetas_secuenciales)
    """
    sims = []
    for i in range(len(embeddings)):
        if i == 0:
            sims.append(1.0)
        else:
            # Calcular similitud coseno
            sim = float(
                np.dot(embeddings[i], embeddings[i-1]) /
                (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-9)
            )
            sims.append(sim)

    sims = np.array(sims)

    # Detectar cambios donde similitud < threshold
    change_flags = sims < threshold
    change_flags[0] = False  # Primera ventana no es cambio

    # Asignar etiquetas secuenciales
    labels_seq = []
    current_label = 0
    for i, ch in enumerate(change_flags):
        if i == 0:
            labels_seq.append(current_label)
        else:
            if ch:
                current_label += 1
            labels_seq.append(current_label)

    return sims, np.array(labels_seq)


def create_segments_from_labels(
    labels: np.ndarray,
    starts: List[float],
    window_sec: float,
    duration: float,
    min_seg_sec: float = MIN_SEG_SEC
) -> List[Dict[str, Any]]:
    """
    Crea segmentos a partir de etiquetas de hablante.

    Args:
        labels: Array de etiquetas de hablante
        starts: Lista de tiempos de inicio de ventanas
        window_sec: Tamaño de ventana en segundos
        duration: Duración total del audio
        min_seg_sec: Segmentos mínimos a mantener

    Returns:
        Lista de segmentos con índice, start, end, speaker
    """
    segments = []
    for i, lab in enumerate(labels):
        s = starts[i]
        e = s + window_sec
        if e > duration:
            e = duration
        if e - s < min_seg_sec:
            continue

        segments.append({
            "index": i,
            "start": float(s),
            "end": float(e),
            "speaker": int(lab)
        })

    return segments


def merge_consecutive_same_speaker(
    segments_with_text: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Fusiona segmentos consecutivos del mismo hablante.

    Args:
        segments_with_text: Lista de segmentos con campos 'speaker' y 'text'

    Returns:
        Lista de bloques fusionados con 'speaker' y 'text'
    """
    merged = []
    for s in segments_with_text:
        sp = f"speaker_{s['speaker']}"
        txt = s.get('text', '').strip()

        if not merged:
            merged.append({"speaker": sp, "text": txt})
        else:
            if merged[-1]['speaker'] == sp:
                if txt:
                    merged[-1]['text'] = (merged[-1]['text'] + " " + txt).strip()
            else:
                merged.append({"speaker": sp, "text": txt})

    return merged


def format_labeled_transcription(merged_blocks: List[Dict[str, str]]) -> str:
    """
    Formatea bloques fusionados en transcripción con etiquetas.

    Args:
        merged_blocks: Lista de bloques con 'speaker' y 'text'

    Returns:
        Texto formateado con etiquetas [speaker_X]
    """
    labeled_paragraphs = []
    for block in merged_blocks:
        if block['text']:
            labeled_paragraphs.append(f"[{block['speaker']}] {block['text']}")
        else:
            labeled_paragraphs.append(f"[{block['speaker']}] (sin texto)")

    return "\n\n".join(labeled_paragraphs).strip()