"""
Lógica de transcripción con Vosk.
"""
import json
import wave
from typing import Dict, List, Any, Tuple
from pathlib import Path

from vosk import Model, KaldiRecognizer
from config import VOSK_CHUNK_FRAMES


def transcribe_wav_with_vosk(model: Model, wav_path: Path) -> Dict[str, Any]:
    """
    Transcribe un archivo WAV usando Vosk.

    Args:
        model: Modelo Vosk cargado
        wav_path: Ruta al archivo WAV

    Returns:
        Diccionario con 'text' (transcripción completa) y 'raw' (resultados detallados)
    """
    wf = wave.open(str(wav_path), "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(VOSK_CHUNK_FRAMES)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))

    results.append(json.loads(rec.FinalResult()))
    wf.close()

    texts = [r.get("text", "") for r in results if r.get("text")]
    return {"text": " ".join(texts).strip(), "raw": results}


def transcribe_full_with_word_timestamps(
    wav_path: Path,
    model: Model
) -> List[Dict[str, Any]]:
    """
    Transcribe audio completo con timestamps de palabras.

    Args:
        wav_path: Ruta al archivo WAV
        model: Modelo Vosk cargado

    Returns:
        Lista de diccionarios con formato:
        [{'word': 'texto', 'start': 1.23, 'end': 1.45}, ...]
    """
    wf = wave.open(str(wav_path), "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    words = []
    while True:
        data = wf.readframes(VOSK_CHUNK_FRAMES)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            for w in res.get("result", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0))
                })

    final = json.loads(rec.FinalResult())
    for w in final.get("result", []):
        words.append({
            "word": w.get("word", ""),
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0))
        })

    wf.close()
    return words


def assign_words_to_segments(
    words: List[Dict[str, Any]],
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Asigna palabras a segmentos basándose en timestamps.

    Args:
        words: Lista de palabras con timestamps
        segments: Lista de segmentos con 'start' y 'end'

    Returns:
        Lista de segmentos con campo 'text' añadido
    """
    segments_sorted = sorted(segments, key=lambda s: s['start'])

    # Inicializar lista de palabras en cada segmento
    for s in segments_sorted:
        s['words'] = []

    for w in words:
        mid = (w['start'] + w['end']) / 2.0
        assigned = False

        # Intentar asignar a un segmento que contenga el punto medio
        for s in segments_sorted:
            if mid >= s['start'] - 1e-6 and mid <= s['end'] + 1e-6:
                s['words'].append(w['word'])
                assigned = True
                break

        # Si no se asignó, buscar el segmento más cercano
        if not assigned:
            best = None
            best_dist = None
            for s in segments_sorted:
                if mid < s['start']:
                    d = s['start'] - mid
                elif mid > s['end']:
                    d = mid - s['end']
                else:
                    d = 0.0

                if best is None or d < best_dist:
                    best = s
                    best_dist = d

            if best is not None:
                best['words'].append(w['word'])

    # Construir texto de cada segmento
    for s in segments_sorted:
        s['text'] = " ".join(s.get('words', [])).strip()

    return segments_sorted