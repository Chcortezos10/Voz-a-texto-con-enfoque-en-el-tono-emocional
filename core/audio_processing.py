"""
Funciones de procesamiento de audio compartidas.
Incluye carga, remuestreo, VAD, y generación de ventanas.
"""
import io
import wave
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, BinaryIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import webrtcvad

from config import TARGET_SR, VAD_MODE, VAD_FRAME_MS, WINDOW_SEC, HOP_SEC, VAD_MERGE_GAP_SEC, PCM16_MAX


def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SR
) -> Tuple[np.ndarray, int]:
    """
    Carga un archivo de audio y lo convierte a mono con el sample rate objetivo.

    Args:
        path: Ruta al archivo de audio
        target_sr: Sample rate objetivo (default: 16000)

    Returns:
        Tuple de (audio_array, sample_rate)
    """
    data, sr = sf.read(str(path), always_2d=True)
    data = data.astype("float32")

    # Mezclar a mono
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    data = data[:, 0]

    # Remuestrear si es necesario
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down)

    # Normalizar
    max_abs = np.max(np.abs(data)) if data.size else 0.0
    if max_abs > 0:
        data = data / max_abs

    return data, target_sr


def load_audio_bytes(
    source: Union[bytes, bytearray, str, Path, BinaryIO],
    target_sr: int = TARGET_SR
) -> Tuple[np.ndarray, int]:
    """
    Carga audio desde bytes, path o file-like object.

    Args:
        source: bytes-like, path-like o file-like object
        target_sr: Sample rate objetivo (default: 16000)

    Returns:
        Tuple de (audio_array, sample_rate)
    """
    if isinstance(source, (bytes, bytearray)):
        buf = io.BytesIO(source)
        data, sr = sf.read(buf, always_2d=True)
    elif hasattr(source, "read"):
        data, sr = sf.read(source, always_2d=True)
    else:
        data, sr = sf.read(str(source), always_2d=True)

    data = data.astype("float32")
    if data.ndim == 2 and data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    if data.ndim == 2:
        data = data[:, 0]

    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down)

    max_abs = np.max(np.abs(data)) if data.size else 0.0
    if max_abs > 0:
        data = data / max_abs

    return data, target_sr


def pcm16_bytes_from_float32(arr: np.ndarray) -> bytes:
    """
    Convierte array float32 normalizado a bytes PCM16.

    Args:
        arr: Array de audio normalizado (-1.0 a 1.0)

    Returns:
        Bytes en formato PCM16
    """
    pcm16 = (arr * 32767).astype(np.int16)
    return pcm16.tobytes()


def write_wav_pcm16(path: Union[str, Path], pcm16_array: np.ndarray, sr: int) -> None:
    """
    Escribe un archivo WAV en formato PCM16.

    Args:
        path: Ruta del archivo de salida
        pcm16_array: Array de int16 con los samples
        sr: Sample rate
    """
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16_array.tobytes())


def write_wav_from_array(arr: np.ndarray, sr: int) -> str:
    """
    Escribe un array de audio a un archivo temporal WAV.

    Args:
        arr: Array de audio normalizado
        sr: Sample rate

    Returns:
        Ruta del archivo temporal creado
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm16 = (arr * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    return tmp.name


def get_voiced_regions(
    audio: np.ndarray,
    sr: int,
    vad_mode: int = VAD_MODE,
    frame_ms: int = VAD_FRAME_MS
) -> List[Tuple[float, float]]:
    """
    Detecta regiones con voz usando WebRTC VAD.

    Args:
        audio: Array de audio
        sr: Sample rate (debe ser 16000 para webrtcvad)
        vad_mode: Modo de agresividad (0-3)
        frame_ms: Tamaño del frame en ms (10, 20, 30)

    Returns:
        Lista de tuplas (start_s, end_s) en segundos

    Raises:
        RuntimeError: Si sr != 16000
    """
    vad = webrtcvad.Vad(vad_mode)

    if sr != 16000:
        raise RuntimeError("webrtcvad requiere audio a 16000 Hz")

    # Convertir a PCM16
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    frame_bytes = int(sr * (frame_ms / 1000.0) * 2)  # bytes por frame
    num_frames = len(pcm16) // frame_bytes

    voiced_flags = []
    for i in range(num_frames):
        start = i * frame_bytes
        frame = pcm16[start:start + frame_bytes]
        is_speech = vad.is_speech(frame, sample_rate=sr)
        voiced_flags.append(is_speech)

    # Convertir flags a regiones en segundos
    regions = []
    if not voiced_flags:
        return regions

    cur_state = voiced_flags[0]
    cur_start = 0.0
    for i, flag in enumerate(voiced_flags):
        t = i * (frame_ms / 1000.0)
        if flag != cur_state:
            if cur_state:
                regions.append((cur_start, t))
            cur_state = flag
            cur_start = t

    # Cerrar última región si es voz
    end_t = num_frames * (frame_ms / 1000.0)
    if cur_state:
        regions.append((cur_start, end_t))

    # Merge regiones muy cercanas (gap <= 200ms)
    merged = []
    for s, e in regions:
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= VAD_MERGE_GAP_SEC:  # gap <= VAD_MERGE_GAP_SEC -> unir
                merged[-1][1] = e
            else:
                merged.append([s, e])

    return [(float(s), float(e)) for s, e in merged]


def windows_within_regions(
    regions: List[Tuple[float, float]],
    audio_len_s: float,
    window_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC
) -> List[Tuple[float, float]]:
    """
    Genera ventanas deslizantes dentro de regiones de voz.

    Args:
        regions: Lista de tuplas (start, end) de regiones con voz
        audio_len_s: Duración total del audio en segundos
        window_sec: Tamaño de ventana en segundos
        hop_sec: Salto entre ventanas en segundos

    Returns:
        Lista de tuplas (start, end) de ventanas
    """
    windows = []

    for (rstart, rend) in regions:
        region_len = rend - rstart
        if region_len <= 0:
            continue

        # Si la región es más corta que la ventana, centrar la ventana
        if region_len <= window_sec:
            start = max(0.0, rstart)
            end = min(audio_len_s, rstart + window_sec)
            windows.append((start, end))
            continue

        # Generar ventanas con hop
        t = rstart
        while t + window_sec <= rend + 1e-6:
            windows.append((t, t + window_sec))
            t += hop_sec

        # Tail: si queda un trozo al final, añadir ventana final
        if windows and windows[-1][1] < rend - 1e-6:
            last_end = windows[-1][1]
            if rend - last_end >= 0.5:
                windows.append((rend - window_sec, rend))

    # Ordenar y deduplicar ventanas muy similares
    windows_sorted = sorted(windows, key=lambda x: x[0])
    merged = []
    for s, e in windows_sorted:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s <= ps + 1e-6 and abs(e - pe) < 1e-3:
                continue
            if s <= pe - 0.01:  # Solapamiento fuerte -> extender
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))

    return merged

def validate_audio_energy(audio: np.ndarray, sr: int) -> tuple:
    #valida que el audio tenga energia 

    if audio.size == 0:
        return False,0.0,"Audio vacio"

    rms = np.sqrt(np.mean(audio**2))
    if rms < min_rms:
        return False,rms,f"Audio sin energia, rms={rms:.4f}"

    peak = np.max(np.abs(audio))
    warning = None
    if peak > 0.99:
        warning = "audio posiblemente saturado, max={peak:.4f}"
    return True,rms,warning
    