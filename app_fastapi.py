# app_fastapi.py - VERSIÓN SIMPLIFICADA Y OPTIMIZADA
import uvicorn
import asyncio
import logging
import os
import tempfile
import functools
import shutil
import librosa
import numpy as np
import traceback
import time
import torch
import gc
import psutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

# Routers
from routes.history_routes import router as history_router
from routes.export_routes import router as export_router

# Importar rutas adicionales (cloud, export, sessions)
try:
    from routes.additional_routes import router as additional_router
    ADDITIONAL_ROUTES_AVAILABLE = True
except ImportError:
    ADDITIONAL_ROUTES_AVAILABLE = False
    logging.warning("additional_routes no disponible")

# Core modules
from core.models import load_whisper_models
from core.audio_processing import load_audio, write_wav_from_array
from core.speaker_diarization import diarize_audio, DiarizationResult   
from core.translation import translate_batch
from core.emotion_analysis import (
    analyze_audio_emotion, 
    analyze_text_emotion_es, 
    analyze_text_emotion_en, 
    EmotionResult,
    TemporalEmotionAnalyzer
)
from core.diarization import (
    compute_embeddings_with_vad,
    detect_speaker_changes,
    merge_consecutive_same_speaker,
    format_labeled_transcription
)
from resemblyzer import VoiceEncoder

# Config
from config import (
    MAX_UPLOAD_SIZE,
    ALLOWED_MIME,
    WORKERS,
    MAX_AUDIO_DURATION_SEC,
    WINDOW_SEC,
    HOP_SEC,
    CHANGE_SIM_THRESHOLD,
    MIN_SEG_SEC
)

# Resilience & Validators
from Resilience import (
    WHISPER_BREAKER,
    EMOTION_TEXT_BREAKER,
    retry_with_backoff_async
)
from Validators import AudioValidator, ParametersValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI (opcional)
try:
    from openai import OpenAI
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI no disponible. /transcribe/cloud-whisper no funcionará.")

from fastapi.responses import HTMLResponse

# GESTIÓN DE MEMORIA


def log_memory_usage() -> float:
    """Retorna el uso actual de memoria en MB."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"RAM: {mem_mb:.1f} MB")
    return mem_mb

def cleanup_memory():
    """Limpia memoria después de procesar."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# MÉTRICAS SIMPLES


_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "total_processing_time": 0.0
}

def increment_metric(key: str, value: float = 1.0):
    """Incrementa una métrica."""
    if key in _metrics:
        _metrics[key] += value

def get_metrics() -> Dict[str, Any]:
    """Retorna métricas actuales."""
    total = max(1, _metrics["requests_total"])
    success = _metrics["requests_success"]
    
    return {
        "requests_total": _metrics["requests_total"],
        "requests_success": success,
        "requests_failed": _metrics["requests_failed"],
        "success_rate": round(success / total * 100, 2),
        "avg_processing_time": round(_metrics["total_processing_time"] / max(1, success), 2)
    }

executor = ThreadPoolExecutor(max_workers=WORKERS)
_models_cache: Optional[Dict[str, Any]] = None
_voice_encoder: Optional[VoiceEncoder] = None

def load_models() -> Dict[str, Any]:
    """Carga modelos de Whisper (lazy loading)."""
    global _models_cache
    if _models_cache is None:
        _models_cache = load_whisper_models()
    return _models_cache

def get_voice_encoder() -> VoiceEncoder:
    """Carga VoiceEncoder bajo demanda."""
    global _voice_encoder
    if _voice_encoder is None:
        logger.info("Cargando VoiceEncoder para diarización...")
        _voice_encoder = VoiceEncoder()
        logger.info("VoiceEncoder cargado")
    return _voice_encoder

async def run_blocking(func, *args, **kwargs):
    """Ejecuta función bloqueante en threadpool."""
    loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

# FASTAPI APP


app = FastAPI(
    title="Voz-a-Texto Emocional API",
    description="API optimizada v5.0 con gestión inteligente de memoria",
    version="5.0.0"
)

# CORS - Configuración permisiva para desarrollo
# Nota: allow_credentials=True no funciona con wildcard "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen
    allow_credentials=False,  # Debe ser False cuando se usa "*"
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los headers
)

# Incluir routers
app.include_router(history_router)
app.include_router(export_router)

if ADDITIONAL_ROUTES_AVAILABLE:
    app.include_router(additional_router)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema."""
    logger.info("=" * 60)
    logger.info(" INICIANDO SISTEMA (Modo Optimizado RAM)")
    logger.info("=" * 60)
    
    # Precargar solo Whisper
    load_models() 
    logger.info("Whisper cargado")
    
    # Modelos de emoción se cargarán bajo demanda
    logger.info("ℹ Modelos de emoción: Carga bajo demanda")
    
    log_memory_usage()
    
    logger.info("=" * 60)
    logger.info(" SISTEMA LISTO: http://127.0.0.1:8000")
    logger.info("=" * 60)

# FUNCIONES AUXILIARES

async def save_upload_to_tempfile(upload_file: UploadFile) -> str:
    """Guarda archivo subido en temporal."""
    try:
        suffix = os.path.splitext(upload_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo: {e}")

def validate_upload(file: UploadFile):
    """Valida extensión del archivo."""
    ext = file.filename.lower().split('.')[-1] if file.filename else ""
    allowed = ["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: .{ext}. Use: {', '.join(allowed)}"
        )

async def validate_request_params(lite_mode: bool, audio_weight: float) -> Tuple[float, List[str], bool]:
    """Valida y ajusta parámetros de request."""
    audio_weight, warnings, lite_mode = ParametersValidator.validate_request_params(
        lite_mode, audio_weight
    )
    if warnings:
        logger.warning(f"Parámetros ajustados: {warnings}")
    return audio_weight, warnings, lite_mode

@retry_with_backoff_async(max_retries=2, base_delay=1.0)
async def safe_transcribe(model, path: str) -> Dict[str, Any]:
    """Transcribe audio con circuit breaker y retry."""
    def _wrapped_transcribe():
        return WHISPER_BREAKER.call(
            model.transcribe,
            path,
            language="es",
            fallback=lambda *a, **k: {"text": "", "segments": []}
        )
    return await run_blocking(_wrapped_transcribe)

async def perform_diarization(
    audio_data: np.ndarray, 
    sr: int, 
    num_speakers: Optional[int]
) -> Tuple[Dict[Tuple[float, float], int], int]:
    """
    Realiza diarización de hablantes.
    
    Returns:
        Tuple de (speaker_labels dict, num_speakers detectados)
    """
    speaker_labels = {}
    num_speakers_detected = 1
    
    if len(audio_data) == 0:
        return speaker_labels, num_speakers_detected
    
    try:
        logger.info("Ejecutando diarización de hablantes...")
        encoder = get_voice_encoder()
        
        duration = len(audio_data) / sr
        windows = []
        t = 0.0
        while t < duration:
            end_t = min(t + WINDOW_SEC, duration)
            windows.append((t, end_t))
            t += HOP_SEC
        
        if len(windows) < 2:
            logger.info("Audio muy corto para diarización")
            return speaker_labels, 1
        
        embeddings, starts = await run_blocking(
            compute_embeddings_with_vad, audio_data, sr, windows, encoder, WINDOW_SEC
        )
        
        if len(embeddings) < 2:
            logger.info("Pocos segmentos de voz detectados")
            return speaker_labels, 1
        
        _, labels = detect_speaker_changes(
            embeddings, 
            CHANGE_SIM_THRESHOLD,
            num_speakers=num_speakers
        )
        
        for i, start_t in enumerate(starts):
            end_t = start_t + WINDOW_SEC
            speaker_labels[(start_t, end_t)] = int(labels[i])
        
        num_speakers_detected = len(set(labels))
        logger.info(f"Detectados {num_speakers_detected} hablante(s)")
        
        return speaker_labels, num_speakers_detected
        
    except Exception as e:
        logger.warning(f"Error en diarización: {e}")
        return {}, 1

async def analyze_segments_with_emotions(
    segments_raw: List[Dict],
    audio_data: np.ndarray,
    sr: int,
    lite_mode: bool,
    audio_weight: float,
    speaker_labels: Dict[Tuple[float, float], int],
    enable_diarization: bool
) -> List[Dict[str, Any]]:
    """
    Analiza emociones para cada segmento.
    
    Returns:
        Lista de segmentos enriquecidos con análisis emocional
    """
    # Traducir todos los textos en batch
    segment_texts = [s.get("text", "").strip() for s in segments_raw]
    translations = []
    
    if not lite_mode and segment_texts:
        logger.info("Traduciendo segmentos...")
        translations = await run_blocking(translate_batch, segment_texts)
    
    # Analizar emociones
    enriched_segments = []
    emotion_analyzer = TemporalEmotionAnalyzer(use_prosody=not lite_mode)
    logger.info("Analizando emociones multimodales...")
    
    for i, seg in enumerate(segments_raw):
        try:
            seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
            seg_text_en = translations[i] if i < len(translations) else ""
            
            if not seg_text_es:
                continue
            
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            
            # Extraer audio del segmento
            seg_audio_path = None
            seg_chunk = None
            
            if not lite_mode and len(audio_data) > 0:
                try:
                    start_idx = max(0, int(start * sr))
                    end_idx = min(len(audio_data), int(end * sr))
                    
                    if (end_idx - start_idx) > (sr * 0.1):
                        seg_chunk = audio_data[start_idx:end_idx]
                        seg_audio_path = await run_blocking(write_wav_from_array, seg_chunk, sr)
                except Exception as e:
                    logger.warning(f"Error extrayendo audio seg {i}: {e}")
            
            # Analizar emoción
            emotion_result = emotion_analyzer.analyze_segment(
                text_es=seg_text_es,
                text_en=seg_text_en,
                audio_path=seg_audio_path,
                audio_array=seg_chunk,
                sr=sr,
                audio_weight=audio_weight,
                apply_smoothing=True,
                use_ensemble=True
            )
            
            # Limpiar archivo temporal
            if seg_audio_path and os.path.exists(seg_audio_path):
                try:
                    os.remove(seg_audio_path)
                except:
                    pass
            
            # Determinar speaker
            speaker_id = 0
            if enable_diarization and speaker_labels:
                seg_mid = (start + end) / 2
                for (win_s, win_e), spk in speaker_labels.items():
                    if win_s <= seg_mid < win_e:
                        speaker_id = spk
                        break
            
            # Construir segmento enriquecido
            enriched_segments.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "text_es": seg_text_es,
                "text_en": seg_text_en,
                "text": seg_text_es,
                "emotion": emotion_result.top_emotion,
                "intensity": round(emotion_result.top_score, 2),
                "emotions": {
                    "fused": {
                        "top_emotion": emotion_result.top_emotion,
                        "score": round(emotion_result.top_score, 4),
                        "all_emotions": emotion_result.emotions
                    }
                },
                "speaker_id": speaker_id,
                "speaker_label": f"Hablante {speaker_id + 1}",
                "speaker": f"speaker_{speaker_id}"
            })
            
        except Exception as e:
            logger.error(f"Error procesando segmento {i}: {e}")
            continue
    
    return enriched_segments

def calculate_global_stats(enriched_segments: List[Dict]) -> Dict[str, Any]:
    """Calcula estadísticas globales de emociones."""
    global_emotions = {}
    
    for seg in enriched_segments:
        fused = seg["emotions"]["fused"]["all_emotions"]
        for k, v in fused.items():
            global_emotions[k] = global_emotions.get(k, 0.0) + v
    
    total_w = sum(global_emotions.values())
    if total_w > 0:
        global_emotions = {k: round(v / total_w, 4) for k, v in global_emotions.items()}
    
    top_global = max(global_emotions, key=global_emotions.get) if global_emotions else "neutral"
    
    return {
        "top_emotion": top_global,
        "top_score": global_emotions.get(top_global, 0.0),
        "emotion_distribution": global_emotions
    }

def calculate_speaker_stats(enriched_segments: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """Calcula estadísticas por hablante."""
    speaker_stats = {}
    
    for seg in enriched_segments:
        spk_id = seg.get("speaker_id", 0)
        
        if spk_id not in speaker_stats:
            speaker_stats[spk_id] = {
                "label": seg.get("speaker_label", f"Hablante {spk_id + 1}"),
                "total_duration": 0.0,
                "segment_count": 0,
                "emotions": []
            }
        
        speaker_stats[spk_id]["total_duration"] += seg.get("duration", 0)
        speaker_stats[spk_id]["segment_count"] += 1
        
        emo = seg.get("emotion")
        if emo:
            speaker_stats[spk_id]["emotions"].append(emo)
    
    # Calcular emoción dominante
    for spk_id in speaker_stats:
        emos = speaker_stats[spk_id].pop("emotions")
        if emos:
            c = Counter(emos)
            speaker_stats[spk_id]["dominant_emotion"] = c.most_common(1)[0][0]
        else:
            speaker_stats[spk_id]["dominant_emotion"] = "neutral"
        
        speaker_stats[spk_id]["total_duration"] = round(speaker_stats[spk_id]["total_duration"], 2)
    
    return speaker_stats

def create_emotion_timeline(enriched_segments: List[Dict]) -> List[Dict]:
    """Crea timeline de emociones."""
    return [{
        "time": s["start"],
        "emotion": s["emotion"],
        "score": s["emotions"]["fused"]["score"],
        "all_emotions": s["emotions"]["fused"]["all_emotions"],
        "speaker_id": s.get("speaker_id", 0),
        "speaker_label": s.get("speaker_label", "Hablante 1")
    } for s in enriched_segments]

# HEALTH ENDPOINTS

@app.get("/health", tags=["Health"])
def health_check():
    """Health check básico."""
    return {"status": "ok", "version": "4.1.0", "mode": "OPTIMIZED"}

@app.get("/health/detailed", tags=["Health"])
def health_check_detailed():
    """Health check con métricas."""
    return {
        "status": "ok",
        "version": "4.1.0",
        "metrics": get_metrics(),
        "memory_mb": log_memory_usage()
    }

# ADMIN ENDPOINTS (GESTIÓN DE MEMORIA)

@app.post("/admin/cleanup-memory", tags=["Admin"])
async def cleanup_memory_endpoint():
    """Limpia memoria manualmente (útil para desarrollo)."""
    try:
        # Limpiar modelos de emoción si ModelManager está disponible
        try:
            from core.model_manager import cleanup_all_models
            cleanup_all_models()
            logger.info("Modelos de emoción limpiados")
        except ImportError:
            logger.info("ModelManager no disponible, solo limpieza básica")
        
        cleanup_memory()
        mem_after = log_memory_usage()
        
        return {
            "status": "success",
            "message": "Memoria limpiada",
            "memory_mb": mem_after
        }
    except Exception as e:
        logger.error(f"Error limpiando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/model-stats", tags=["Admin"])
async def model_stats():
    """Retorna estadísticas de modelos cargados."""
    try:
        stats = {
            "memory_mb": log_memory_usage(),
            "metrics": get_metrics()
        }
        
        # Agregar stats de ModelManager si está disponible
        try:
            from core.model_manager import get_model_manager
            manager = get_model_manager()
            stats["model_manager"] = manager.get_stats()
        except ImportError:
            stats["model_manager"] = "not_available"
        
        # Agregar stats de cache si está disponible
        try:
            from core.emotion_analysis import _result_cache
            stats["result_cache"] = _result_cache.get_stats()
        except (ImportError, AttributeError):
            stats["result_cache"] = "not_available"
        
        return stats
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TRANSCRIPTION ENDPOINTS

@app.post("/transcribe/full-analysis", tags=["Analysis"])
async def transcribe_full_analysis(
    file: UploadFile = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    """
    Análisis completo de audio con transcripción y análisis emocional.
    """
    validate_upload(file)
    start_time = time.time()
    increment_metric("requests_total")
    tmp_path = None
    
    try:
        # Validar parámetros
        audio_weight, warnings, lite_mode = await validate_request_params(lite_mode, audio_weight)
        
        # Guardar archivo temporal
        tmp_path = await save_upload_to_tempfile(file)
        
        # Validar audio
        validator = AudioValidator()
        validation_result = validator.validate_audio(tmp_path)
        
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Audio inválido",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        if validation_result.warnings:
            warnings.extend(validation_result.warnings)
        
        # Transcribir
        logger.info(f"Transcribiendo (lite={lite_mode}, audio_weight={audio_weight}, diarization={enable_diarization})...")
        models = load_models()
        whisper_model = models["whisper"]
        
        result = await safe_transcribe(whisper_model, tmp_path)
        tx_full = result.get("text", "")
        segments_raw = result.get("segments", [])
        
        if not segments_raw:
            raise HTTPException(status_code=422, detail="No se detectó voz en el audio")
        
        # Cargar audio completo para análisis
        audio_data = np.array([])
        sr = 16000
        
        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y
        
        # Diarización
        speaker_labels = {}
        num_speakers_detected = 1
        
        if enable_diarization and not lite_mode:
            speaker_labels, num_speakers_detected = await perform_diarization(
                audio_data, sr, num_speakers
            )
        
        # Analizar emociones
        enriched_segments = await analyze_segments_with_emotions(
            segments_raw,
            audio_data,
            sr,
            lite_mode,
            audio_weight,
            speaker_labels,
            enable_diarization
        )
        
        # Estadísticas globales
        global_emotions = calculate_global_stats(enriched_segments)
        speaker_stats = calculate_speaker_stats(enriched_segments) if enable_diarization else {}
        timeline = create_emotion_timeline(enriched_segments)
        
        # Transcripción con etiquetas de hablante
        labeled_transcription = ""
        if enable_diarization and not lite_mode:
            merged_blocks = merge_consecutive_same_speaker(enriched_segments)
            labeled_transcription = format_labeled_transcription(merged_blocks)
        
        # Resultado final
        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)
        
        # Limpiar memoria
        cleanup_memory()
        
        return {
            "status": "success",
            "mode": "full_analysis_v4.1",
            "transcription": tx_full,
            "labeled_transcription": labeled_transcription,
            "segments": enriched_segments,
            "global_emotions": global_emotions,
            "emotion_timeline": timeline,
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": num_speakers_detected,
                "speaker_stats": speaker_stats
            },
            "metadata": {
                "total_duration": validation_result.duration_sec,
                "processing_time": round(processing_time, 2),
                "warnings": warnings,
                "params": {
                    "lite_mode": lite_mode,
                    "audio_weight": audio_weight,
                    "diarization": enable_diarization
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        increment_metric("requests_failed")
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@app.post("/transcribe/cloud-whisper", tags=["Cloud"])
async def transcribe_with_cloud_whisper(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    """
    Transcripción con OpenAI Whisper Cloud API.
    
    Requiere API key de OpenAI válida.
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI no disponible. Instale: pip install openai httpx"
        )
    
    validate_upload(file)
    start_time = time.time()
    increment_metric("requests_total")
    tmp_path = None
    
    try:
        # Validar parámetros
        audio_weight, warnings, lite_mode = await validate_request_params(lite_mode, audio_weight)
        
        # Guardar archivo
        tmp_path = await save_upload_to_tempfile(file)
        
        # Validar audio
        validator = AudioValidator()
        validation_result = validator.validate_audio(tmp_path)
        
        if not validation_result.is_valid:
            raise HTTPException(status_code=400, detail=validation_result.errors)
        
        # Transcribir con OpenAI
        logger.info("Transcribiendo con OpenAI Whisper Cloud...")
        
        try:
            client = OpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(300.0, connect=60.0)
            )
            
            with open(tmp_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language="es",
                    timestamp_granularities=["segment"]
                )
        except Exception as e:
            logger.error(f"Error OpenAI API: {e}")
            raise HTTPException(status_code=500, detail=f"Error OpenAI API: {str(e)}")
        
        # Convertir respuesta a formato interno
        segments_raw = [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        } for seg in transcript_response.segments]
        
        tx_full = transcript_response.text
        logger.info(f"Transcripción completada: {len(segments_raw)} segmentos")
        
        # Cargar audio para análisis
        audio_data = np.array([])
        sr = 16000
        
        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y
        
        # Diarización
        speaker_labels = {}
        num_speakers_detected = 1
        
        if enable_diarization and not lite_mode:
            speaker_labels, num_speakers_detected = await perform_diarization(
                audio_data, sr, num_speakers
            )
        
        # Analizar emociones
        enriched_segments = await analyze_segments_with_emotions(
            segments_raw,
            audio_data,
            sr,
            lite_mode,
            audio_weight,
            speaker_labels,
            enable_diarization
        )
        
        # Estadísticas
        global_emotions = calculate_global_stats(enriched_segments)
        speaker_stats = calculate_speaker_stats(enriched_segments) if enable_diarization else {}
        timeline = create_emotion_timeline(enriched_segments)
        
        # Resultado
        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)
        
        cleanup_memory()
        
        return {
            "status": "success",
            "mode": "cloud_whisper",
            "transcription": tx_full,
            "segments": enriched_segments,
            "global_emotions": global_emotions,
            "emotion_timeline": timeline,
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": num_speakers_detected,
                "speaker_stats": speaker_stats
            },
            "metadata": {
                "total_duration": transcript_response.duration,
                "processing_time": round(processing_time, 2),
                "transcription_engine": "openai_whisper_cloud",
                "params": {
                    "lite_mode": lite_mode,
                    "audio_weight": audio_weight,
                    "diarization": enable_diarization
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        increment_metric("requests_failed")
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

# MAIN

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
