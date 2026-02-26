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
import io
import json
import psutil
from core.feedback_system import FeedbackCollector
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter

# Routers
from routes.history_routes import router as history_router
from routes.export_routes import router as export_router
from routes.scoring_routes import router as scoring_router
from routes.alert_routes import router as alert_router
from routes.kpi_routes import router as kpi_router

# Importar rutas adicionales (cloud, export, sessions)
try:
    from routes.additional_routes import router as additional_router
    ADDITIONAL_ROUTES_AVAILABLE = True
except ImportError:
    ADDITIONAL_ROUTES_AVAILABLE = False
    logging.warning("additional_routes no disponible")

# Core modules
from core.audio_processing import load_audio, write_wav_from_array
from core.translation import translate_batch
from core.emotion_analysis import (
    analyze_audio_emotion, 
    analyze_text_emotion_es, 
    analyze_text_emotion_en, 
    EmotionResult,
    TemporalEmotionAnalyzer
)
from core.whisperx_service import WhisperXService
from core.pyannote_diarizer import get_local_diarizer
from core.diarization import (
    merge_consecutive_same_speaker,
    format_labeled_transcription
)
from core.scoring_engine import calculate_quality_score
from core.alert_system import check_alerts
from core.call_summary import generate_call_summary

# Config
from config import (
    MAX_UPLOAD_SIZE,
    ALLOWED_MIME,
    WORKERS,
    MAX_AUDIO_DURATION_SEC
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

async def run_blocking(func, *args, **kwargs):
    """Ejecuta una función bloqueante en un thread pool para no bloquear el loop principal."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

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
_whisperx_service: Optional[WhisperXService] = None

def get_whisperx_service() -> WhisperXService:
    """Carga WhisperX Service bajo demanda."""
    global _whisperx_service
    if _whisperx_service is None:
        logger.info("Inicializando WhisperX Service...")
        _whisperx_service = WhisperXService()
    return _whisperx_service

# Funciones obsoletas eliminadas por migración a WhisperX

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

feedback_collector = FeedbackCollector(storage_dir="./feedback_data")

from pydantic import BaseModel

class ValidatePredictionRequest(BaseModel):
    prediction_id: str
    correct_label: str
    comment: Optional[str] = None

@app.post("/validate_prediction", tags=["Feedback"])
async def validate_prediction(request: ValidatePredictionRequest):
    """
    Permite al usuario corregir una prediccion de emocion.
    """
    valid_labels = ["feliz", "triste", "enojado", "neutral"]
    if request.correct_label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Label invalido. Debe ser: {', '.join(valid_labels)}"
        )
    
    try:
        result = feedback_collector.validate_prediction(
            prediction_id=request.prediction_id,
            correct_label=request.correct_label,
            user_comment=request.comment
        )
        
        validated_count = feedback_collector.get_validated_count()
        
        logger.info(
            f"Feedback: {request.prediction_id} -> {request.correct_label} "
            f"(Total: {validated_count})"
        )
        
        return {
            "status": "success",
            "message": "Gracias por tu feedback",
            "was_correct": result["predicted_label"] == result["correct_label"],
            "total_validated": validated_count,
            "ready_for_retraining": validated_count >= 100
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error validando prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats", tags=["Feedback"])
async def get_feedback_stats():
    """Obtiene estadísticas del sistema de feedback."""
    try:
        return {
            "validated": feedback_collector.get_validated_count(),
            "pending": feedback_collector.get_pending_count(),
            "ready_for_retraining": feedback_collector.get_validated_count() >= 100,
            "threshold": 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Incluir routers
app.include_router(history_router)
app.include_router(export_router)
app.include_router(scoring_router)
app.include_router(alert_router)
app.include_router(kpi_router)

if ADDITIONAL_ROUTES_AVAILABLE:
    app.include_router(additional_router)

# Configuración del logo para exportación PDF
_company_logo_path: Optional[str] = None

@app.post("/config/logo", tags=["Config"])
async def upload_company_logo(file: UploadFile = File(...)):
    """Sube un logo de empresa para incluir en reportes PDF."""
    global _company_logo_path
    allowed_ext = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}. Use: {', '.join(allowed_ext)}")
    logo_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(logo_dir, exist_ok=True)
    logo_path = os.path.join(logo_dir, f"company_logo{ext}")
    with open(logo_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _company_logo_path = logo_path
    return {"status": "success", "message": "Logo guardado", "path": logo_path}

@app.get("/config/logo", tags=["Config"])
async def get_company_logo_status():
    """Verifica si hay un logo configurado."""
    if _company_logo_path and os.path.exists(_company_logo_path):
        return {"has_logo": True, "path": _company_logo_path}
    return {"has_logo": False}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema."""
    logger.info("=" * 60)
    logger.info(" INICIANDO SISTEMA VOZ-A-TEXTO EMOCIONAL v5.0")
    logger.info("=" * 60)
    
    # Detectar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✅ GPU DETECTADA: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.warning("⚠️  GPU NO DETECTADA - Usando CPU (procesamiento más lento)")
    
    # Precargar WhisperX
    try:
        get_whisperx_service()
        logger.info("✅ WhisperX Service inicializado")
    except Exception as e:
        logger.error(f"❌ Error inicializando WhisperX: {e}")
    
    # Modelos de emoción se cargarán bajo demanda
    logger.info("ℹ️  Modelos de emoción: Carga bajo demanda")
    
    log_memory_usage()
    
    logger.info("=" * 60)
    logger.info("🚀 SERVIDOR LISTO")
    logger.info("   Dashboard: http://127.0.0.1:8000/")
    logger.info("   API Docs:  http://127.0.0.1:8000/docs")
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


class _MockUploadFile:
    """Envuelve bytes en un objeto compatible con UploadFile para batch SSE."""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)

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

# Removed safe_transcribe and perform_diarization

async def analyze_segments_with_emotions(
    segments_raw: List[Dict],
    audio_data: np.ndarray,
    sr: int,
    lite_mode: bool,
    audio_weight: float,
    speaker_labels: Dict[Tuple[float, float], int],
    enable_diarization: bool
) -> List[Dict[str, Any]]:
    segment_texts = [s.get("text", "").strip() for s in segments_raw]
    translations = []

    if not lite_mode and segment_texts:
        logger.info("Traduciendo segmentos...")
        translations = await run_blocking(translate_batch, segment_texts)

    enriched_segments = []
    emotion_analyzer = TemporalEmotionAnalyzer(use_prosody=not lite_mode)
    logger.info("Analizando emociones multimodales...")

    async def _analyze_single_segment(i, seg):
        seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
        seg_text_en = translations[i] if i < len(translations) else ""

        if not seg_text_es:
            return None

        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))

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

        if seg_audio_path and os.path.exists(seg_audio_path):
            try:
                os.remove(seg_audio_path)
            except:
                pass

        del seg_chunk

        speaker_id = 0
        # Check if segment already has speaker info (WhisperX)
        if "speaker" in seg:
            try:
                # Expecting "SPEAKER_01" -> 1
                spk_str = seg["speaker"]
                speaker_id = int(spk_str.split("_")[-1])
            except:
                speaker_id = 0
        elif enable_diarization and speaker_labels:
            seg_mid = (start + end) / 2
            for (win_s, win_e), spk in speaker_labels.items():
                if win_s <= seg_mid < win_e:
                    speaker_id = spk
                    break

        feedback_audio_path = None
        if not lite_mode and len(audio_data) > 0:
            try:
                ctx_end_sec = end
                ctx_start_sec = max(0, ctx_end_sec - 10.0)
                ctx_start_idx = int(ctx_start_sec * sr)
                ctx_end_idx = int(ctx_end_sec * sr)
                if ctx_end_idx > ctx_start_idx:
                    ctx_chunk = audio_data[ctx_start_idx:ctx_end_idx]
                    feedback_audio_path = await run_blocking(write_wav_from_array, ctx_chunk, sr)
            except Exception as e:
                logger.warning(f"Error prepping feedback audio: {e}")

        prediction_id = feedback_collector.save_prediction(
            predicted_label=emotion_result.top_emotion,
            confidence=emotion_result.top_score,
            audio_temp_path=feedback_audio_path,
            metadata={
                "text_es": seg_text_es,
                "text_en": seg_text_en,
                "start": start,
                "end": end
            }
        )

        if feedback_audio_path and os.path.exists(feedback_audio_path):
            try:
                os.remove(feedback_audio_path)
            except:
                pass

        return {
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
            "speaker": f"speaker_{speaker_id}",
            "prediction_id": prediction_id
        }

    BATCH_SIZE = 3
    for batch_start in range(0, len(segments_raw), BATCH_SIZE):
        batch = list(enumerate(segments_raw[batch_start:batch_start + BATCH_SIZE], start=batch_start))
        tasks = [_analyze_single_segment(i, seg) for i, seg in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, dict):
                enriched_segments.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error procesando segmento: {result}")

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
    Análisis completo de audio con transcripción y análisis emocional (WhisperX).
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
        
        # 1. Transcribir con WhisperX (solo transcripción + alineación)
        logger.info(f"Procesando con WhisperX (lite={lite_mode}, audio_weight={audio_weight})...")
        
        service = get_whisperx_service()
        
        # Ejecutar en threadpool para no bloquear event loop
        wx_result = await run_blocking(service.process_audio, tmp_path)
        
        segments_raw = wx_result["segments"]
        tx_full = " ".join([s["text"].strip() for s in segments_raw])
        
        # 2. Diarización local (MFCC + clustering, sin HuggingFace)
        if enable_diarization:
            try:
                logger.info("Ejecutando diarización local (MFCC + clustering)...")
                diarizer = get_local_diarizer()
                segments_raw = await run_blocking(
                    diarizer.diarize_from_segments, tmp_path, segments_raw, num_speakers
                )
                unique_spk = len(set(s.get("speaker", "SPEAKER_00") for s in segments_raw))
                logger.info(f"Diarización completada: {unique_spk} hablantes detectados")
            except Exception as diar_err:
                logger.warning(f"Diarización falló, continuando sin ella: {diar_err}")
                for seg in segments_raw:
                    if "speaker" not in seg:
                        seg["speaker"] = "SPEAKER_00"
        else:
            for seg in segments_raw:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"
        
        # 3. Cargar audio completo para análisis emocional
        audio_data = np.array([])
        sr = 16000
        
        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y
        
        # 4. Analizar emociones por segmento
        enriched_segments = await analyze_segments_with_emotions(
            segments_raw,
            audio_data,
            sr,
            lite_mode,
            audio_weight,
            speaker_labels={},
            enable_diarization=enable_diarization
        )
        
        # Estadísticas globales
        global_emotions = calculate_global_stats(enriched_segments)
        speaker_stats = calculate_speaker_stats(enriched_segments)
        timeline = create_emotion_timeline(enriched_segments)
        
        # Transcripción con etiquetas de hablante
        labeled_transcription = ""
        if enable_diarization and not lite_mode:
            merged_blocks = merge_consecutive_same_speaker(enriched_segments)
            labeled_transcription = format_labeled_transcription(merged_blocks)
            
        num_speakers_detected = len(speaker_stats)
        
        # Quality Score
        scoring_result = calculate_quality_score(enriched_segments, global_emotions, speaker_stats)
        quality_score_data = {
            "total_score": scoring_result.total_score,
            "classification": scoring_result.classification,
            "breakdown": scoring_result.breakdown,
            "recommendations": scoring_result.recommendations
        }
        
        # Resumen de Llamada
        call_summary_data = generate_call_summary(
            segments=enriched_segments,
            global_emotions=global_emotions,
            speaker_stats=speaker_stats,
            duration=validation_result.duration_sec,
            filename=file.filename
        ).to_dict()
        
        # Alertas de Escalamiento
        emotion_dist = global_emotions.get("emotion_distribution", {})
        sentiment_score = round(
            emotion_dist.get("feliz", 0) - (emotion_dist.get("enojado", 0) + emotion_dist.get("triste", 0)), 4
        )
        alert = check_alerts(
            segments=enriched_segments,
            global_emotions=global_emotions,
            filename=file.filename,
            quality_score=scoring_result.total_score,
            sentiment_score=sentiment_score
        )
        alert_data = alert.to_dict() if alert else None
        
        # Resultado final
        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)
        
        # Limpiar memoria
        cleanup_memory()
        
        return {
            "status": "success",
            "mode": "whisperx_v1.0",
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
            "quality_score": quality_score_data,
            "call_summary": call_summary_data,
            "alert": alert_data,
            "sentiment_score": sentiment_score,
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
        
        if enable_diarization:
            logger.warning("Diarización no disponible en modo Cloud (requiere WhisperX local).")
            # speaker_labels, num_speakers_detected = await perform_diarization(...)
            pass
        
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

async def _process_single_audio(
    file: UploadFile,
    lite_mode: bool,
    audio_weight: float,
    enable_diarization: bool,
    num_speakers: Optional[int]
) -> Dict[str, Any]:
    tmp_path = None
    try:
        audio_weight, warnings, lite_mode = await validate_request_params(lite_mode, audio_weight)
        tmp_path = await save_upload_to_tempfile(file)

        validator = AudioValidator()
        validation_result = validator.validate_audio(tmp_path)

        if not validation_result.is_valid:
            return {
                "filename": file.filename,
                "status": "error",
                "error": ", ".join(validation_result.errors)
            }

        # Transcribir con WhisperX
        service = get_whisperx_service()
        wx_result = await run_blocking(service.process_audio, tmp_path)
        
        segments_raw = wx_result["segments"]
        tx_full = " ".join([s["text"].strip() for s in segments_raw])

        if not segments_raw:
             return {
                "filename": file.filename,
                "status": "error",
                "error": "No se detectó voz en el audio"
            }

        # Diarización local
        if enable_diarization:
            try:
                diarizer = get_local_diarizer()
                segments_raw = await run_blocking(
                    diarizer.diarize_from_segments, tmp_path, segments_raw, num_speakers
                )
            except Exception as diar_err:
                logger.warning(f"Diarización falló en batch para {file.filename}: {diar_err}")
                for seg in segments_raw:
                    if "speaker" not in seg:
                        seg["speaker"] = "SPEAKER_00"
        else:
            for seg in segments_raw:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"

        audio_data = np.array([])
        sr = 16000

        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y

        enriched_segments = await analyze_segments_with_emotions(
            segments_raw, audio_data, sr, lite_mode,
            audio_weight, speaker_labels={}, enable_diarization=enable_diarization
        )

        global_emotions = calculate_global_stats(enriched_segments)
        speaker_stats = calculate_speaker_stats(enriched_segments)
        timeline = create_emotion_timeline(enriched_segments)
        num_speakers_detected = len(speaker_stats)

        labeled_transcription = ""
        if enable_diarization and not lite_mode:
            merged_blocks = merge_consecutive_same_speaker(enriched_segments)
            labeled_transcription = format_labeled_transcription(merged_blocks)

        emotion_dist = global_emotions.get("emotion_distribution", {})
        feliz = emotion_dist.get("feliz", 0.0)
        enojado = emotion_dist.get("enojado", 0.0)
        triste = emotion_dist.get("triste", 0.0)
        neutral = emotion_dist.get("neutral", 0.0)
        sentiment_score = round(feliz - (enojado + triste), 4)

        # Quality Score
        scoring_result = calculate_quality_score(enriched_segments, global_emotions, speaker_stats)
        quality_score_data = {
            "total_score": scoring_result.total_score,
            "classification": scoring_result.classification,
            "breakdown": scoring_result.breakdown,
            "recommendations": scoring_result.recommendations
        }

        # esumen de Llamada
        call_summary_data = generate_call_summary(
            segments=enriched_segments,
            global_emotions=global_emotions,
            speaker_stats=speaker_stats,
            duration=validation_result.duration_sec,
            filename=file.filename
        ).to_dict()

        #  Alertas
        alert = check_alerts(
            segments=enriched_segments,
            global_emotions=global_emotions,
            filename=file.filename,
            quality_score=scoring_result.total_score,
            sentiment_score=sentiment_score
        )
        alert_data = alert.to_dict() if alert else None

        cleanup_memory()

        return {
            "filename": file.filename,
            "status": "success",
            "duration_sec": round(validation_result.duration_sec, 2),
            "transcription": tx_full,
            "transcription_preview": tx_full[:200] + "..." if len(tx_full) > 200 else tx_full,
            "labeled_transcription": labeled_transcription,
            "segments": enriched_segments,
            "total_segments": len(enriched_segments),
            "global_emotions": global_emotions,
            "emotion_scores": {
                "feliz": round(feliz, 4),
                "enojado": round(enojado, 4),
                "triste": round(triste, 4),
                "neutral": round(neutral, 4)
            },
            "dominant_emotion": global_emotions.get("top_emotion", "neutral"),
            "dominant_score": round(global_emotions.get("top_score", 0.0), 4),
            "sentiment_score": sentiment_score,
            "emotion_timeline": timeline,
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": num_speakers_detected,
                "speaker_stats": speaker_stats
            },
            "quality_score": quality_score_data,
            "call_summary": call_summary_data,
            "alert": alert_data
        }

    except Exception as e:
        logger.error(f"Error procesando {file.filename}: {e}\n{traceback.format_exc()}")
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e)
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@app.post("/transcribe/batch-scoring", tags=["Batch"])
async def transcribe_batch_scoring(
    files: List[UploadFile] = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    start_time = time.time()

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 archivos por batch")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Debe enviar al menos 1 archivo")

    for f in files:
        validate_upload(f)

    semaphore = asyncio.Semaphore(1)

    async def _process_with_limit(file):
        async with semaphore:
            return await _process_single_audio(
                file, lite_mode, audio_weight, enable_diarization, num_speakers
            )

    tasks = [_process_with_limit(f) for f in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "filename": files[i].filename,
                "status": "error",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    successful = [r for r in processed_results if r.get("status") == "success"]
    failed = [r for r in processed_results if r.get("status") == "error"]

    ranking_positive = sorted(
        successful, key=lambda x: x.get("sentiment_score", 0), reverse=True
    )
    ranking_negative = sorted(
        successful, key=lambda x: x.get("sentiment_score", 0)
    )

    avg_sentiment = 0.0
    emotion_totals = {"feliz": 0.0, "enojado": 0.0, "triste": 0.0, "neutral": 0.0}
    dominant_counts = {}

    if successful:
        avg_sentiment = round(
            sum(r.get("sentiment_score", 0) for r in successful) / len(successful), 4
        )
        for r in successful:
            scores = r.get("emotion_scores", {})
            for emo, val in scores.items():
                emotion_totals[emo] = emotion_totals.get(emo, 0.0) + val
            dom = r.get("dominant_emotion", "neutral")
            dominant_counts[dom] = dominant_counts.get(dom, 0) + 1

        total = sum(emotion_totals.values())
        if total > 0:
            emotion_totals = {k: round(v / total, 4) for k, v in emotion_totals.items()}

    alerts = [r["filename"] for r in successful if r.get("sentiment_score", 0) < -0.3]

    processing_time = time.time() - start_time

    cleanup_memory()

    return {
        "status": "completed",
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "total_processing_time": round(processing_time, 2),
        "results": processed_results,
        "summary": {
            "average_sentiment": avg_sentiment,
            "emotion_averages": emotion_totals,
            "dominant_emotion_distribution": dominant_counts,
            "negative_alerts": alerts,
            "negative_alert_count": len(alerts)
        },
        "ranking": {
            "most_positive": [
                {"filename": r["filename"], "sentiment_score": r["sentiment_score"]}
                for r in ranking_positive[:5]
            ],
            "most_negative": [
                {"filename": r["filename"], "sentiment_score": r["sentiment_score"]}
                for r in ranking_negative[:5]
            ]
        }
    }

@app.post("/transcribe/batch-scoring/stream", tags=["Batch"])
async def transcribe_batch_scoring_stream(
    files: List[UploadFile] = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    """
    Igual que /batch-scoring pero usa Server-Sent Events (SSE).
    Envía un evento JSON por cada archivo procesado, sin esperar al final.
    Evita timeouts del navegador en batches grandes (>15 archivos).
    Soporta hasta 100 archivos.
    """
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 archivos por batch")
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Debe enviar al menos 1 archivo")
    for f in files:
        validate_upload(f)

    # Leer todos los bytes ANTES de iniciar el stream
    # (los UploadFile se vuelven inválidos una vez que el generador async empieza)
    file_data: List[Tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        file_data.append((f.filename, content))

    async def event_generator():
        results = []
        start_time = time.time()
        total = len(file_data)

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        for i, (filename, content) in enumerate(file_data):
            # Heartbeat antes de cada archivo para mantener la conexión viva
            yield f"data: {json.dumps({'type': 'heartbeat', 'index': i, 'total': total, 'filename': filename})}\n\n"

            mock_file = _MockUploadFile(filename, content)
            try:
                result = await _process_single_audio(
                    mock_file, lite_mode, audio_weight, enable_diarization, num_speakers
                )
            except Exception as e:
                result = {"filename": filename, "status": "error", "error": str(e)}
            results.append(result)

            yield f"data: {json.dumps({'type': 'file_complete', 'index': i + 1, 'total': total, 'result': result})}\n\n"

        # Calcular resumen final (misma lógica que el endpoint síncrono)
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        avg_sentiment = 0.0
        emotion_totals: Dict[str, float] = {"feliz": 0.0, "enojado": 0.0, "triste": 0.0, "neutral": 0.0}
        dominant_counts: Dict[str, int] = {}

        if successful:
            avg_sentiment = round(
                sum(r.get("sentiment_score", 0) for r in successful) / len(successful), 4
            )
            for r in successful:
                for emo, val in r.get("emotion_scores", {}).items():
                    emotion_totals[emo] = emotion_totals.get(emo, 0.0) + val
                dom = r.get("dominant_emotion", "neutral")
                dominant_counts[dom] = dominant_counts.get(dom, 0) + 1
            total_emo = sum(emotion_totals.values())
            if total_emo > 0:
                emotion_totals = {k: round(v / total_emo, 4) for k, v in emotion_totals.items()}

        alerts = [r["filename"] for r in successful if r.get("sentiment_score", 0) < -0.3]
        ranking_positive = sorted(successful, key=lambda x: x.get("sentiment_score", 0), reverse=True)
        ranking_negative = sorted(successful, key=lambda x: x.get("sentiment_score", 0))

        summary_event = {
            "type": "complete",
            "status": "completed",
            "total_files": total,
            "successful": len(successful),
            "failed": len(failed),
            "total_processing_time": round(time.time() - start_time, 2),
            "results": results,
            "summary": {
                "average_sentiment": avg_sentiment,
                "emotion_averages": emotion_totals,
                "dominant_emotion_distribution": dominant_counts,
                "negative_alerts": alerts,
                "negative_alert_count": len(alerts)
            },
            "ranking": {
                "most_positive": [
                    {"filename": r["filename"], "sentiment_score": r.get("sentiment_score", 0)}
                    for r in ranking_positive[:5]
                ],
                "most_negative": [
                    {"filename": r["filename"], "sentiment_score": r.get("sentiment_score", 0)}
                    for r in ranking_negative[:5]
                ]
            }
        }
        yield f"data: {json.dumps(summary_event)}\n\n"
        cleanup_memory()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
