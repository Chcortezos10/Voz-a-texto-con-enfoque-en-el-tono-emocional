# app_fastapi.py
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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from routes.history_routes import router as history_router
from routes.export_routes import router as export_router
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
    compute_embeddings,
    detect_speaker_changes,
    create_segments_from_labels,
    merge_consecutive_same_speaker,
    format_labeled_transcription
)
from resemblyzer import VoiceEncoder

from config import (
    MAX_UPLOAD_SIZE,
    ALLOWED_MIME,
    WORKERS,
    CORS_ORIGINS,
    MAX_AUDIO_DURATION_SEC,
    WINDOW_SEC,
    HOP_SEC,
    CHANGE_SIM_THRESHOLD,
    MIN_SEG_SEC
)
from Resilience import (
    WHISPER_BREAKER,
    EMOTION_TEXT_BREAKER,
    retry_with_backoff_async
)
from Validators import AudioValidator, ParametersValidator
import gc
import psutil

# Imports para cloud whisper (usados en endpoint /transcribe/cloud-whisper)
try:
    from openai import OpenAI
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI y/o httpx no disponibles. El endpoint /transcribe/cloud-whisper no funcionará.")



# === FUNCIONES DE GESTIÓN DE MEMORIA ===

def log_memory_usage():
    """Log del uso actual de memoria"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logger.info(f"💾 Uso de RAM: {mem_mb:.1f} MB")
    return mem_mb


def cleanup_memory():
    """Limpia memoria después de procesar"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Threadpool para no bloquear el loop principal
executor = ThreadPoolExecutor(max_workers=WORKERS)

# Cache de modelos
_models_cache: Optional[Dict[str, Any]] = None

#metricas simples
_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "total_processing_time": 0.0
}

def increment_metric(key: str, value: float = 1.0):
    if key in _metrics:
        _metrics[key] += value
increment_metric("requests_total")

def get_metrics() -> Dict[str, Any]:
    return {
        "requests_total": _metrics["requests_total"],
        "requests_success": _metrics["requests_success"],
        "requests_failed": _metrics["requests_failed"],
        "success_rate": round(_metrics["requests_success"] / max(1, _metrics["requests_total"]) * 100, 2),
        "avg_processing_time": round(_metrics["total_processing_time"] / max(1, _metrics["requests_success"]), 2)
    }


app = FastAPI(
    title="Voz-a-Texto Emocional Unified API",
    description="API Unificada (v4) con soporte dinamico para pesos y modo Lite.",
    version="4.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(history_router)
app.include_router(export_router)

# Helpers

def load_models() -> Dict[str, Any]:
    global _models_cache
    if _models_cache is None:
        _models_cache = load_whisper_models()
    return _models_cache

_voice_encoder: Optional[VoiceEncoder] = None

def get_voice_encoder() -> VoiceEncoder:
    """Carga VoiceEncoder bajo demanda para diarización."""
    global _voice_encoder
    if _voice_encoder is None:
        logger.info("Cargando VoiceEncoder para diarización...")
        _voice_encoder = VoiceEncoder()
        logger.info("✔ VoiceEncoder cargado")
    return _voice_encoder

async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

@app.on_event("startup")
async def startup_event():
    logger.info("===================================================")
    logger.info(" INICIANDO SISTEMA (Modo Optimizado RAM)")
    logger.info("===================================================")
    
    # Preload SOLO Whisper (critico)
    load_models() 
    logger.info("✔ Whisper cargado")
    
    # Los demas modelos se cargaran bajo demanda para evitar OOM
    logger.info("ℹ Modelos de emoción: Carga bajo demanda (ahorro RAM)")
    
    logger.info("===================================================")
    logger.info(" SISTEMA LISTO: http://127.0.0.1:8000")
    logger.info(" Uso de RAM optimizado")
    logger.info("===================================================")


async def save_upload_to_tempfile(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo: {e}")

def validate_upload(file: UploadFile):
    """Valida que el archivo tenga una extensión permitida."""
    ext = file.filename.lower().split('.')[-1] if file.filename else ""
    if ext not in ["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de archivo no soportado: .{ext}. Use: wav, mp3, m4a, mp4, ogg, flac, webm"
        )


async def validate_request_params(lite_mode: bool, audio_weight: float):
    """Valida parámetros con ParametersValidator"""
    audio_weight, warnings, lite_mode = ParametersValidator.validate_request_params(
        lite_mode, audio_weight
    )
    if warnings:
        logger.warning(f"Parámetros ajustados: {warnings}")
    return audio_weight, warnings, lite_mode

def validate_audio_basic(file_path: str) -> tuple:
    import soundfile as sf
    try:
        info = sf.info(file_path)
        if info.duration < 0.5:
            return False, f"Audio muy corto: {info.duration:.2f}s (min: 0.5s)", 0
        if info.duration > 600:
            return False, f"Audio muy largo: {info.duration:.1f}s (max: 600s)", 0
        return True, None, info.duration
    except Exception as e:
        return False, f"Error leyendo audio: {e}", 0

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "4.0.0", "mode": "UNIFIED"}

@app.get("/health/detailed")
def health_check_detailed():
    return {
        "status": "ok",
        "version": "4.0.0",
        "metrics": get_metrics()
    }

@retry_with_backoff_async(max_retries=2, base_delay=1.0)
async def safe_transcribe(model, path):
    # WHISPER_BREAKER.call espera func, *args, fallback, **kwargs
    # Como run_blocking ejecuta func(*args), aqui pasamos WHISPER_BREAKER.call a run_blocking?
    # No, run_blocking ejecuta en threadpool. 
    # WHISPER_BREAKER.call es sincrono (a menos que usemos call_async pero ese no estaba exposed como sync wrapper).
    # Pero transcribe es bloqueante.
    # Asi que mejor ejecutamos WHISPER_BREAKER.call dentro del threadpool, y que el call llame a model.transcribe.
    
    def _wrapped_transcribe():
        return WHISPER_BREAKER.call(
            model.transcribe,
            path,
            language="es",
            fallback=lambda *a, **k: {"text": "", "segments": []}
        )
    
    return await run_blocking(_wrapped_transcribe)

@app.post("/transcribe/full-analysis", tags=["Analysis"])
async def transcribe_full_analysis(
    file: UploadFile = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    validate_upload(file)
    start_time = time.time()
    increment_metric("requests_total")
    
    # Validar parametros
    audio_weight, param_warnings, lite_mode = await validate_request_params(lite_mode, audio_weight)
    
    warnings = param_warnings
    tmp_path = None
    
    if num_speakers:
        logger.info(f"Solicitado diarización manual con {num_speakers} hablantes")
    try:
        tmp_path = await save_upload_to_tempfile(file)
        
        # Nueva validacion con AudioValidator
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
            logger.warning(f"Advertencias de audio: {validation_result.warnings}")
        
        models = load_models()
        whisper_model = models["whisper"]
        
        logger.info(f"Iniciando Transcripcion (Lite={lite_mode}, AudioW={audio_weight}, Diarization={enable_diarization})...")
        result = await safe_transcribe(whisper_model, tmp_path)
        tx_full = result.get("text", "")
        segments_raw = result.get("segments", [])
        
        segment_texts = [s.get("text", "").strip() for s in segments_raw]
        translations = []
        if not lite_mode and segment_texts:
            logger.info("Traduciendo segmentos para contexto (En)...")
            translations = await run_blocking(translate_batch, segment_texts)

        audio_data = np.array([])
        sr = 16000
        if not lite_mode: 
             y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
             audio_data = y
        
        # Diarización: detectar hablantes
        speaker_labels = {}
        # DEBUG: Mostrar valores recibidos
        logger.info(f"[DBG API] enable_diarization={enable_diarization}, num_speakers={num_speakers} (type={type(num_speakers).__name__})")
        
        if enable_diarization and len(audio_data) > 0:
            try:
                logger.info("Detectando hablantes (diarización)...")
                encoder = get_voice_encoder()
                
                # Crear ventanas para embeddings
                duration = len(audio_data) / sr
                windows = []
                t = 0.0
                while t < duration:
                    end_t = min(t + WINDOW_SEC, duration)
                    windows.append((t, end_t))
                    t += HOP_SEC
                
                if len(windows) >= 2:
                    embeddings, starts = await run_blocking(
                        compute_embeddings, audio_data, sr, windows, encoder, WINDOW_SEC
                    )
                    if len(embeddings) < 2:
                         logger.info("Pocos segmentos de voz detectados tras filtro de silencio.")
                         num_speakers = 1
                    else:
                        _, labels = detect_speaker_changes(
                            embeddings, 
                            CHANGE_SIM_THRESHOLD,
                            num_speakers=num_speakers if enable_diarization else None
                        )
                        
                        # Mapear tiempos a etiquetas de hablante
                        # Usamos starts devueltos porque algunas ventanas pudieron ser filtradas por silencio
                        for i, start_t in enumerate(starts):
                            end_t = start_t + WINDOW_SEC
                            speaker_labels[(start_t, end_t)] = int(labels[i])
                        
                        num_speakers = len(set(labels))
                        logger.info(f"✔ Detectados {num_speakers} hablante(s) (Silencio filtrado)")
                    

                else:
                    logger.info("Audio muy corto para diarización, asumiendo 1 hablante")
            except Exception as e_diar:
                logger.warning(f"Error en diarización: {e_diar}, continuando sin diarización")
                enable_diarization = False
        
        enriched_segments = []
        emotion_analyzer = TemporalEmotionAnalyzer()
        logger.info("Analizando Emociones Multimodal con suavizado temporal...")

        for i, seg in enumerate(segments_raw):
            try:
                # informacion base
                seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
                seg_text_en = translations[i] if i < len(translations) else ""
                
                if not seg_text_es: continue

                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                
                # Analisis de audio
                res_audio = None
                valid_audio = False
                seg_audio_path = None
                seg_chunk = None
                
                if not lite_mode:
                    try:
                        start_idx = int(start * sr)
                        end_idx = int(end * sr)
                        
                        start_idx = max(0, min(start_idx, len(audio_data)))
                        end_idx = max(start_idx, min(end_idx, len(audio_data)))
                        
                        if (end_idx - start_idx) > (sr * 0.1):
                            seg_chunk = audio_data[start_idx:end_idx]
                            seg_audio_path = await run_blocking(write_wav_from_array, seg_chunk, sr)
                            
                            if seg_audio_path:
                                valid_audio = True
                    except Exception as e_audio:
                        logger.warning(f"Audio analysis failed (seg {i}): {e_audio}")
                        valid_audio = False

                emotion_result = emotion_analyzer.analyze_segment(
                    text_es=seg_text_es,
                    text_en=seg_text_en,
                    audio_path=seg_audio_path if valid_audio else None,
                    audio_array=seg_chunk if seg_chunk is not None else None,
                    sr=sr,
                    audio_weight=audio_weight,
                    apply_smoothing=True,
                    use_ensemble=True
                )
                
                # Borrar archivo de audio temporal
                if seg_audio_path and os.path.exists(seg_audio_path):
                    try: os.remove(seg_audio_path)
                    except: pass
                
                top_emo = emotion_result.top_emotion
                top_score = emotion_result.top_score
                final_emotions = emotion_result.emotions

                # Determinar speaker del segmento
                speaker = 0
                if enable_diarization and speaker_labels:
                    # Buscar la ventana más cercana al tiempo medio del segmento
                    seg_mid = (start + end) / 2
                    for (win_s, win_e), spk in speaker_labels.items():
                        if win_s <= seg_mid < win_e:
                            speaker = spk
                            break

                # Formateo
                formatted_emotions = {
                    "spanish_analysis": {"top_emotion": top_emo, "emotions": final_emotions},
                    "english_analysis": {"top_emotion": None},
                    "fused": {
                        "top_emotion": top_emo,
                        "score": round(top_score, 4),
                        "all_emotions": final_emotions 
                    },
                    "audio_analysis": {"top_emotion": None}
                }

                enriched_segments.append({
                    "start": start, "end": end, "duration": end - start,
                    "text_es": seg_text_es, "text_en": seg_text_en,
                    "emotions": formatted_emotions,
                    "text": seg_text_es, 
                    "emotion": top_emo, 
                    "intensity": round(top_score, 2),
                    "time_start": round(start, 2), "time_end": round(end, 2),
                    "speaker": f"speaker_{speaker}"
                })

            except Exception as e_seg:
                logger.error(f"Critical error seg {i}: {e_seg}")
                continue

        # Estadisticas
        global_emotions = {}
        for seg in enriched_segments:
            fused = seg["emotions"]["fused"]["all_emotions"]
            for k, v in fused.items():
                global_emotions[k] = global_emotions.get(k, 0.0) + v
        
        total_w = sum(global_emotions.values())
        if total_w > 0:
            for k in global_emotions: global_emotions[k] = round(global_emotions[k] / total_w, 4)

        top_global = max(global_emotions, key=global_emotions.get) if global_emotions else "neutral"

        # Timeline con speaker
        timeline = []
        for s in enriched_segments:
            timeline.append({
                "time": s["start"],
                "emotion": s["emotions"]["fused"]["top_emotion"],
                "score": s["emotions"]["fused"]["score"],
                "all_emotions": s["emotions"]["fused"]["all_emotions"],
                "speaker": s.get("speaker", "speaker_0")
            })

        # Generar transcripción con etiquetas de hablante
        labeled_transcription = ""
        if enable_diarization:
            merged_blocks = merge_consecutive_same_speaker(enriched_segments)
            labeled_transcription = format_labeled_transcription(merged_blocks)

        # Calcular estadísticas por hablante
        speaker_stats = {}
        if enable_diarization:
            from collections import Counter
            
            for s in enriched_segments:
                # speaker viene como "speaker_X" o int X
                spk_raw = s.get("speaker", 0)
                if isinstance(spk_raw, str) and spk_raw.startswith("speaker_"):
                     spk_id = int(spk_raw.split("_")[1])
                else:
                     spk_id = int(spk_raw) if spk_raw is not None else 0
                
                if spk_id not in speaker_stats:
                    speaker_stats[spk_id] = {
                        "label": f"Hablante {spk_id + 1}",
                        "total_duration": 0.0,
                        "segment_count": 0,
                        "emotions": []
                    }
                
                dur = s.get("duration", 0)
                speaker_stats[spk_id]["total_duration"] += dur
                speaker_stats[spk_id]["segment_count"] += 1
                
                emo = s.get("emotion")
                if emo:
                    speaker_stats[spk_id]["emotions"].append(emo)

            # Calcular emoción dominante por hablante
            for spk_id in speaker_stats:
                emos = speaker_stats[spk_id].pop("emotions")
                if emos:
                    c = Counter(emos)
                    dom_emo = c.most_common(1)[0][0]
                    speaker_stats[spk_id]["dominant_emotion"] = dom_emo
                else:
                    speaker_stats[spk_id]["dominant_emotion"] = "neutral"

        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)

        return {
            "status": "success",
            "mode": "consolidated_v4",
            "transcription": tx_full,
            "labeled_transcription": labeled_transcription if enable_diarization else "",
            "translation": " ".join(translations) if translations else "",
            "segments": enriched_segments,
            "global_emotions": {
                "top_emotion": top_global,
                "top_score": global_emotions.get(top_global, 0.0),
                "emotion_distribution": global_emotions
            },
            "emotion_timeline": timeline,
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": num_speakers if enable_diarization else 1,
                "speaker_stats": speaker_stats
            },
            "metadata": {
                "total_duration": result.get("duration", 0),
                "processing_time": round(processing_time, 2),
                "params": {"lite": lite_mode, "audio_weight": audio_weight, "diarization": enable_diarization}
            }
        }

    except Exception as e:
        increment_metric("requests_failed")
        logger.error(f"Error Gen: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
             try: os.remove(tmp_path)
             except: pass

@app.post("/transcribe/full-analysis-diarized", tags=["Analysis"])
async def transcribe_with_diarization(
    file: UploadFile = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    num_speakers: Optional[int] = Form(None),
    enable_diarization: bool = Form(True)
):
    """Análisis completo con diarización de hablantes."""
    validate_upload(file)
    start_time = time.time()
    increment_metric("requests_total")
    tmp_path = None
    
    try:
        tmp_path = await save_upload_to_tempfile(file)
        
        is_valid, error_msg, audio_duration = validate_audio_basic(tmp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        models = load_models()
        whisper_model = models["whisper"]
        
        logger.info(f"Transcribiendo (diarización={enable_diarization}, speakers={num_speakers or 'auto'})...")
        result = await run_blocking(whisper_model.transcribe, tmp_path, language="es")
        tx_full = result.get("text", "")
        segments_raw = result.get("segments", [])
        
        segment_texts = [s.get("text", "").strip() for s in segments_raw]
        translations = []
        if not lite_mode and segment_texts:
            translations = await run_blocking(translate_batch, segment_texts)
        
        audio_data = np.array([])
        sr = 16000
        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y
        
        # Diarización
        speaker_map = {}
        diarization_result = None
        
        if enable_diarization and len(audio_data) > 0:
            logger.info("Ejecutando diarización...")
            try:
                diarization_result = await run_blocking(
                    diarize_audio, audio_data, sr, segments_raw, num_speakers
                )
                
                for i, seg in enumerate(segments_raw):
                    seg_start = seg.get("start", 0)
                    for diar_seg in diarization_result.segments:
                        if abs(diar_seg.start - seg_start) < 0.5:
                            speaker_map[i] = {
                                "speaker_id": diar_seg.speaker_id,
                                "speaker_label": diar_seg.speaker_label,
                                "confidence": diar_seg.confidence
                            }
                            break
                    if i not in speaker_map:
                        speaker_map[i] = {"speaker_id": 0, "speaker_label": "Hablante 1", "confidence": 0.5}
                
                logger.info(f"Diarización: {diarization_result.num_speakers} hablantes")
            except Exception as e:
                logger.warning(f"Error diarización: {e}")
        
        # Análisis emocional
        enriched_segments = []
        emotion_analyzer = TemporalEmotionAnalyzer(use_prosody=not lite_mode)
        
        for i, seg in enumerate(segments_raw):
            try:
                seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
                seg_text_en = translations[i] if i < len(translations) else ""
                
                if not seg_text_es:
                    continue
                
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                
                seg_audio = None
                seg_audio_path = None
                
                if not lite_mode and len(audio_data) > 0:
                    start_idx = max(0, int(start * sr))
                    end_idx = min(len(audio_data), int(end * sr))
                    
                    if (end_idx - start_idx) > (sr * 0.1):
                        seg_audio = audio_data[start_idx:end_idx]
                        seg_audio_path = await run_blocking(write_wav_from_array, seg_audio, sr)
                
                emotion_result = emotion_analyzer.analyze_segment(
                    text_es=seg_text_es,
                    text_en=seg_text_en,
                    audio_path=seg_audio_path,
                    audio_array=seg_audio,
                    sr=sr,
                    audio_weight=audio_weight
                )
                
                if seg_audio_path and os.path.exists(seg_audio_path):
                    try:
                        os.remove(seg_audio_path)
                    except:
                        pass
                
                speaker_info = speaker_map.get(i, {"speaker_id": 0, "speaker_label": "Hablante 1", "confidence": 0.5})
                
                enriched_segments.append({
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text_es": seg_text_es,
                    "text_en": seg_text_en,
                    "text": seg_text_es,
                    "speaker_id": speaker_info["speaker_id"],
                    "speaker_label": speaker_info["speaker_label"],
                    "emotion": emotion_result.top_emotion,
                    "intensity": round(emotion_result.top_score, 2),
                    "emotions": {
                        "fused": {
                            "top_emotion": emotion_result.top_emotion,
                            "score": round(emotion_result.top_score, 4),
                            "all_emotions": emotion_result.emotions
                        }
                    }
                })
            except Exception as e:
                logger.error(f"Error segmento {i}: {e}")
        
        # Estadísticas globales
        global_emotions = {}
        for seg in enriched_segments:
            for k, v in seg["emotions"]["fused"]["all_emotions"].items():
                global_emotions[k] = global_emotions.get(k, 0) + v
        
        total_w = sum(global_emotions.values())
        if total_w > 0:
            global_emotions = {k: round(v/total_w, 4) for k, v in global_emotions.items()}
        
        top_global = max(global_emotions, key=global_emotions.get) if global_emotions else "neutral"
        
        # Stats por hablante
        speaker_stats = {}
        for seg in enriched_segments:
            sid = seg["speaker_id"]
            if sid not in speaker_stats:
                speaker_stats[sid] = {
                    "label": seg["speaker_label"],
                    "total_duration": 0,
                    "segment_count": 0,
                    "emotions": {}
                }
            speaker_stats[sid]["total_duration"] += seg["duration"]
            speaker_stats[sid]["segment_count"] += 1
            for emo, score in seg["emotions"]["fused"]["all_emotions"].items():
                speaker_stats[sid]["emotions"][emo] = speaker_stats[sid]["emotions"].get(emo, 0) + score
        
        for sid, stats in speaker_stats.items():
            total = sum(stats["emotions"].values())
            if total > 0:
                stats["emotions"] = {k: round(v/total, 4) for k, v in stats["emotions"].items()}
            stats["dominant_emotion"] = max(stats["emotions"], key=stats["emotions"].get) if stats["emotions"] else "neutral"
            stats["total_duration"] = round(stats["total_duration"], 2)
        
        timeline = [{
            "time": s["start"],
            "emotion": s["emotions"]["fused"]["top_emotion"],
            "score": s["emotions"]["fused"]["score"],
            "all_emotions": s["emotions"]["fused"]["all_emotions"],
            "speaker_id": s["speaker_id"],
            "speaker_label": s["speaker_label"]
        } for s in enriched_segments]
        
        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)
        
        return {
            "status": "success",
            "mode": "diarized_v4",
            "transcription": tx_full,
            "translation": " ".join(translations) if translations else "",
            "segments": enriched_segments,
            "global_emotions": {
                "top_emotion": top_global,
                "top_score": global_emotions.get(top_global, 0),
                "emotion_distribution": global_emotions
            },
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": diarization_result.num_speakers if diarization_result else 1,
                "speaker_stats": speaker_stats
            },
            "emotion_timeline": timeline,
            "metadata": {
                "total_duration": result.get("duration", 0),
                "processing_time": round(processing_time, 2),
                "params": {"lite": lite_mode, "audio_weight": audio_weight, "num_speakers": num_speakers}
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
@app.post("/transcribe/cloud-whisper", tags=["Cloud"])
async def transcribe_with_cloud_whisper(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[int] = Form(None)
):
    # Verificar que OpenAI esté disponible
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI Whisper Cloud no disponible. Instale: pip install openai httpx"
        )

    validate_upload(file)
    start_time = time.time()
    increment_metric("requests_total")

    
    try:
        # Guardar archivo
        tmp_path = await save_upload_to_tempfile(file)
        
        # Validar
        validator = AudioValidator()
        validation_result = validator.validate_audio(tmp_path)

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Audio inválido",
                    "errors": validation_result.errors
                }
            )
        
        # Transcribir con OpenAI
        logger.info("🌐 Transcribiendo con OpenAI Whisper Cloud...")
        
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
            logger.error(f"Error con OpenAI API: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error en OpenAI API: {str(e)}"
            )
        
        # Convertir respuesta
        segments_raw = []
        for seg in transcript_response.segments:
            segments_raw.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })
        
        tx_full = transcript_response.text
        logger.info(f"✅ Transcripción completada: {len(segments_raw)} segmentos")
        
        # Traducir
        segment_texts = [s["text"].strip() for s in segments_raw]
        translations = []
        if not lite_mode and segment_texts:
            logger.info("🔄 Traduciendo segmentos...")
            translations = await run_blocking(translate_batch, segment_texts)
        
        # Cargar audio para análisis
        audio_data = np.array([])
        sr = 16000
        if not lite_mode:
            y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
            audio_data = y
        
        # Diarización
        speaker_labels = {}
        num_speakers_detected = 1
        if enable_diarization and len(audio_data) > 0:
            try:
                logger.info("🎙️ Ejecutando diarización...")
                encoder = get_voice_encoder()
                
                duration = len(audio_data) / sr
                windows = []
                t = 0.0
                while t < duration:
                    end_t = min(t + WINDOW_SEC, duration)
                    windows.append((t, end_t))
                    t += HOP_SEC
                
                if len(windows) >= 2:
                    embeddings, starts = await run_blocking(
                        compute_embeddings, audio_data, sr, windows, encoder, WINDOW_SEC
                    )
                    
                    if len(embeddings) >= 2:
                        _, labels = detect_speaker_changes(
                            embeddings, 
                            CHANGE_SIM_THRESHOLD,
                            num_speakers=num_speakers
                        )
                        
                        for i, start_t in enumerate(starts):
                            end_t = start_t + WINDOW_SEC
                            speaker_labels[(start_t, end_t)] = int(labels[i])
                        
                        num_speakers_detected = len(set(labels))
                        logger.info(f"✔ Detectados {num_speakers_detected} hablante(s)")
            except Exception as e_diar:
                logger.warning(f"Error en diarización: {e_diar}")
                enable_diarization = False
        
        # Análisis emocional
        enriched_segments = []
        emotion_analyzer = TemporalEmotionAnalyzer()
        logger.info("💭 Analizando emociones...")

        for i, seg in enumerate(segments_raw):
            try:
                seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
                seg_text_en = translations[i] if i < len(translations) else ""
                
                if not seg_text_es:
                    continue

                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                
                seg_audio_path = None
                seg_chunk = None
                
                if not lite_mode and len(audio_data) > 0:
                    try:
                        start_idx = int(start * sr)
                        end_idx = int(end * sr)
                        
                        start_idx = max(0, min(start_idx, len(audio_data)))
                        end_idx = max(start_idx, min(end_idx, len(audio_data)))
                        
                        if (end_idx - start_idx) > (sr * 0.1):
                            seg_chunk = audio_data[start_idx:end_idx]
                            seg_audio_path = await run_blocking(write_wav_from_array, seg_chunk, sr)
                    except Exception as e_audio:
                        logger.warning(f"Audio seg {i} failed: {e_audio}")

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
                
                top_emo = emotion_result.top_emotion
                top_score = emotion_result.top_score
                final_emotions = emotion_result.emotions

                speaker = 0
                if enable_diarization and speaker_labels:
                    seg_mid = (start + end) / 2
                    for (win_s, win_e), spk in speaker_labels.items():
                        if win_s <= seg_mid < win_e:
                            speaker = spk
                            break

                enriched_segments.append({
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text_es": seg_text_es,
                    "text_en": seg_text_en,
                    "text": seg_text_es,
                    "emotion": top_emo,
                    "intensity": round(top_score, 2),
                    "emotions": {
                        "fused": {
                            "top_emotion": top_emo,
                            "score": round(top_score, 4),
                            "all_emotions": final_emotions
                        }
                    },
                    "speaker_id": speaker,
                    "speaker_label": f"Hablante {speaker + 1}",
                    "speaker": f"speaker_{speaker}"
                })

            except Exception as e_seg:
                logger.error(f"Error seg {i}: {e_seg}")
                continue

        # Estadísticas globales
        global_emotions = {}
        for seg in enriched_segments:
            fused = seg["emotions"]["fused"]["all_emotions"]
            for k, v in fused.items():
                global_emotions[k] = global_emotions.get(k, 0.0) + v
        
        total_w = sum(global_emotions.values())
        if total_w > 0:
            for k in global_emotions:
                global_emotions[k] = round(global_emotions[k] / total_w, 4)

        top_global = max(global_emotions, key=global_emotions.get) if global_emotions else "neutral"

        # Timeline
        timeline = []
        for s in enriched_segments:
            timeline.append({
                "time": s["start"],
                "emotion": s["emotion"],
                "score": s["emotions"]["fused"]["score"],
                "all_emotions": s["emotions"]["fused"]["all_emotions"],
                "speaker_id": s.get("speaker_id", 0),
                "speaker_label": s.get("speaker_label", "Hablante 1")
            })

        # Speaker stats
        speaker_stats = {}
        if enable_diarization:
            from collections import Counter
            
            for s in enriched_segments:
                spk_id = s.get("speaker_id", 0)
                
                if spk_id not in speaker_stats:
                    speaker_stats[spk_id] = {
                        "label": s.get("speaker_label", f"Hablante {spk_id + 1}"),
                        "total_duration": 0.0,
                        "segment_count": 0,
                        "emotions": []
                    }
                
                speaker_stats[spk_id]["total_duration"] += s.get("duration", 0)
                speaker_stats[spk_id]["segment_count"] += 1
                
                emo = s.get("emotion")
                if emo:
                    speaker_stats[spk_id]["emotions"].append(emo)

            for spk_id in speaker_stats:
                emos = speaker_stats[spk_id].pop("emotions")
                if emos:
                    c = Counter(emos)
                    speaker_stats[spk_id]["dominant_emotion"] = c.most_common(1)[0][0]
                else:
                    speaker_stats[spk_id]["dominant_emotion"] = "neutral"

        processing_time = time.time() - start_time
        increment_metric("requests_success")
        increment_metric("total_processing_time", processing_time)

        return {
            "status": "success",
            "mode": "cloud_whisper",
            "transcription": tx_full,
            "translation": " ".join(translations) if translations else "",
            "segments": enriched_segments,
            "global_emotions": {
                "top_emotion": top_global,
                "top_score": global_emotions.get(top_global, 0.0),
                "emotion_distribution": global_emotions
            },
            "emotion_timeline": timeline,
            "diarization": {
                "enabled": enable_diarization,
                "num_speakers": num_speakers_detected,
                "speaker_stats": speaker_stats
            },
            "metadata": {
                "total_duration": transcript_response.duration,
                "processing_time": round(processing_time, 2),
                "params": {
                    "lite": lite_mode,
                    "audio_weight": audio_weight,
                    "diarization": enable_diarization,
                    "transcription_engine": "openai_whisper_cloud"
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

