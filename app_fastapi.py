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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

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
from core.speaker_diarization import diarize_audio, DiarizationResult
from config import (
    MAX_UPLOAD_SIZE,
    ALLOWED_MIME,
    WORKERS,
    CORS_ORIGINS,
    MAX_AUDIO_DURATION_SEC
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

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

# Helpers

def load_models() -> Dict[str, Any]:
    global _models_cache
    if _models_cache is None:
        _models_cache = load_whisper_models()
    return _models_cache

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
    if file.filename.lower().split('.')[-1] not in ["wav", "mp3", "m4a", "mp4"]:
        pass

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
        "metrics": {
            "requests_total": _metrics["requests_total"],
            "requests_success": _metrics["requests_success"],
            "requests_failed": _metrics["requests_failed"],
            "success_rate": round(_metrics["requests_success"] / max(1, _metrics["requests_total"]) * 100, 2),
            "avg_processing_time": round(_metrics["total_processing_time"] / max(1, _metrics["requests_success"]), 2)
        }
    }

@app.post("/transcribe/full-analysis", tags=["Analysis"])
async def transcribe_full_analysis(
    file: UploadFile = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4)
):
    validate_upload(file)
    start_time = time.time()
    _metrics["requests_total"] += 1
    warnings = []
    tmp_path = None
    try:
        tmp_path = await save_upload_to_tempfile(file)
        
        is_valid, error_msg, audio_duration = validate_audio_basic(tmp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        models = load_models()
        whisper_model = models["whisper"]
        
        logger.info(f"Iniciando Transcripcion (Lite={lite_mode}, AudioW={audio_weight})...")
        result = await run_blocking(whisper_model.transcribe, tmp_path, language="es")
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
                
                if not lite_mode:
                    try:
                        start_idx = int(start * sr)
                        end_idx = int(end * sr)
                        
                        start_idx = max(0, min(start_idx, len(audio_data)))
                        end_idx = max(start_idx, min(end_idx, len(audio_data)))
                        
                        if (end_idx - start_idx) > (sr * 0.1): #el audio teine que ser minimo de 1s
                            seg_chunk = audio_data[start_idx:end_idx]
                            seg_audio_path = await run_blocking(write_wav_from_array, seg_chunk, sr)
                            
                            if seg_audio_path:
                                # Ya no analizamos aqui, se pasa al TemporalEmotionAnalyzer
                                valid_audio = True
                                # NO BORRAR AQUI: if os.path.exists(seg_audio_path): os.remove(seg_audio_path)
                    except Exception as e_audio:
                        logger.warning(f"Audio analysis failed (seg {i}): {e_audio}")
                        valid_audio = False

                #analisis de texto (solo textos para no bloquear, el Analyzer hara lo demas)
                # NOTA: El TemporalEmotionAnalyzer llama a analyze_text_emotion_es internamente? 
                # Revisando la implementacion de TemporalEmotionAnalyzer en emotion_analysis.py:
                # Sí, llama a analyze_text_emotion_es y analyze_text_emotion_en.
                # Entonces podemos eliminar las llamadas manuales a analyze_text_emotion_* aqui tambien para evitar duplicidad,
                # PERO app_fastapi.py actual aun tiene esas llamadas antes de llamar a emotion_analyzer.
                # Para simplificar y arreglar el error de archivo, primero arreglemos lo del archivo.
                # La duplicidad de texto es menos grave (cache lru ayuda).
                # Pero la clase TemporalEmotionAnalyzer SI hace el trabajo completo.
                
                emotion_result = emotion_analyzer.analyze_segment(
                    text_es=seg_text_es,
                    text_en=seg_text_en,
                    audio_path=seg_audio_path if valid_audio else None,
                    audio_weight=audio_weight,
                    apply_smoothing=True
                )
                
                # AHORA borramos el archivo de audio
                if valid_audio and seg_audio_path and os.path.exists(seg_audio_path):
                    try: os.remove(seg_audio_path)
                    except: pass
                
                top_emo = emotion_result.top_emotion
                top_score = emotion_result.top_score
                final_emotions = emotion_result.emotions

                # Formateo (Simplificado para usar solo emotion_result del Analyzer)
                formatted_emotions = {
                    "spanish_analysis": {"top_emotion": top_emo, "emotions": final_emotions}, # Aprox
                    "english_analysis": {"top_emotion": None}, # Ya no desglosamos indiv
                    "fused": {
                        "top_emotion": top_emo,
                        "score": round(top_score, 4),
                        "all_emotions": final_emotions 
                    },
                    "audio_analysis": {"top_emotion": None} # Ya no desglosamos indiv
                }

                enriched_segments.append({
                    "start": start, "end": end, "duration": end - start,
                    "text_es": seg_text_es, "text_en": seg_text_en,
                    "emotions": formatted_emotions,
                    "text": seg_text_es, 
                    "emotion": top_emo, 
                    "intensity": round(top_score, 2),
                    "time_start": round(start, 2), "time_end": round(end, 2)
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

        # Timeline
        timeline = []
        for s in enriched_segments:
            timeline.append({
                "time": s["start"],
                "emotion": s["emotions"]["fused"]["top_emotion"],
                "score": s["emotions"]["fused"]["score"],
                "all_emotions": s["emotions"]["fused"]["all_emotions"]
            })

        processing_time = time.time() - start_time
        _metrics["requests_success"] += 1
        _metrics["total_processing_time"] += processing_time

        return {
            "status": "success",
            "mode": "consolidated_v4",
            "transcription": tx_full,
            "translation": " ".join(translations) if translations else "",
            "segments": enriched_segments,
            "global_emotions": {
                "top_emotion": top_global,
                "top_score": global_emotions.get(top_global, 0.0),
                "emotion_distribution": global_emotions
            },
            "emotion_timeline": timeline,
            "metadata": {
                "total_duration": result.get("duration", 0),
                "processing_time": round(processing_time, 2),
                "params": {"lite": lite_mode, "audio_weight": audio_weight}
            }
        }

    except Exception as e:
        _metrics["requests_failed"] += 1
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
    _metrics["requests_total"] += 1
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
        _metrics["requests_success"] += 1
        _metrics["total_processing_time"] += processing_time
        
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
        _metrics["requests_failed"] += 1
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
