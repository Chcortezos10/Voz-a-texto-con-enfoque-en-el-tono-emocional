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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from core.models import load_whisper_models
from core.audio_processing import load_audio, write_wav_from_array
from core.translation import translate_batch
from core.emotion_analysis import (
    analyze_audio_emotion, 
    analyze_text_emotion_es, 
    analyze_text_emotion_en, 
    EmotionResult
)
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

app = FastAPI(
    title="Voz-a-Texto Emocional Unified API",
    description="API Unificada (v3) con soporte dinámico para pesos y modo Lite.",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ==========================================
# Helpers
# ==========================================

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
    logger.info("  🚀 INICIANDO SISTEMA (Modo Optimizado RAM)")
    logger.info("===================================================")
    
    # Preload SOLO Whisper (crítico)
    load_models() 
    logger.info("✔ Whisper cargado")
    
    # Los demás modelos se cargarán BAJO DEMANDA para evitar OOM
    logger.info("ℹ Modelos de emoción: Carga bajo demanda (ahorro RAM)")
    
    logger.info("===================================================")
    logger.info("  ✅ SISTEMA LISTO: http://127.0.0.1:8000")
    logger.info("  💾 Uso de RAM optimizado")
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
        # Simple extension check fallback
        pass

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0.0", "mode": "UNIFIED"}

@app.post("/transcribe/full-analysis", tags=["Analysis"])
async def transcribe_full_analysis(
    file: UploadFile = File(...),
    lite_mode: bool = Form(False),
    audio_weight: float = Form(0.4)
):
    validate_upload(file)
    tmp_path = None
    try:
        tmp_path = await save_upload_to_tempfile(file)
        models = load_models()
        whisper_model = models["whisper"]
        
        # 1. Transcribe
        logger.info(f"Iniciando Transcripción (Lite={lite_mode}, AudioW={audio_weight})...")
        result = await run_blocking(whisper_model.transcribe, tmp_path, language="es")
        tx_full = result.get("text", "")
        segments_raw = result.get("segments", [])
        
        # 2. Translate
        segment_texts = [s.get("text", "").strip() for s in segments_raw]
        translations = []
        if not lite_mode and segment_texts:
            logger.info("Traduciendo segmentos para contexto (En)...")
            translations = await run_blocking(translate_batch, segment_texts)

        # 3. Audio Load
        audio_data = np.array([])
        sr = 16000
        if not lite_mode: 
             y, sr = await run_blocking(librosa.load, tmp_path, sr=16000)
             audio_data = y
        
        enriched_segments = []
        logger.info("Analizando Emociones Multimodal...")

        for i, seg in enumerate(segments_raw):
            try:
                # Basic info
                seg_text_es = segment_texts[i] if i < len(segment_texts) else ""
                seg_text_en = translations[i] if i < len(translations) else ""
                
                if not seg_text_es: continue

                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                
                # --- Audio Analysis (Robust) ---
                res_audio = None
                valid_audio = False
                
                if not lite_mode:
                    try:
                        start_idx = int(start * sr)
                        end_idx = int(end * sr)
                        
                        # Bounds Check
                        start_idx = max(0, min(start_idx, len(audio_data)))
                        end_idx = max(start_idx, min(end_idx, len(audio_data)))
                        
                        if (end_idx - start_idx) > (sr * 0.1): # Min 0.1s
                            seg_chunk = audio_data[start_idx:end_idx]
                            seg_audio_path = await run_blocking(write_wav_from_array, seg_chunk, sr)
                            
                            if seg_audio_path:
                                res_audio = await run_blocking(analyze_audio_emotion, seg_audio_path)
                                valid_audio = True
                                if os.path.exists(seg_audio_path): os.remove(seg_audio_path)
                    except Exception as e_audio:
                        logger.warning(f"Audio analysis failed (seg {i}): {e_audio}")
                        valid_audio = False

                # --- Text Analysis ---
                res_es = await run_blocking(analyze_text_emotion_es, seg_text_es)
                if not res_es:
                     res_es = EmotionResult(top_emotion="neutral", emotions={"neutral": 1.0})

                res_en = None
                if not lite_mode and seg_text_en:
                    res_en = await run_blocking(analyze_text_emotion_en, seg_text_en)

                # --- Fusion Logic (Dynamic Weights) ---
                # 1. Text Average (ES + EN)
                text_emotions = res_es.emotions.copy()
                if res_en:
                    for k, v in res_en.emotions.items():
                        text_emotions[k] = (text_emotions.get(k, 0) + v) / 2
                        
                # 2. Multimodal Fusion
                final_emotions = {}
                score_audio = res_audio.emotions if (res_audio and valid_audio) else {}
                
                # Dynamic Logic
                eff_audio_w = audio_weight if valid_audio else 0.0
                eff_text_w = 1.0 - eff_audio_w

                all_keys = set(text_emotions.keys()) | set(score_audio.keys())
                for k in all_keys:
                    val_t = text_emotions.get(k, 0.0)
                    val_a = score_audio.get(k, 0.0)
                    final_emotions[k] = (val_t * eff_text_w) + (val_a * eff_audio_w)

                if not final_emotions: final_emotions = {"neutral": 1.0}
                top_emo = max(final_emotions, key=final_emotions.get)
                top_score = final_emotions.get(top_emo, 0.0)

                # Formatting
                formatted_emotions = {
                    "spanish_analysis": {"top_emotion": res_es.top_emotion, "emotions": res_es.emotions},
                    "english_analysis": {"top_emotion": res_en.top_emotion if res_en else None},
                    "fused": {
                        "top_emotion": top_emo,
                        "score": round(top_score, 4),
                        "all_emotions": final_emotions 
                    },
                    "audio_analysis": {"top_emotion": res_audio.top_emotion if (res_audio and valid_audio) else None}
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

        # Global Stats
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
                "all_emotions": s["emotions"]["fused"]["all_emotions"]  # NEW: Full distribution for multi-line chart
            })

        return {
            "status": "success",
            "mode": "consolidated_v3",
            "transcription": tx_full,
            "translation": " ".join(translations) if translations else "",
            "segments": enriched_segments,
            "global_emotions": {
                "top_emotion": top_global,
                "top_score": global_emotions.get(top_global, 0.0), # NEW: Added intensity
                "emotion_distribution": global_emotions
            },
            "emotion_timeline": timeline,
            "metadata": {
                "total_duration": result.get("duration", 0),
                "params": {"lite": lite_mode, "audio_weight": audio_weight}
            }
        }

    except Exception as e:
        logger.error(f"Error Gen: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
             try: os.remove(tmp_path)
             except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
