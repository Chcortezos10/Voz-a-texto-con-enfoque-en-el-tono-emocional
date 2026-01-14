# routes/export_routes.py
"""
Módulo para exportación de datos de análisis.
Permite exportar transcripciones y análisis emocionales en diferentes formatos.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


@router.post("/json")
async def export_to_json(data: Dict[str, Any]):
    """
    Exporta los resultados del análisis en formato JSON.
    """
    try:
        return JSONResponse(
            content=data,
            headers={
                "Content-Disposition": "attachment; filename=analysis_export.json"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def export_summary(data: Dict[str, Any]):
    """
    Genera un resumen del análisis emocional.
    """
    try:
        segments = data.get("segments", [])
        global_emotions = data.get("global_emotions", {})
        
        summary = {
            "total_segments": len(segments),
            "dominant_emotion": global_emotions.get("top_emotion", "neutral"),
            "emotion_distribution": global_emotions.get("emotion_distribution", {}),
            "transcription_preview": data.get("transcription", "")[:500] + "..." if len(data.get("transcription", "")) > 500 else data.get("transcription", "")
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))