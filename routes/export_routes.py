from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from typing import Dict, Any
import json
import logging

from core.export_manager import ExportManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


@router.post("/json")
async def export_to_json(data: Dict[str, Any]):
    try:
        manager = ExportManager()
        content = manager.export_json(data, pretty=True)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=analysis_export.json"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/csv")
async def export_to_csv(data: Dict[str, Any]):
    try:
        manager = ExportManager()
        content = manager.export_csv(data)
        return Response(
            content=content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=analysis_export.csv"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/srt")
async def export_to_srt(data: Dict[str, Any]):
    try:
        manager = ExportManager()
        content = manager.export_srt(data)
        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": "attachment; filename=subtitulos.srt"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a SRT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/txt")
async def export_to_txt(data: Dict[str, Any]):
    try:
        manager = ExportManager()
        content = manager.export_txt(data)
        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": "attachment; filename=transcripcion.txt"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a TXT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vtt")
async def export_to_vtt(data: Dict[str, Any]):
    try:
        manager = ExportManager()
        content = manager.export_vtt(data)
        return Response(
            content=content,
            media_type="text/vtt",
            headers={
                "Content-Disposition": "attachment; filename=subtitulos.vtt"
            }
        )
    except Exception as e:
        logger.error(f"Error exportando a VTT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def export_summary(data: Dict[str, Any]):
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