# routes/history_routes.py
"""
Módulo para gestión del historial de análisis.
Permite consultar y gestionar sesiones de análisis previas.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])

# Almacenamiento en memoria para el historial (en producción usar DB)
_analysis_history: List[Dict[str, Any]] = []


@router.get("/")
async def get_history(limit: int = 10):
    """
    Obtiene el historial de análisis recientes.
    """
    try:
        return {
            "status": "success",
            "count": len(_analysis_history),
            "history": _analysis_history[-limit:][::-1]  # Más recientes primero
        }
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_to_history(data: Dict[str, Any]):
    """
    Guarda un análisis en el historial.
    """
    try:
        entry = {
            "id": len(_analysis_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "filename": data.get("filename", "unknown"),
            "dominant_emotion": data.get("global_emotions", {}).get("top_emotion", "neutral"),
            "num_segments": len(data.get("segments", [])),
            "duration": data.get("metadata", {}).get("total_duration", 0),
            "summary": data.get("transcription", "")[:200] + "..." if len(data.get("transcription", "")) > 200 else data.get("transcription", "")
        }
        
        _analysis_history.append(entry)
        
        # Mantener solo los últimos 50 análisis
        if len(_analysis_history) > 50:
            _analysis_history.pop(0)
        
        return {"status": "success", "saved_id": entry["id"]}
    except Exception as e:
        logger.error(f"Error guardando en historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_history():
    """
    Limpia todo el historial de análisis.
    """
    global _analysis_history
    _analysis_history = []
    return {"status": "success", "message": "Historial limpiado"}