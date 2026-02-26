"""
Rutas de Alertas de Escalamiento Automático.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from core.alert_system import (
    check_alerts,
    get_all_alerts,
    mark_alert_reviewed,
    delete_alert,
    get_alert_stats
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/")
async def list_alerts(
    severity: Optional[str] = Query(None, description="Filtrar por severidad: alta, media, baja"),
    reviewed: Optional[bool] = Query(None, description="Filtrar por estado de revisión"),
    limit: int = Query(50, ge=1, le=200)
):
    """Lista alertas de escalamiento."""
    try:
        alerts = get_all_alerts(
            severity_filter=severity,
            reviewed_filter=reviewed,
            limit=limit
        )
        return {
            "status": "success",
            "alerts": alerts,
            "total": len(alerts)
        }
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def alert_stats():
    """Estadísticas de alertas."""
    try:
        stats = get_alert_stats()
        return {"status": "success", **stats}
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_call_alerts(data: Dict[str, Any]):
    """
    Evalúa una llamada y retorna alertas si las hay.
    Espera formato de /transcribe/full-analysis.
    """
    try:
        segments = data.get("segments", [])
        global_emotions = data.get("global_emotions", {})
        filename = data.get("filename", "unknown")
        quality_score = data.get("quality_score", {}).get("total_score") if isinstance(data.get("quality_score"), dict) else data.get("quality_score")
        sentiment_score = data.get("sentiment_score")

        alert = check_alerts(
            segments=segments,
            global_emotions=global_emotions,
            filename=filename,
            quality_score=quality_score,
            sentiment_score=sentiment_score
        )

        if alert:
            return {
                "status": "alert_generated",
                "alert": alert.to_dict()
            }
        else:
            return {
                "status": "ok",
                "message": "No se detectaron problemas que requieran escalamiento"
            }
    except Exception as e:
        logger.error(f"Error verificando alertas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{alert_id}/review")
async def review_alert(alert_id: str, notes: str = ""):
    """Marca una alerta como revisada."""
    success = mark_alert_reviewed(alert_id, notes)
    if success:
        return {"status": "success", "message": "Alerta marcada como revisada"}
    raise HTTPException(status_code=404, detail="Alerta no encontrada")


@router.delete("/{alert_id}")
async def remove_alert(alert_id: str):
    """Elimina una alerta."""
    success = delete_alert(alert_id)
    if success:
        return {"status": "success", "message": "Alerta eliminada"}
    raise HTTPException(status_code=404, detail="Alerta no encontrada")
