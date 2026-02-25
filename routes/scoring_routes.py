"""
Rutas de Scoring de Calidad del Agente.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
import json
from pathlib import Path
from dataclasses import asdict

from core.scoring_engine import calculate_quality_score, calculate_general_quality_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scoring", tags=["Scoring"])

history_file = Path("data/analysis_history.json")


def _load_history() -> List[Dict]:
    """Carga el historial de análisis."""
    try:
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando historial para scoring general: {e}")
    return []


@router.post("/calculate")
async def calculate_score(data: Dict[str, Any]):
    """
    Calcula el score de calidad para una llamada.
    Espera el mismo formato de datos que retorna /transcribe/full-analysis.
    """
    try:
        segments = data.get("segments", [])
        global_emotions = data.get("global_emotions", {})
        speaker_stats = data.get("diarization", {}).get("speaker_stats", {})

        result = calculate_quality_score(segments, global_emotions, speaker_stats)

        return {
            "status": "success",
            "quality_score": result.total_score,
            "classification": result.classification,
            "breakdown": result.breakdown,
            "details": result.details,
            "recommendations": result.recommendations
        }
    except Exception as e:
        logger.error(f"Error calculando score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/general")
async def get_general_score():
    """
    Calcula el score de calidad GENERAL consolidado de TODOS los audios procesados.
    Lee el historial completo y retorna métricas agregadas:
    - Score promedio, min, max, desviación estándar
    - Promedios por dimensión (tono, keywords, resolución, protocolo)
    - Distribución de clasificaciones
    - Top issues detectados
    - Mejores y peores audios
    - Recomendaciones globales
    """
    try:
        history = _load_history()

        if not history:
            return {
                "status": "success",
                "message": "No hay datos en el historial",
                "total_audios": 0
            }

        result = calculate_general_quality_metrics(history)

        return {
            "status": "success",
            **asdict(result)
        }
    except Exception as e:
        logger.error(f"Error en scoring general: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def calculate_batch_scores(data: Dict[str, Any]):
    """
    Calcula scores para múltiples resultados de batch.
    Espera el formato de /transcribe/batch-scoring.
    """
    try:
        results = data.get("results", [])
        scores = []

        for r in results:
            if r.get("status") != "success":
                scores.append({
                    "filename": r.get("filename", "unknown"),
                    "status": "skipped",
                    "reason": "Procesamiento fallido"
                })
                continue

            segments = r.get("segments", [])
            global_emotions = r.get("global_emotions", {})
            speaker_stats = r.get("diarization", {}).get("speaker_stats", {})

            result = calculate_quality_score(segments, global_emotions, speaker_stats)

            scores.append({
                "filename": r.get("filename", "unknown"),
                "status": "success",
                "quality_score": result.total_score,
                "classification": result.classification,
                "breakdown": result.breakdown,
                "recommendations": result.recommendations
            })

        # Estadísticas del batch
        valid_scores = [s["quality_score"] for s in scores if s.get("status") == "success"]
        avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0

        return {
            "status": "success",
            "scores": scores,
            "summary": {
                "average_score": avg_score,
                "total_evaluated": len(valid_scores),
                "excellent": sum(1 for s in valid_scores if s >= 85),
                "good": sum(1 for s in valid_scores if 70 <= s < 85),
                "regular": sum(1 for s in valid_scores if 50 <= s < 70),
                "deficient": sum(1 for s in valid_scores if s < 50)
            }
        }
    except Exception as e:
        logger.error(f"Error en batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))
