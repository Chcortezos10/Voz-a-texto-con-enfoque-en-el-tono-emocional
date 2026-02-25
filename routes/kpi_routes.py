"""
Rutas de KPIs por Agente/Equipo.
Consume el historial de análisis para generar tendencias y comparativas.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
import logging
import json
from pathlib import Path

from core.scoring_engine import calculate_general_quality_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kpi", tags=["KPIs"])

history_file = Path("data/analysis_history.json")


def _load_history() -> List[Dict]:
    """Carga el historial de análisis."""
    try:
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando historial para KPIs: {e}")
    return []


@router.get("/summary")
async def kpi_summary():
    """
    Resumen general de KPIs.
    """
    try:
        history = _load_history()

        if not history:
            return {
                "status": "success",
                "total_calls": 0,
                "avg_sentiment_score": 0,
                "emotion_distribution": {"feliz": 0, "enojado": 0, "triste": 0, "neutral": 0},
                "message": "No hay datos en el historial"
            }

        total = len(history)
        emotions_count = {"feliz": 0, "enojado": 0, "triste": 0, "neutral": 0}

        for item in history:
            dom = item.get("dominant_emotion", "neutral")
            if dom in emotions_count:
                emotions_count[dom] += 1

        # Convert to proportions (0-1) for dashboard charts
        emotion_distribution = {}
        for emo, count in emotions_count.items():
            emotion_distribution[emo] = round(count / total, 4) if total > 0 else 0

        # Average sentiment score (from saved sentiment_score field)
        sentiment_scores = [
            item.get("sentiment_score", 0)
            for item in history
            if item.get("sentiment_score") is not None
        ]
        avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 4) if sentiment_scores else 0

        # Average quality score
        quality_scores = [
            item["quality_score"]["total_score"]
            for item in history
            if isinstance(item.get("quality_score"), dict) and "total_score" in item["quality_score"]
        ]
        avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else None

        # Alert stats
        alert_count = sum(1 for item in history if item.get("has_alert", False))

        return {
            "status": "success",
            "total_calls": total,
            "avg_sentiment_score": avg_sentiment,
            "average_quality_score": avg_quality,
            "emotion_distribution": emotion_distribution,
            "positive_percentage": round(emotions_count.get("feliz", 0) / total * 100, 1) if total > 0 else 0,
            "negative_percentage": round((emotions_count.get("enojado", 0) + emotions_count.get("triste", 0)) / total * 100, 1) if total > 0 else 0,
            "most_common_emotion": max(emotions_count, key=emotions_count.get) if emotions_count else "neutral",
            "alert_count": alert_count
        }
    except Exception as e:
        logger.error(f"Error en KPI summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/general-score")
async def kpi_general_score():
    """
    Score de calidad GENERAL consolidado con desglose por dimensión.
    Retorna métricas completas: promedios, distribución, top issues,
    mejores/peores audios, y recomendaciones globales.
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
        logger.error(f"Error en KPI general-score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def kpi_trends(
    period: str = Query("7d", description="Período: 7d, 30d, 90d")
):
    """
    Tendencias de emociones en el tiempo.
    Retorna datos agrupados por día para gráficos.
    """
    try:
        history = _load_history()

        # Determinar período
        days_map = {"7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(period, 7)
        cutoff = datetime.now() - timedelta(days=days)

        # Filtrar por período
        filtered = []
        for item in history:
            try:
                ts = datetime.fromisoformat(item.get("timestamp", ""))
                if ts >= cutoff:
                    filtered.append(item)
            except (ValueError, TypeError):
                filtered.append(item)  # Incluir si no tiene timestamp válido

        # Agrupar por fecha
        daily_data = {}
        for item in filtered:
            try:
                date_str = item.get("timestamp", "")[:10]  # YYYY-MM-DD
            except:
                date_str = "unknown"

            if date_str not in daily_data:
                daily_data[date_str] = {
                    "date": date_str,
                    "total_calls": 0,
                    "emotions": {"feliz": 0, "enojado": 0, "triste": 0, "neutral": 0},
                    "scores": []
                }

            daily_data[date_str]["total_calls"] += 1

            dom = item.get("dominant_emotion", "neutral")
            if dom in daily_data[date_str]["emotions"]:
                daily_data[date_str]["emotions"][dom] += 1

            if isinstance(item.get("quality_score"), dict):
                daily_data[date_str]["scores"].append(
                    item["quality_score"].get("total_score", 0)
                )

        # Calcular promedios y convertir emociones a proporciones
        daily_trends = []
        for date_str in sorted(daily_data.keys()):
            d = daily_data[date_str]
            scores = d.pop("scores")
            d["avg_score"] = round(sum(scores) / len(scores), 1) if scores else None
            # Convertir emociones a proporciones (0-1)
            total_calls = d["total_calls"]
            if total_calls > 0:
                for emo in d["emotions"]:
                    d["emotions"][emo] = round(d["emotions"][emo] / total_calls, 4)
            daily_trends.append(d)

        return {
            "status": "success",
            "period": period,
            "data_points": len(daily_trends),
            "daily_trends": daily_trends
        }
    except Exception as e:
        logger.error(f"Error en KPI trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def kpi_comparison():
    """
    Comparativa entre llamadas/agentes.
    """
    try:
        history = _load_history()

        if not history:
            return {"status": "success", "calls": [], "message": "Sin datos"}

        calls = []
        for item in history[-20:]:  # Últimas 20 llamadas
            emotion_dist = item.get("global_emotions", {}).get("emotion_distribution", {})
            quality = item.get("quality_score", {})

            calls.append({
                "filename": item.get("filename", "unknown"),
                "timestamp": item.get("timestamp", ""),
                "dominant_emotion": item.get("dominant_emotion", "neutral"),
                "sentiment_score": item.get("sentiment_score", 0),
                "quality_score": quality.get("total_score") if isinstance(quality, dict) else None,
                "classification": quality.get("classification") if isinstance(quality, dict) else None,
                "feliz": round(emotion_dist.get("feliz", 0) * 100, 1),
                "enojado": round(emotion_dist.get("enojado", 0) * 100, 1),
                "triste": round(emotion_dist.get("triste", 0) * 100, 1),
                "neutral": round(emotion_dist.get("neutral", 0) * 100, 1),
                "duration": item.get("duration", 0),
                "num_segments": item.get("num_segments", 0),
                "has_alert": item.get("has_alert", False)
            })

        return {
            "status": "success",
            "total_compared": len(calls),
            "calls": calls
        }
    except Exception as e:
        logger.error(f"Error en KPI comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))
