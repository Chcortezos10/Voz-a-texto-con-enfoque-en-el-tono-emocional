"""
Sistema de Alertas de Escalamiento Automático.
Detecta llamadas con emociones negativas intensas y las marca para revisión.

Criterios de alerta:
- Alta negatividad emocional (enojado >= 40%)
- Segmentos consecutivos negativos (>= 3)
- Palabras críticas (insultos, amenazas)
- Sentiment score muy bajo (< -0.3)
- Quality score deficiente (< 50)
"""
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Palabras que indican escalamiento urgente
CRITICAL_WORDS = [
    # Amenazas
    "demanda", "demandar", "abogado", "legal", "denuncia", "denunciar",
    "superintendencia", "defensoría", "regulador",
    # Insultos fuertes
    "idiota", "imbécil", "estúpido", "estúpida", "inútil", "incompetente",
    "basura", "maldito", "maldita", "desgraciado", "desgraciada",
    # Escalamiento explícito
    "supervisor", "jefe", "gerente", "encargado",
    "quiero hablar con", "páseme con", "exijo hablar",
    "voy a cancelar", "cancelar todo", "me voy",
    "nunca más", "jamás vuelvo", "no vuelvo",
]

ALERTS_FILE = Path("data/alerts.json")


@dataclass
class Alert:
    """Una alerta de escalamiento."""
    id: str
    timestamp: str
    filename: str
    severity: str  # alta, media, baja
    reasons: List[str]
    quality_score: Optional[int] = None
    sentiment_score: Optional[float] = None
    dominant_emotion: str = "neutral"
    reviewed: bool = False
    review_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_alerts() -> List[Dict]:
    """Carga alertas desde archivo."""
    try:
        if ALERTS_FILE.exists():
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando alertas: {e}")
    return []


def _save_alerts(alerts: List[Dict]):
    """Guarda alertas a archivo."""
    try:
        ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error guardando alertas: {e}")


def check_alerts(
    segments: List[Dict],
    global_emotions: Dict[str, Any],
    filename: str = "unknown",
    quality_score: Optional[int] = None,
    sentiment_score: Optional[float] = None
) -> Optional[Alert]:
    """
    Evalúa una llamada y genera alerta si cumple criterios.

    Returns:
        Alert si se detectan problemas, None si todo está OK
    """
    reasons = []
    severity_level = 0  # 0=ok, 1=baja, 2=media, 3=alta

    emotion_dist = global_emotions.get("emotion_distribution", {})
    enojado = emotion_dist.get("enojado", 0.0)
    triste = emotion_dist.get("triste", 0.0)

    # 1. Alta negatividad emocional
    if enojado >= 0.4:
        reasons.append(f"🔴 Enojo muy alto: {enojado:.0%}")
        severity_level = max(severity_level, 3)
    elif enojado >= 0.25:
        reasons.append(f"🟡 Enojo elevado: {enojado:.0%}")
        severity_level = max(severity_level, 2)

    if triste >= 0.4:
        reasons.append(f"🔵 Tristeza muy alta: {triste:.0%}")
        severity_level = max(severity_level, 2)

    # 2. Segmentos consecutivos negativos
    consecutive_negative = 0
    max_consecutive = 0
    for seg in segments:
        if seg.get("emotion") in ("enojado", "triste"):
            consecutive_negative += 1
            max_consecutive = max(max_consecutive, consecutive_negative)
        else:
            consecutive_negative = 0

    if max_consecutive >= 5:
        reasons.append(f"🔴 {max_consecutive} segmentos negativos consecutivos")
        severity_level = max(severity_level, 3)
    elif max_consecutive >= 3:
        reasons.append(f"🟡 {max_consecutive} segmentos negativos consecutivos")
        severity_level = max(severity_level, 2)

    # 3. Palabras críticas
    full_text = " ".join(
        s.get("text_es", s.get("text", "")).lower() for s in segments
    )
    critical_found = [w for w in CRITICAL_WORDS if w in full_text]
    if critical_found:
        reasons.append(f"⚠️ Palabras críticas: {', '.join(critical_found[:5])}")
        severity_level = max(severity_level, 2)
        # Amenazas legales = alta
        legal_words = {"demanda", "demandar", "abogado", "legal", "denuncia"}
        if any(w in critical_found for w in legal_words):
            severity_level = max(severity_level, 3)

    # 4. Sentiment score muy bajo
    if sentiment_score is not None and sentiment_score < -0.3:
        reasons.append(f"📉 Sentiment score muy bajo: {sentiment_score:.3f}")
        severity_level = max(severity_level, 2)

    # 5. Quality score deficiente
    if quality_score is not None and quality_score < 50:
        reasons.append(f"📊 Score de calidad deficiente: {quality_score}/100")
        severity_level = max(severity_level, 2)
        if quality_score < 30:
            severity_level = max(severity_level, 3)

    # Si no hay razones, no generar alerta
    if not reasons:
        return None

    severity_map = {0: "baja", 1: "baja", 2: "media", 3: "alta"}
    severity = severity_map.get(severity_level, "media")

    alert = Alert(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        filename=filename,
        severity=severity,
        reasons=reasons,
        quality_score=quality_score,
        sentiment_score=sentiment_score,
        dominant_emotion=global_emotions.get("top_emotion", "neutral")
    )

    # Guardar la alerta
    alerts = _load_alerts()
    alerts.append(alert.to_dict())
    # Mantener máximo 200 alertas
    if len(alerts) > 200:
        alerts = alerts[-200:]
    _save_alerts(alerts)

    logger.warning(f"🚨 ALERTA [{severity.upper()}] para {filename}: {', '.join(reasons)}")

    return alert


def get_all_alerts(
    severity_filter: Optional[str] = None,
    reviewed_filter: Optional[bool] = None,
    limit: int = 50
) -> List[Dict]:
    """Retorna alertas filtradas."""
    alerts = _load_alerts()

    if severity_filter:
        alerts = [a for a in alerts if a.get("severity") == severity_filter]

    if reviewed_filter is not None:
        alerts = [a for a in alerts if a.get("reviewed") == reviewed_filter]

    # Ordenar por fecha (más recientes primero)
    alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return alerts[:limit]


def mark_alert_reviewed(alert_id: str, notes: str = "") -> bool:
    """Marca una alerta como revisada."""
    alerts = _load_alerts()
    for alert in alerts:
        if alert.get("id") == alert_id:
            alert["reviewed"] = True
            alert["review_notes"] = notes
            _save_alerts(alerts)
            return True
    return False


def delete_alert(alert_id: str) -> bool:
    """Elimina una alerta."""
    alerts = _load_alerts()
    initial = len(alerts)
    alerts = [a for a in alerts if a.get("id") != alert_id]
    if len(alerts) < initial:
        _save_alerts(alerts)
        return True
    return False


def get_alert_stats() -> Dict[str, Any]:
    """Retorna estadísticas de alertas."""
    alerts = _load_alerts()
    total = len(alerts)
    reviewed = sum(1 for a in alerts if a.get("reviewed"))
    pending = total - reviewed

    severity_counts = {}
    for a in alerts:
        sev = a.get("severity", "media")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        "total": total,
        "pending": pending,
        "reviewed": reviewed,
        "by_severity": severity_counts
    }
