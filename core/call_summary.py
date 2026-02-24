"""
Generador Automático de Resumen de Llamada.
Crea un resumen estructurado sin necesidad de LLM externo,
basado en heurísticas del contenido y emociones detectadas.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Categorías de motivo de llamada basadas en palabras clave
TOPIC_KEYWORDS = {
    "facturación": ["factura", "cobro", "pago", "cargo", "cuenta", "recibo", "deuda", "mora", "saldo"],
    "soporte técnico": ["internet", "señal", "conexión", "sistema", "error", "falla", "no funciona", "caído", "lento"],
    "consulta": ["información", "consulta", "pregunta", "saber", "cómo", "cuándo", "dónde", "precio", "plan"],
    "reclamo": ["reclamo", "queja", "molesto", "inconveniente", "problema", "mal servicio", "pésimo"],
    "cancelación": ["cancelar", "cancelación", "dar de baja", "no quiero", "retirar", "devolver"],
    "activación": ["activar", "activación", "nuevo", "contratar", "adquirir", "solicitar"],
    "seguimiento": ["seguimiento", "estado", "proceso", "solicitud", "ticket", "caso", "reporte"],
}

SATISFACTION_LEVELS = {
    "alta": "El cliente terminó la llamada con tono positivo",
    "media": "El cliente terminó la llamada en tono neutral",
    "baja": "El cliente terminó la llamada con tono negativo",
    "muy_baja": "El cliente mostró frustración o enojo al final"
}


@dataclass
class CallSummary:
    """Resumen estructurado de una llamada."""
    motivo: str
    desarrollo: str
    resolucion: str
    satisfaccion_estimada: str
    satisfaccion_detalle: str
    duracion_seg: float
    num_hablantes: int
    num_segmentos: int
    resumen_breve: str
    topico_detectado: str
    emociones_principales: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_topic(text: str) -> str:
    """Detecta el tópico principal de la llamada."""
    text_lower = text.lower()
    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            topic_scores[topic] = score

    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    return "general"


def _extract_motive(segments: List[Dict], num_speakers: int) -> str:
    """Extrae el motivo de la llamada de los primeros segmentos."""
    if not segments:
        return "No se pudo determinar el motivo de la llamada."

    # Tomar los primeros 3-5 segmentos
    early_segments = segments[:min(5, len(segments))]

    # Si hay diarización, intentar tomar solo segmentos del "cliente"
    # (asumimos que el agente es speaker_0 y el cliente speaker_1+)
    if num_speakers >= 2:
        client_segments = [s for s in early_segments if s.get("speaker_id", 0) != 0]
        if client_segments:
            early_segments = client_segments

    text = " ".join(
        s.get("text_es", s.get("text", "")) for s in early_segments
    ).strip()

    if len(text) > 200:
        text = text[:200] + "..."

    return text if text else "No se pudo determinar el motivo."


def _analyze_development(
    segments: List[Dict],
    global_emotions: Dict[str, Any]
) -> str:
    """Analiza el desarrollo emocional de la llamada."""
    if not segments:
        return "Sin datos de desarrollo."

    n = len(segments)
    emotion_dist = global_emotions.get("emotion_distribution", {})
    top_emotion = global_emotions.get("top_emotion", "neutral")

    # Dividir en tercios
    if n >= 6:
        tercio = n // 3
        inicio = segments[:tercio]
        medio = segments[tercio:2*tercio]
        final = segments[2*tercio:]

        def dominant(segs):
            emos = {}
            for s in segs:
                e = s.get("emotion", "neutral")
                emos[e] = emos.get(e, 0) + 1
            return max(emos, key=emos.get) if emos else "neutral"

        emo_inicio = dominant(inicio)
        emo_medio = dominant(medio)
        emo_final = dominant(final)

        parts = []
        parts.append(f"Inicio: tono {emo_inicio}")
        if emo_medio != emo_inicio:
            parts.append(f"desarrollo: cambio a {emo_medio}")
        else:
            parts.append(f"desarrollo: se mantuvo {emo_medio}")
        if emo_final != emo_medio:
            parts.append(f"cierre: transición a {emo_final}")
        else:
            parts.append(f"cierre: {emo_final}")

        return ". ".join(parts) + f". Emoción predominante: {top_emotion} ({emotion_dist.get(top_emotion, 0):.0%})"
    else:
        return f"Llamada breve con tono predominante {top_emotion} ({emotion_dist.get(top_emotion, 0):.0%})"


def _extract_resolution(segments: List[Dict], num_speakers: int) -> str:
    """Extrae indicadores de resolución de los últimos segmentos."""
    if not segments:
        return "No se pudo determinar la resolución."

    last_segments = segments[-min(5, len(segments)):]

    text = " ".join(
        s.get("text_es", s.get("text", "")) for s in last_segments
    ).strip()

    if len(text) > 200:
        text = text[:200] + "..."

    return text if text else "No se detectaron indicadores de resolución."


def _estimate_satisfaction(segments: List[Dict]) -> tuple:
    """Estima el nivel de satisfacción basado en evolución emocional."""
    if not segments:
        return "media", SATISFACTION_LEVELS["media"]

    # Evaluar los últimos segmentos (30% final)
    n = max(1, len(segments) // 3)
    final_segments = segments[-n:]

    positive = sum(1 for s in final_segments if s.get("emotion") in ("feliz",))
    neutral = sum(1 for s in final_segments if s.get("emotion") == "neutral")
    negative = sum(1 for s in final_segments if s.get("emotion") in ("enojado", "triste"))

    total = len(final_segments)

    if total == 0:
        return "media", SATISFACTION_LEVELS["media"]

    pos_ratio = (positive + neutral) / total
    neg_ratio = negative / total

    if positive / total > 0.3:
        return "alta", SATISFACTION_LEVELS["alta"]
    elif pos_ratio > 0.7:
        return "media", SATISFACTION_LEVELS["media"]
    elif neg_ratio > 0.5:
        emotion_final = final_segments[-1].get("emotion", "neutral")
        if emotion_final == "enojado":
            return "muy_baja", SATISFACTION_LEVELS["muy_baja"]
        return "baja", SATISFACTION_LEVELS["baja"]
    else:
        return "media", SATISFACTION_LEVELS["media"]


def generate_call_summary(
    segments: List[Dict],
    global_emotions: Dict[str, Any],
    speaker_stats: Optional[Dict] = None,
    duration: float = 0.0,
    filename: str = ""
) -> CallSummary:
    """
    Genera un resumen automático de la llamada.

    Args:
        segments: Segmentos enriquecidos con emociones
        global_emotions: Estadísticas emocionales globales
        speaker_stats: Estadísticas por hablante
        duration: Duración total en segundos
        filename: Nombre del archivo

    Returns:
        CallSummary con resumen estructurado
    """
    try:
        num_speakers = len(speaker_stats) if speaker_stats else 1

        # Texto completo para detección de tópico
        full_text = " ".join(
            s.get("text_es", s.get("text", "")) for s in segments
        )

        topic = _detect_topic(full_text)
        motivo = _extract_motive(segments, num_speakers)
        desarrollo = _analyze_development(segments, global_emotions)
        resolucion = _extract_resolution(segments, num_speakers)
        satisfaccion, satisfaccion_detalle = _estimate_satisfaction(segments)

        # Emociones principales (top 3)
        emotion_dist = global_emotions.get("emotion_distribution", {})
        sorted_emotions = sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [f"{e} ({v:.0%})" for e, v in sorted_emotions[:3]]

        # Resumen breve (una línea)
        topic_label = topic.capitalize() if topic != "general" else "Consulta"
        resumen_breve = (
            f"{topic_label} con tono {global_emotions.get('top_emotion', 'neutral')} "
            f"y satisfacción {satisfaccion}"
        )

        return CallSummary(
            motivo=motivo,
            desarrollo=desarrollo,
            resolucion=resolucion,
            satisfaccion_estimada=satisfaccion,
            satisfaccion_detalle=satisfaccion_detalle,
            duracion_seg=round(duration, 1),
            num_hablantes=num_speakers,
            num_segmentos=len(segments),
            resumen_breve=resumen_breve,
            topico_detectado=topic,
            emociones_principales=top_emotions
        )

    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        return CallSummary(
            motivo="Error generando resumen",
            desarrollo="",
            resolucion="",
            satisfaccion_estimada="desconocida",
            satisfaccion_detalle="Error en el procesamiento",
            duracion_seg=duration,
            num_hablantes=0,
            num_segmentos=len(segments),
            resumen_breve="Error en generación de resumen",
            topico_detectado="desconocido",
            emociones_principales=[]
        )
