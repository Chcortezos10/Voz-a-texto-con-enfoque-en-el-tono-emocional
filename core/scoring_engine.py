"""
Motor de Scoring de Calidad del Agente (Agent Quality Score).
Calcula una puntuación automática de 0-100 basada en:
  - Tono Emocional (30%)
  - Palabras Clave (25%)
  - Resolución de Conflictos (25%)
  - Protocolo de saludo/despedida (20%)
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =====================================================
# LEXICONES DE PALABRAS CLAVE
# =====================================================

# Saludos esperados al inicio de la llamada
GREETING_KEYWORDS = [
    "buenos días", "buenas tardes", "buenas noches",
    "buen día", "hola", "bienvenido", "bienvenida",
    "gracias por llamar", "gracias por comunicarse",
    "mi nombre es", "le habla", "le atiende",
    "en qué puedo ayudarle", "en qué le puedo ayudar",
    "cómo le puedo ayudar", "cómo puedo ayudarte",
    "a la orden",
]

# Despedidas esperadas al final de la llamada
FAREWELL_KEYWORDS = [
    "gracias por llamar", "gracias por comunicarse",
    "fue un placer", "que tenga un buen día",
    "que tenga buena tarde", "que tenga buena noche",
    "hasta luego", "adiós", "que le vaya bien",
    "algo más en lo que pueda ayudarle",
    "algo más", "necesita algo más",
    "con gusto", "a la orden",
    "feliz día", "feliz tarde", "feliz noche",
    "estamos para servirle",
]

# Palabras de empatía y servicio al cliente
EMPATHY_KEYWORDS = [
    "entiendo", "comprendo", "lamento", "lo siento",
    "disculpe", "permítame", "con gusto", "por supuesto",
    "claro que sí", "no se preocupe", "le ayudo",
    "tiene razón", "le entiendo perfectamente",
    "vamos a resolver", "voy a ayudarle",
    "me imagino", "debe ser difícil",
    "le comento", "le informo", "le explico",
]

# Palabras de resolución (indican que se está resolviendo el problema)
RESOLUTION_KEYWORDS = [
    "resuelto", "solucionado", "listo", "ya está",
    "quedó registrado", "se realizó", "ya se hizo",
    "confirmado", "procesado", "actualizado",
    "correcto", "efectivamente", "así es",
    "le confirmo", "queda registrado",
    "se generó", "se envió", "se activó",
    "en las próximas horas", "se va a reflejar",
]

# Palabras prohibidas / negativas del agente
PROHIBITED_KEYWORDS = [
    "no sé", "no puedo", "eso no es mi problema",
    "llame después", "no me interesa",
    "cálmese", "cállese", "estúpido", "estúpida",
    "idiota", "inútil", "incompetente",
    "ese no es mi departamento",
]


@dataclass
class ScoringResult:
    """Resultado del scoring de calidad."""
    total_score: int  # 0-100
    classification: str  # Excelente, Bueno, Regular, Deficiente
    breakdown: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


def _check_keywords(text: str, keywords: List[str]) -> List[str]:
    """Busca palabras clave en el texto y retorna las encontradas."""
    text_lower = text.lower()
    found = []
    for kw in keywords:
        if kw in text_lower:
            found.append(kw)
    return found


def _score_emotional_tone(
    segments: List[Dict],
    global_emotions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa el tono emocional (30 puntos máximo).
    - Segmentos positivos/neutrales suman puntos
    - Segmentos negativos restan puntos
    """
    if not segments:
        return {"score": 15, "max": 30, "details": "Sin segmentos para evaluar"}

    emotion_dist = global_emotions.get("emotion_distribution", {})
    feliz = emotion_dist.get("feliz", 0.0)
    neutral = emotion_dist.get("neutral", 0.0)
    enojado = emotion_dist.get("enojado", 0.0)
    triste = emotion_dist.get("triste", 0.0)

    positive_ratio = feliz + neutral
    negative_ratio = enojado + triste

    # Base: 15 puntos (neutro)
    # Positivo: hasta +15
    # Negativo: hasta -15
    score = 15 + (positive_ratio * 15) - (negative_ratio * 15)
    score = max(0, min(30, round(score)))

    # Bonus por evolución positiva (inicio negativo → final positivo)
    if len(segments) >= 4:
        first_quarter = segments[:len(segments)//4]
        last_quarter = segments[-len(segments)//4:]

        first_neg = sum(1 for s in first_quarter if s.get("emotion") in ("enojado", "triste"))
        last_pos = sum(1 for s in last_quarter if s.get("emotion") in ("feliz", "neutral"))

        if first_neg > 0 and last_pos > len(last_quarter) * 0.6:
            score = min(30, score + 3)  # Bonus por mejorar situación

    return {
        "score": score,
        "max": 30,
        "positive_ratio": round(positive_ratio, 3),
        "negative_ratio": round(negative_ratio, 3),
        "evolution_bonus": score > 27
    }


def _score_keywords(segments: List[Dict]) -> Dict[str, Any]:
    """
    Evalúa uso de palabras clave (25 puntos máximo).
    - Empatía detectada: +10
    - Resolución detectada: +10
    - Palabras prohibidas: -5 cada una (máx -15)
    - Bonus sin prohibidas: +5
    """
    full_text = " ".join(s.get("text_es", s.get("text", "")) for s in segments)

    empathy_found = _check_keywords(full_text, EMPATHY_KEYWORDS)
    resolution_found = _check_keywords(full_text, RESOLUTION_KEYWORDS)
    prohibited_found = _check_keywords(full_text, PROHIBITED_KEYWORDS)

    score = 0

    # Empatía: hasta 10 puntos
    empathy_score = min(10, len(empathy_found) * 3)
    score += empathy_score

    # Resolución: hasta 10 puntos
    resolution_score = min(10, len(resolution_found) * 3)
    score += resolution_score

    # Palabras prohibidas: penalty
    prohibited_penalty = min(15, len(prohibited_found) * 5)
    score -= prohibited_penalty

    # Bonus si no hay prohibidas
    if not prohibited_found:
        score += 5

    score = max(0, min(25, score))

    return {
        "score": score,
        "max": 25,
        "empathy_found": empathy_found[:5],
        "resolution_found": resolution_found[:5],
        "prohibited_found": prohibited_found,
        "empathy_count": len(empathy_found),
        "resolution_count": len(resolution_found),
        "prohibited_count": len(prohibited_found)
    }


def _score_resolution(
    segments: List[Dict],
    global_emotions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa indicadores de resolución (25 puntos máximo).
    - Evolución emocional positiva: +8
    - Keywords de resolución en últimos segmentos: +8
    - Llamada termina en tono positivo/neutro: +5
    - Sin segmentos de alta negatividad: +4
    """
    if not segments:
        return {"score": 12, "max": 25, "details": "Sin segmentos"}

    score = 0

    # 1. Evolución emocional (inicio vs fin)
    if len(segments) >= 3:
        last_emotions = [s.get("emotion", "neutral") for s in segments[-3:]]
        positive_end = sum(1 for e in last_emotions if e in ("feliz", "neutral"))
        score += min(8, positive_end * 3)
    else:
        score += 4  # Medio por defecto

    # 2. Keywords de resolución en últimos segmentos
    last_text = " ".join(
        s.get("text_es", s.get("text", "")) for s in segments[-5:]
    )
    resolution_found = _check_keywords(last_text, RESOLUTION_KEYWORDS)
    score += min(8, len(resolution_found) * 3)

    # 3. Tono final
    if segments:
        last_emotion = segments[-1].get("emotion", "neutral")
        if last_emotion in ("feliz", "neutral"):
            score += 5
        elif last_emotion == "triste":
            score += 2

    # 4. Sin picos de negatividad extrema
    high_negative = sum(
        1 for s in segments
        if s.get("intensity", 0) > 0.7 and s.get("emotion") in ("enojado", "triste")
    )
    if high_negative == 0:
        score += 4
    elif high_negative <= 2:
        score += 2

    score = max(0, min(25, score))

    return {
        "score": score,
        "max": 25,
        "positive_ending": segments[-1].get("emotion", "neutral") if segments else "N/A",
        "resolution_keywords_found": len(resolution_found),
        "high_negative_segments": high_negative
    }


def _score_protocol(segments: List[Dict]) -> Dict[str, Any]:
    """
    Evalúa protocolo de atención (20 puntos máximo).
    - Saludo al inicio: +8
    - Despedida al final: +8
    - Identificación del agente: +4
    """
    if not segments:
        return {"score": 10, "max": 20, "details": "Sin segmentos"}

    score = 0

    # Texto de los primeros y últimos segmentos
    first_text = " ".join(
        s.get("text_es", s.get("text", "")) for s in segments[:3]
    )
    last_text = " ".join(
        s.get("text_es", s.get("text", "")) for s in segments[-3:]
    )

    # 1. Saludo al inicio
    greetings_found = _check_keywords(first_text, GREETING_KEYWORDS)
    has_greeting = len(greetings_found) > 0
    if has_greeting:
        score += 8

    # 2. Despedida al final
    farewells_found = _check_keywords(last_text, FAREWELL_KEYWORDS)
    has_farewell = len(farewells_found) > 0
    if has_farewell:
        score += 8

    # 3. Identificación del agente
    id_keywords = ["mi nombre es", "le habla", "le atiende", "soy"]
    id_found = _check_keywords(first_text, id_keywords)
    has_identification = len(id_found) > 0
    if has_identification:
        score += 4

    score = max(0, min(20, score))

    return {
        "score": score,
        "max": 20,
        "has_greeting": has_greeting,
        "has_farewell": has_farewell,
        "has_identification": has_identification,
        "greetings_found": greetings_found[:3],
        "farewells_found": farewells_found[:3]
    }


def _classify_score(score: int) -> str:
    """Clasifica el score en categoría."""
    if score >= 85:
        return "Excelente"
    elif score >= 70:
        return "Bueno"
    elif score >= 50:
        return "Regular"
    else:
        return "Deficiente"


def _generate_recommendations(breakdown: Dict[str, Any]) -> List[str]:
    """Genera recomendaciones basadas en el desglose."""
    recs = []

    tone = breakdown.get("emotional_tone", {})
    keywords = breakdown.get("keywords", {})
    protocol = breakdown.get("protocol", {})
    resolution = breakdown.get("resolution", {})

    if tone.get("score", 0) < 15:
        recs.append("⚠️ Mejorar el tono emocional general. Se detectaron muchas emociones negativas.")

    if tone.get("negative_ratio", 0) > 0.4:
        recs.append("🔴 Alta proporción de emociones negativas. Considerar capacitación en manejo de emociones.")

    if keywords.get("empathy_count", 0) < 2:
        recs.append("💬 Usar más expresiones de empatía (\"entiendo\", \"comprendo\", \"le ayudo\").")

    if keywords.get("prohibited_count", 0) > 0:
        recs.append(f"🚫 Se detectaron {keywords['prohibited_count']} palabras/frases prohibidas. Revisar urgente.")

    if not protocol.get("has_greeting"):
        recs.append("👋 Falta saludo al inicio de la llamada. Incluir saludo y presentación.")

    if not protocol.get("has_farewell"):
        recs.append("🤝 Falta despedida al final. Incluir despedida cordial y ofrecimiento de ayuda adicional.")

    if not protocol.get("has_identification"):
        recs.append("🏷️ El agente no se identificó. Incluir nombre y cargo al inicio.")

    if resolution.get("score", 0) < 12:
        recs.append("🎯 Mejorar indicadores de resolución. Confirmar al cliente que su problema fue resuelto.")

    if not recs:
        recs.append("✅ Excelente desempeño. Mantener la calidad de atención.")

    return recs


def calculate_quality_score(
    segments: List[Dict],
    global_emotions: Dict[str, Any],
    speaker_stats: Optional[Dict] = None
) -> ScoringResult:
    """
    Calcula el score de calidad del agente (0-100).

    Args:
        segments: Lista de segmentos enriquecidos con emociones
        global_emotions: Estadísticas emocionales globales
        speaker_stats: Estadísticas por hablante (opcional)

    Returns:
        ScoringResult con score total, clasificación, desglose y recomendaciones
    """
    try:
        # Calcular cada dimensión
        tone_result = _score_emotional_tone(segments, global_emotions)
        keywords_result = _score_keywords(segments)
        resolution_result = _score_resolution(segments, global_emotions)
        protocol_result = _score_protocol(segments)

        # Score total
        total = (
            tone_result["score"] +
            keywords_result["score"] +
            resolution_result["score"] +
            protocol_result["score"]
        )
        total = max(0, min(100, total))

        breakdown = {
            "emotional_tone": tone_result,
            "keywords": keywords_result,
            "resolution": resolution_result,
            "protocol": protocol_result
        }

        classification = _classify_score(total)
        recommendations = _generate_recommendations(breakdown)

        return ScoringResult(
            total_score=total,
            classification=classification,
            breakdown=breakdown,
            details={
                "total_segments": len(segments),
                "num_speakers": len(speaker_stats) if speaker_stats else 0,
                "dominant_emotion": global_emotions.get("top_emotion", "neutral")
            },
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error calculando quality score: {e}")
        return ScoringResult(
            total_score=0,
            classification="Error",
            breakdown={},
            details={"error": str(e)},
            recommendations=["Error en el cálculo. Verificar datos de entrada."]
        )
