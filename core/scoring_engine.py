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


# =====================================================
# SCORING GENERAL (CONSOLIDADO DE TODOS LOS AUDIOS)
# =====================================================

@dataclass
class GeneralScoringResult:
    """Resultado del scoring general consolidado."""
    total_audios: int
    avg_score: float
    std_deviation: float
    min_score: int
    max_score: int
    general_classification: str
    classification_distribution: Dict[str, int]
    dimension_averages: Dict[str, float]
    dimension_details: Dict[str, Any]
    emotion_summary: Dict[str, Any]
    global_recommendations: List[str]
    top_issues: List[str]
    best_audios: List[Dict[str, Any]]
    worst_audios: List[Dict[str, Any]]


def calculate_general_quality_metrics(
    history_items: List[Dict[str, Any]]
) -> GeneralScoringResult:
    """
    Calcula métricas de calidad consolidadas a partir del historial de análisis.

    Args:
        history_items: Lista de entradas del historial (data/analysis_history.json)

    Returns:
        GeneralScoringResult con métricas generales, promedios por dimensión,
        distribución de clasificaciones y recomendaciones globales.
    """
    import math

    # Filtrar items que tienen quality_score válido
    valid_items = []
    for item in history_items:
        qs = item.get("quality_score")
        if isinstance(qs, dict) and "total_score" in qs:
            valid_items.append(item)

    if not valid_items:
        return GeneralScoringResult(
            total_audios=0,
            avg_score=0.0,
            std_deviation=0.0,
            min_score=0,
            max_score=0,
            general_classification="Sin datos",
            classification_distribution={},
            dimension_averages={},
            dimension_details={},
            emotion_summary={},
            global_recommendations=["No hay audios con quality score para analizar."],
            top_issues=[],
            best_audios=[],
            worst_audios=[]
        )

    # ─── Scores totales ───
    scores = [item["quality_score"]["total_score"] for item in valid_items]
    avg_score = round(sum(scores) / len(scores), 1)
    std_dev = round(math.sqrt(sum((s - avg_score) ** 2 for s in scores) / len(scores)), 1)
    min_score = min(scores)
    max_score = max(scores)

    # ─── Distribución de clasificaciones ───
    classification_dist = {"Excelente": 0, "Bueno": 0, "Regular": 0, "Deficiente": 0}
    for s in scores:
        cls = _classify_score(s)
        if cls in classification_dist:
            classification_dist[cls] += 1

    # ─── Promedios por dimensión ───
    dimension_sums = {
        "emotional_tone": {"score": 0, "max": 30, "count": 0},
        "keywords": {"score": 0, "max": 25, "count": 0},
        "resolution": {"score": 0, "max": 25, "count": 0},
        "protocol": {"score": 0, "max": 20, "count": 0},
    }

    # Contadores detallados para dimensiones
    protocol_has_greeting = 0
    protocol_has_farewell = 0
    protocol_has_id = 0
    total_empathy_count = 0
    total_prohibited_count = 0
    total_resolution_kw = 0
    high_negative_total = 0

    for item in valid_items:
        breakdown = item["quality_score"].get("breakdown", {})
        for dim_key, dim_data in breakdown.items():
            if dim_key in dimension_sums and isinstance(dim_data, dict):
                dimension_sums[dim_key]["score"] += dim_data.get("score", 0)
                dimension_sums[dim_key]["count"] += 1

            # Detalles de protocolo
            if dim_key == "protocol" and isinstance(dim_data, dict):
                if dim_data.get("has_greeting"):
                    protocol_has_greeting += 1
                if dim_data.get("has_farewell"):
                    protocol_has_farewell += 1
                if dim_data.get("has_identification"):
                    protocol_has_id += 1

            # Detalles de keywords
            if dim_key == "keywords" and isinstance(dim_data, dict):
                total_empathy_count += dim_data.get("empathy_count", 0)
                total_prohibited_count += dim_data.get("prohibited_count", 0)
                total_resolution_kw += dim_data.get("resolution_count", 0)

            # Detalles de resolución
            if dim_key == "resolution" and isinstance(dim_data, dict):
                high_negative_total += dim_data.get("high_negative_segments", 0)

    dimension_averages = {}
    for dim_key, data in dimension_sums.items():
        if data["count"] > 0:
            avg = round(data["score"] / data["count"], 1)
            dimension_averages[dim_key] = avg
        else:
            dimension_averages[dim_key] = 0.0

    label_map = {
        "emotional_tone": "Tono Emocional",
        "keywords": "Palabras Clave",
        "resolution": "Resolución",
        "protocol": "Protocolo"
    }

    dimension_details = {}
    for dim_key, avg_val in dimension_averages.items():
        max_val = dimension_sums[dim_key]["max"]
        pct = round((avg_val / max_val) * 100, 1) if max_val > 0 else 0
        dimension_details[dim_key] = {
            "label": label_map.get(dim_key, dim_key),
            "average_score": avg_val,
            "max_score": max_val,
            "percentage": pct,
        }

    # Agregar detalles extra de protocolo
    n = len(valid_items)
    dimension_details["protocol"]["greeting_compliance"] = round(protocol_has_greeting / n * 100, 1)
    dimension_details["protocol"]["farewell_compliance"] = round(protocol_has_farewell / n * 100, 1)
    dimension_details["protocol"]["identification_compliance"] = round(protocol_has_id / n * 100, 1)

    # Detalles extra de keywords
    dimension_details["keywords"]["avg_empathy_per_call"] = round(total_empathy_count / n, 1)
    dimension_details["keywords"]["avg_prohibited_per_call"] = round(total_prohibited_count / n, 2)
    dimension_details["keywords"]["total_prohibited_found"] = total_prohibited_count

    # ─── Resumen emocional global ───
    emotion_counts = {"feliz": 0, "enojado": 0, "triste": 0, "neutral": 0}
    for item in valid_items:
        dom = item.get("dominant_emotion", "neutral")
        if dom in emotion_counts:
            emotion_counts[dom] += 1

    emotion_summary = {
        "distribution": {
            emo: round(count / n * 100, 1) for emo, count in emotion_counts.items()
        },
        "most_common": max(emotion_counts, key=emotion_counts.get),
        "alert_count": sum(1 for item in valid_items if item.get("has_alert", False)),
        "alert_rate": round(
            sum(1 for item in valid_items if item.get("has_alert", False)) / n * 100, 1
        ),
    }

    # ─── Top issues (problemas más frecuentes) ───
    issue_counter: Dict[str, int] = {}
    for item in valid_items:
        recs = item["quality_score"].get("recommendations", [])
        for rec in recs:
            # Limpiar emoji al inicio para agrupar
            clean_rec = rec.lstrip("⚠️🔴💬🚫👋🤝🏷️🎯✅ ")
            if "Excelente desempeño" not in clean_rec:
                issue_counter[clean_rec] = issue_counter.get(clean_rec, 0) + 1

    top_issues = sorted(issue_counter.items(), key=lambda x: x[1], reverse=True)
    top_issues_list = [
        f"{issue} ({count}/{n} audios, {round(count/n*100)}%)"
        for issue, count in top_issues[:8]
    ]

    # ─── Mejores y peores audios ───
    scored_audios = sorted(
        [
            {
                "filename": item.get("filename", "unknown"),
                "score": item["quality_score"]["total_score"],
                "classification": item["quality_score"].get("classification", "N/A"),
                "dominant_emotion": item.get("dominant_emotion", "neutral"),
                "timestamp": item.get("timestamp", ""),
            }
            for item in valid_items
        ],
        key=lambda x: x["score"],
        reverse=True
    )
    best_audios = scored_audios[:5]
    worst_audios = scored_audios[-5:][::-1] if len(scored_audios) >= 5 else scored_audios[::-1]

    # ─── Recomendaciones globales ───
    global_recs = []

    if avg_score < 50:
        global_recs.append("🔴 Score general DEFICIENTE. Se requiere capacitación urgente del equipo.")
    elif avg_score < 70:
        global_recs.append("⚠️ Score general REGULAR. Plan de mejora continua recomendado.")
    elif avg_score < 85:
        global_recs.append("🟢 Score general BUENO. Identificar áreas específicas de mejora.")
    else:
        global_recs.append("🏆 Score general EXCELENTE. Mantener los estándares actuales.")

    # Recomendaciones por dimensión débil
    weakest_dim = min(dimension_averages, key=dimension_averages.get)
    weakest_pct = dimension_details[weakest_dim]["percentage"]
    if weakest_pct < 60:
        global_recs.append(
            f"📉 Área más débil: {label_map.get(weakest_dim, weakest_dim)} "
            f"({weakest_pct}%). Requiere atención prioritaria."
        )

    if dimension_details["protocol"]["greeting_compliance"] < 70:
        global_recs.append(
            f"👋 Solo {dimension_details['protocol']['greeting_compliance']}% "
            f"de las llamadas incluyen saludo. Reforzar protocolo de inicio."
        )

    if dimension_details["protocol"]["farewell_compliance"] < 70:
        global_recs.append(
            f"🤝 Solo {dimension_details['protocol']['farewell_compliance']}% "
            f"de las llamadas incluyen despedida. Reforzar protocolo de cierre."
        )

    if total_prohibited_count > 0:
        global_recs.append(
            f"🚫 Se detectaron {total_prohibited_count} palabras/frases prohibidas "
            f"en total. Revisión urgente del lenguaje utilizado."
        )

    if emotion_summary["alert_rate"] > 20:
        global_recs.append(
            f"🚨 Tasa de alertas elevada ({emotion_summary['alert_rate']}%). "
            f"Revisar manejo de situaciones conflictivas."
        )

    if std_dev > 20:
        global_recs.append(
            f"📊 Alta variabilidad en scores (desv. estándar: {std_dev}). "
            f"Estandarizar la calidad del servicio."
        )

    general_classification = _classify_score(round(avg_score))

    return GeneralScoringResult(
        total_audios=n,
        avg_score=avg_score,
        std_deviation=std_dev,
        min_score=min_score,
        max_score=max_score,
        general_classification=general_classification,
        classification_distribution=classification_dist,
        dimension_averages=dimension_averages,
        dimension_details=dimension_details,
        emotion_summary=emotion_summary,
        global_recommendations=global_recs,
        top_issues=top_issues_list,
        best_audios=best_audios,
        worst_audios=worst_audios
    )
