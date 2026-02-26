"""
Módulo de interpretación de datos usando Ollama (Qwen2.5).
Se usa SOLO cuando se solicita un informe, NO en el flujo de procesamiento.
Genera interpretaciones en español de los datos consolidados de quality score y KPIs.
"""
import os
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# URL de Ollama (local por defecto, configurable vía env para Docker)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


async def generate_data_interpretation(
    general_metrics: Dict[str, Any],
    include_recommendations: bool = True
) -> Optional[str]:
    """
    Genera una interpretación en español de los datos consolidados usando Qwen2.5 vía Ollama.
    
    Args:
        general_metrics: Diccionario con las métricas generales del scoring
        include_recommendations: Si incluir recomendaciones detalladas
    
    Returns:
        Texto con la interpretación en español, o None si Ollama no está disponible.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx no instalado. Instalar con: pip install httpx")
        return None

    # Preparar resumen de datos para el prompt
    data_summary = _prepare_data_summary(general_metrics)

    prompt = f"""Eres un analista experto en calidad de servicio al cliente y call centers.
Analiza los siguientes datos consolidados de calidad de atención telefónica y genera un informe interpretativo en español.

DATOS DEL ANÁLISIS:
{data_summary}

Genera un informe ejecutivo que incluya:
1. **Diagnóstico General**: Evaluación del estado actual de la calidad del servicio
2. **Análisis por Dimensión**: Interpretación de cada área evaluada (tono emocional, uso de palabras clave, resolución de problemas, protocolo de atención)
3. **Patrones Identificados**: Tendencias y patrones relevantes en los datos
4. **Áreas Críticas**: Las áreas que requieren atención inmediata
{"5. **Recomendaciones Estratégicas**: Plan de acción concreto para mejorar la calidad" if include_recommendations else ""}

IMPORTANTE:
- Escribe en español profesional
- Sé específico con los datos y porcentajes
- Usa un tono profesional pero accesible
- Máximo 500 palabras
- No uses markdown con # headers, usa texto plano con viñetas
"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 1024,
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                interpretation = result.get("response", "")
                logger.info(f"Interpretación generada exitosamente ({len(interpretation)} chars)")
                return interpretation.strip()
            else:
                logger.warning(f"Ollama respondió con status {response.status_code}: {response.text[:200]}")
                return None

    except httpx.ConnectError:
        logger.warning(
            "No se pudo conectar a Ollama. "
            "Asegúrese de que Ollama está corriendo: ollama serve"
        )
        return None
    except Exception as e:
        logger.error(f"Error generando interpretación con Ollama: {e}")
        return None


def _prepare_data_summary(metrics: Dict[str, Any]) -> str:
    """Prepara un resumen textual de las métricas para el prompt del LLM."""
    lines = []

    lines.append(f"Total de audios analizados: {metrics.get('total_audios', 0)}")
    lines.append(f"Score promedio: {metrics.get('avg_score', 0)}/100")
    lines.append(f"Clasificación general: {metrics.get('general_classification', 'N/A')}")
    lines.append(f"Desviación estándar: {metrics.get('std_deviation', 0)}")
    lines.append(f"Score mínimo: {metrics.get('min_score', 0)}, máximo: {metrics.get('max_score', 0)}")

    # Distribución de clasificaciones
    dist = metrics.get("classification_distribution", {})
    if dist:
        lines.append("\nDistribución de clasificaciones:")
        for cls, count in dist.items():
            lines.append(f"  - {cls}: {count} audios")

    # Dimensiones
    dim_details = metrics.get("dimension_details", {})
    if dim_details:
        lines.append("\nPromedios por dimensión:")
        for dim_key, details in dim_details.items():
            if isinstance(details, dict):
                label = details.get("label", dim_key)
                avg = details.get("average_score", 0)
                max_s = details.get("max_score", 0)
                pct = details.get("percentage", 0)
                lines.append(f"  - {label}: {avg}/{max_s} ({pct}%)")

                # Detalles extra de protocolo
                if dim_key == "protocol":
                    gc = details.get("greeting_compliance", 0)
                    fc = details.get("farewell_compliance", 0)
                    ic = details.get("identification_compliance", 0)
                    lines.append(f"    Cumplimiento saludo: {gc}%, despedida: {fc}%, identificación: {ic}%")

                # Detalles extra de keywords
                if dim_key == "keywords":
                    emp = details.get("avg_empathy_per_call", 0)
                    proh = details.get("total_prohibited_found", 0)
                    lines.append(f"    Empatía promedio/llamada: {emp}, Palabras prohibidas total: {proh}")

    # Emociones
    emotion_summary = metrics.get("emotion_summary", {})
    if emotion_summary:
        lines.append(f"\nEmoción más común: {emotion_summary.get('most_common', 'N/A')}")
        lines.append(f"Tasa de alertas: {emotion_summary.get('alert_rate', 0)}%")
        dist = emotion_summary.get("distribution", {})
        if dist:
            lines.append("Distribución emocional:")
            for emo, pct in dist.items():
                lines.append(f"  - {emo}: {pct}%")

    # Top issues
    top_issues = metrics.get("top_issues", [])
    if top_issues:
        lines.append("\nProblemas más frecuentes:")
        for issue in top_issues[:5]:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


async def check_ollama_available() -> Dict[str, Any]:
    """
    Verifica si Ollama está disponible y el modelo Qwen2.5 está instalado.
    """
    try:
        import httpx
    except ImportError:
        return {"available": False, "reason": "httpx no instalado"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Verificar que Ollama está corriendo
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return {"available": False, "reason": "Ollama no responde correctamente"}

            # Verificar que el modelo está instalado
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            has_model = any(OLLAMA_MODEL in name for name in model_names)

            return {
                "available": True,
                "model_installed": has_model,
                "model": OLLAMA_MODEL,
                "available_models": model_names,
                "install_command": f"ollama pull {OLLAMA_MODEL}" if not has_model else None
            }

    except Exception as e:
        return {
            "available": False,
            "reason": f"No se pudo conectar a Ollama: {str(e)}",
            "install_instructions": "1. Instalar Ollama: https://ollama.ai\n2. Ejecutar: ollama serve\n3. Descargar modelo: ollama pull qwen2.5:7b"
        }
