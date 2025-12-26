# 📋 Registro de Cambios (CHANGELOG)

## Versión 2.0.0 - Análisis Emocional Multi-Modal

**Fecha:** 25 de Diciembre de 2024

### ✨ Nuevas Funcionalidades

#### Análisis Emocional Multi-Modal

- **Traducción automática ES→EN** usando Helsinki-NLP/opus-mt-es-en
- **Análisis emocional dual**:
  - Español: XLM-RoBERTa (`daveni/twitter-xlm-roberta-emotion-es`)
  - Inglés: DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`)
- **Fusión ponderada** configurable (60% texto, 40% audio por defecto)
- **Score emocional ponderado** por duración de segmentos

#### Nuevo Dashboard HTML

- Dashboard web interactivo (`dashboard.html`)
- Funciona sin dependencias adicionales (evita problemas de Streamlit/Tornado)
- Visualización de emociones por segmento
- Descarga de resultados en JSON

#### Nuevo Endpoint API

- `POST /transcribe/full-analysis` - Análisis completo multi-modal
- `GET /test-emotion` - Endpoint de diagnóstico

### 📦 Nuevos Módulos Creados

| Archivo                    | Descripción                       |
| -------------------------- | --------------------------------- |
| `core/translation.py`      | Traducción ES→EN con Helsinki-NLP |
| `core/emotion_analysis.py` | Análisis emocional multi-modal    |
| `dashboard.html`           | Dashboard web interactivo         |

### ⚙️ Cambios en Configuración (`config.py`)

```python
# Nuevas configuraciones agregadas:
WHISPER_MODEL = "small"  # Mejor precisión con GPU
DEVICE = "auto"  # Detecta GPU automáticamente
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"
EMOTION_ES_MODEL = "daveni/twitter-xlm-roberta-emotion-es"
EMOTION_WEIGHT_TEXT = 0.6
EMOTION_WEIGHT_AUDIO = 0.4
FUSION_MODE = "weighted_average"
CORS_ORIGINS = ["*"]  # Permitir dashboard HTML local
```

### 🔧 Mejoras Técnicas

- **Carga lazy de modelos** - Evita problemas de memoria al inicio
- **Soporte GPU** - Detecta CUDA automáticamente (RTX 4060)
- **CORS configurado** - Permite acceso desde archivos locales
- **Logging mejorado** - Mejor diagnóstico de errores

### 🧹 Limpieza de Código

#### Archivos Eliminados (13 archivos)

- 6 archivos JSON de outputs anteriores
- 2 archivos TXT de transcripciones
- 3 apps Streamlit redundantes
- 2 scripts de prueba legacy

#### Archivos Actualizados

- `README.md` - Documentación completa actualizada
- `requirements.txt` - Agregado: sentencepiece, sacremoses, plotly
- `app_fastapi.py` - Nuevo endpoint full-analysis

### 📝 Notas de Actualización

1. **Requisito de audio**: Mínimo 2-3 segundos para análisis correcto
2. **Primera ejecución**: Descarga ~1-2GB de modelos automáticamente
3. **Python 3.10**: Usar `dashboard.html` en lugar de Streamlit

---

## Versión 1.0.0 - Sistema Original

- Transcripción con Whisper y Vosk
- Diarización con Resemblyzer
- Análisis de sentimiento básico
- Interface Streamlit
