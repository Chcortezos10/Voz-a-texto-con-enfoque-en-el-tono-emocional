# 🎭 Voz-a-Texto Emocional V9

Sistema avanzado de transcripción y análisis emocional de audio con interfaz visual interactiva, desarrollado con FastAPI y modelos de IA. Esta versión incluye capacidades avanzadas de Quality Score, generación de informes consolidados en PDF usando modelos locales (Ollama/Qwen2.5) y un Dashboard rediseñado con Vista General.

---

## ✨ Características Principales

| Característica               | Descripción                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| 🎤 **Transcripción**         | OpenAI Whisper local + Cloud (OpenAI, Groq) en español                               |
| 😊 **Análisis Emocional**    | 4 categorías: Feliz, Enojado, Triste, Neutral                                        |
| 🔀 **Análisis Multi-Modal**  | Combina análisis de texto y tono de voz                                              |
| 👥 **Diarización**           | Identificación automática multihablante Local (MFCC + Clustering) sin límite de lote |
| 📊 **Dashboard Interactivo** | Métricas, gráficos Timeline y Vista General integral de KPIs e Historial             |
| 🌟 **Quality Score**         | Motor de evaluación automatizada de calidad de llamada, con alertas y recomendaciones|
| 📄 **Informes PDF**          | Exportación de reportes consolidados nutridos por LLM local (Ollama / Qwen2.5)       |
| 📁 **Historial**             | Almacenamiento persistente de análisis anteriores con búsqueda avanzada              |
| 📤 **Exportación**           | PDF, JSON, CSV, SRT, VTT, TXT                                                        |
| 🐳 **Docker Ready**          | Despliegue containerizado con soporte GPU NVIDIA y conexión a Ollama en host         |
| 🛡️ **Resiliencia**           | Circuit Breaker, Retry Logic y Graceful Degradation                                  |

---

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Windows)

```bash
# Doble clic en:
run_system_v2.bat
```

Esto iniciará el API y abrirá el dashboard automáticamente.

### Opción 2: Manual

```bash
# 1. Activar entorno virtual
.venv\Scripts\activate

# 2. Instalar dependencias (primera vez)
pip install -r requirements.txt

# 3. Iniciar servidor
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000

# 4. Abrir dashboard en navegador:
# http://localhost:8000
```

---

### Opción 3: Docker (Recomendado para Producción)

**Requisitos previos:**

- Docker Desktop instalado y corriendo
- Puerto 8000 disponible
- (Opcional para Informes PDF) Ollama corriendo en el host en puerto 11434 con el modelo `qwen2.5`.

#### 🚀 Inicio Rápido

```bash
# 1. Construir e iniciar el contenedor
docker-compose up --build

# 2. Espera a ver el mensaje: "SISTEMA LISTO: http://127.0.0.1:8000"

# 3. Abre tu navegador en:
# http://localhost:8000
```

#### 📋 Comandos Docker útiles

```bash
docker-compose up -d         # Iniciar en segundo plano
docker-compose logs -f       # Ver logs en tiempo real
docker-compose restart       # Reiniciar el contenedor
docker-compose down          # Detener el contenedor
docker-compose down -v       # Detener y eliminar volúmenes (limpieza completa)
```

---

## 📊 Dashboard

El dashboard HTML5 incluye:

| Característica            | Descripción                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------ |
| **Vista General**         | Tabla maestra que resume el historial de llamadas, puntaje Quality Score y Alertas   |
| **PDF Consolidado**       | Generación de informes PDF estilizados con feedback generado por LLM                 |
| **Presets**               | Lite (solo texto), Balanceado, Tono (audio)                                          |
| **Slider**                | Control manual del peso audio/texto (0-100%)                                         |
| **Gráfico Timeline**      | Evolución de emociones en el tiempo                                                  |
| **Momentos Destacados**   | Top 3 picos emocionales con texto exacto                                             |
| **Quality Score**         | Análisis semántico de la interacción para obtener una calificación del 0-100%        |
| **Alertas Inteligentes**  | Indicadores de riesgo de pérdida y sentimiento crítico                               |
| **Exportación**           | Descarga en múltiples formatos (incluyendo Reportes PDF)                             |

---

## 📁 Estructura del Proyecto

```
├── core/                     # Módulos principales
│   ├── emotion_analysis.py   # Análisis emocional multi-modal
│   ├── transcription.py      # Transcripción local con Whisper (WhisperX)
│   ├── diarization.py        # Diarización de hablantes y formatos
│   ├── pyannote_diarizer.py  # Módulo local de diarización (MFCC + Clustering)
│   ├── scoring_engine.py     # Motor de cálculo de Quality Score
│   ├── alert_system.py       # Sistema de alertas de sentimiento
│   ├── export_manager.py     # Exportación a múltiples formatos (incluye PDF base)
│   └── call_summary.py       # Generación de resúmenes consolidados
│
├── routes/                   # Rutas API modulares
│   ├── history_routes.py     # Historial de análisis
│   ├── export_routes.py      # Exportación de datos y PDF
│   ├── scoring_routes.py     # Endpoints de Quality Score
│   ├── alert_routes.py       # Endpoints de Alertas
│   └── kpi_routes.py         # Endpoints de Reportes Generales
│
├── app_fastapi.py            # API REST unificada (puerto 8000)
├── config.py                 # Configuración general y de memoria
├── Validators.py             # Validación de audio y parámetros
├── Resilience.py             # Circuit Breaker y Retry Logic
│
├── dashboard.html            # Dashboard web interactivo
├── docker-compose.yml        # Orquestación de contenedores (v9.0.0, soporte Ollama host)
├── requirements.txt          # Dependencias Python
└── ...                       # Carpetas de datos (history, output, feedback)
```

---

## 🔌 API Endpoints

### Transcripción y Análisis
| Método | Endpoint                    | Descripción                            |
| ------ | --------------------------- | -------------------------------------- |
| POST   | `/transcribe/full-analysis` | Análisis completo con WhispeX y emociones |
| POST   | `/transcribe/with-provider` | Transcripción con proveedor específico |

### Quality Score y KPIs (Nuevos)
| Método | Endpoint               | Descripción                            |
| ------ | ---------------------- | -------------------------------------- |
| GET    | `/kpis/summary`        | Obtiene KPIs globales del sistema      |
| POST   | `/scoring/calculate`   | Calcula Quality Score de una sesión    |
| GET    | `/alert/active`        | Analiza alertas críticas activas       |

### Exportación y Reportes
| Método | Endpoint          | Descripción                                |
| ------ | ----------------- | ------------------------------------------ |
| POST   | `/export/pdf`     | Exporta reporte maestro a PDF con insights |
| POST   | `/export/json`    | Exporta a JSON                             |
| POST   | `/config/logo`    | Sube un logo de empresa para el PDF        |
| GET    | `/config/logo`    | Verifica estado del logo para reportes     |

### Historial
| Método | Endpoint             | Descripción                       |
| ------ | -------------------- | --------------------------------- |
| GET    | `/history`           | Obtiene lista de análisis previos |
| POST   | `/history/save`      | Guarda nuevo análisis             |
| DELETE | `/history/clear`     | Limpia todo el historial          |

---

## ⚙️ Configuración de Emociones

Las emociones se simplifican a 4 categorías en `config.py`:
- 😊 **feliz**: alegría, sorpresa, positividad
- 😠 **enojado**: ira, disgusto, rechazo
- 😢 **triste**: tristeza, miedo, vulnerabilidad
- 😐 **neutral**: neutral, otros

---

## 🐳 Docker y Hardware

El proyecto incluye soporte completo para Docker con:

- **GPU NVIDIA**: Habilitado por defecto (`runtime: nvidia`).
- **Ollama LLM (Qwen2.5)**: Conectividad configurable usando `host.docker.internal:11434` para la generación local y privada de insights en PDF sin depender de APIs en nube.
- **Volúmenes persistentes**: Caché optimizada para Whisper, Torch y HuggingFace.

---

## 📋 Modelos Utilizados

| Modelo                                        | Propósito                                            |
| --------------------------------------------- | ---------------------------------------------------- |
| OpenAI Whisper (small/medium)                 | Transcripción de audio ultra-rápida (integración WhisperX) |
| daveni/twitter-xlm-roberta-emotion-es         | Análisis emocional en español                        |
| j-hartmann/emotion-english-distilroberta-base | Análisis emocional en inglés                         |
| Modelo Local (MFCC + Clustering)              | Diarización de código abierto sin límite de lote     |
| **Ollama / Qwen2.5** (Recomendado/Host)       | Resúmenes en lenguaje natural y validación de scores |

---

## 📝 Notas Importantes

1. **Memoria (RAM/VRAM)**: Se ha optimizado la gestión de memoria y se introdujo un Endpoint `/admin/cleanup-memory` para liberación agresiva en GPUs moderadas (< 8GB VRAM).
2. **Audio mínimo**: 2-3 segundos para análisis correcto.
3. **GPU**: Detecta CUDA automáticamente para acelerar procesamiento.
4. **Historial**: Se almacena en `data/analysis_history.json`. La *Vista General* permite explorarlo fácilmente.

---

## 📄 Licencia

MIT License
