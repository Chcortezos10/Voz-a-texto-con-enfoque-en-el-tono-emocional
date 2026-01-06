# 🎭 Voz-a-Texto Emocional

Sistema avanzado de transcripción y análisis emocional de audio con interfaz visual interactiva, desarrollado con FastAPI y modelos de IA.

---

## ✨ Características Principales

| Característica               | Descripción                                          |
| ---------------------------- | ---------------------------------------------------- |
| 🎤 **Transcripción**         | OpenAI Whisper en español con soporte GPU            |
| 😊 **Análisis Emocional**    | 4 categorías: Feliz, Enojado, Triste, Neutral        |
| 🔀 **Análisis Multi-Modal**  | Combina análisis de texto y tono de voz              |
| 📊 **Dashboard Interactivo** | Métricas, gráficos Timeline y momentos destacados    |
| 🐳 **Docker Ready**          | Despliegue containerizado con soporte GPU NVIDIA     |
| 🛡️ **Resiliencia**           | Circuit Breaker, Retry Logic y Graceful Degradation  |
| ✅ **Validación**            | Validación completa de audio, segmentos y parámetros |

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
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
```

Luego abre `dashboard.html` en tu navegador.

### Opción 3: Docker

```bash
# Construir e iniciar
docker-compose up --build

# Iniciar en segundo plano
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

---

## 📊 Dashboard

El dashboard HTML5 incluye:

| Característica           | Descripción                                       |
| ------------------------ | ------------------------------------------------- |
| **Presets**              | Lite (solo texto), Balanceado, Tono (audio)       |
| **Slider**               | Control manual del peso audio/texto (0-100%)      |
| **Gráfico Timeline**     | Evolución de emociones en el tiempo               |
| **Gráfico Distribución** | Pie chart con % de cada emoción                   |
| **Momentos Destacados**  | Top 3 picos emocionales con texto exacto          |
| **Métricas**             | Emoción dominante, intensidad, cambios de emoción |

---

## 📁 Estructura del Proyecto

```
├── core/                          # Módulos principales
│   ├── emotion_analysis.py        # Análisis emocional multi-modal
│   ├── translation.py             # Traducción ES→EN (Helsinki-NLP)
│   ├── audio_processing.py        # Procesamiento de audio
│   ├── transcription.py           # Transcripción con Whisper
│   ├── diarization.py             # Diarización de hablantes
│   └── models.py                  # Carga de modelos Whisper
│
├── app_fastapi.py                 # API REST unificada (puerto 8000)
├── config.py                      # Configuración y mapeo de emociones
├── Validators.py                  # Validación de audio y parámetros
├── Resilience.py                  # Circuit Breaker y Retry Logic
│
├── dashboard.html                 # Dashboard web interactivo
├── run_system_v2.bat              # Script de inicio (Windows)
├── run_system.sh                  # Script de inicio (Linux/Mac)
│
├── Dockerfile                     # Configuración Docker
├── docker-compose.yml             # Orquestación de contenedores
├── requirements.txt               # Dependencias Python
│
├── model/                         # Modelos descargados
├── data/                          # Archivos de datos
└── output/                        # Archivos de salida
```

---

## 🔌 API Endpoints

### Análisis Completo

```bash
POST /transcribe/full-analysis

# Parámetros:
# - file: archivo de audio (mp3, wav, m4a)
# - lite_mode: true/false (solo texto si true)
# - audio_weight: 0.0-1.0 (peso del análisis de tono)
```

**Ejemplo:**

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/full-analysis" \
  -F "file=@audio.mp3" \
  -F "audio_weight=0.4" \
  -F "lite_mode=false"
```

### Health Check

```bash
GET /health           # Estado básico
GET /health/detailed  # Estado detallado con métricas
```

---

## ⚙️ Configuración de Emociones

Las emociones se simplifican a 4 categorías en `config.py`:

| Salida         | Emociones Incluidas             |
| -------------- | ------------------------------- |
| 😊 **feliz**   | alegría, sorpresa, positividad  |
| 😠 **enojado** | ira, disgusto, rechazo          |
| 😢 **triste**  | tristeza, miedo, vulnerabilidad |
| 😐 **neutral** | neutral, otros                  |

---

## 🛡️ Módulos de Resiliencia

### Circuit Breaker

Protege contra fallos en cascada con estados: CLOSED, OPEN, HALF_OPEN.

### Retry with Backoff

Reintentos automáticos con delay exponencial y jitter.

### Graceful Degradation

Valores por defecto cuando fallan servicios externos.

### Fallback Chain

Cadena de handlers alternativos para operaciones críticas.

---

## ✅ Validación

| Validador             | Función                                           |
| --------------------- | ------------------------------------------------- |
| `AudioValidator`      | Valida formato, duración, sample rate y contenido |
| `SegmentValidator`    | Valida segmentos de transcripción                 |
| `ParametersValidator` | Valida parámetros de API                          |

---

## 🛠️ Requisitos

| Requisito | Especificación                                    |
| --------- | ------------------------------------------------- |
| Python    | 3.10+                                             |
| RAM       | 4GB mínimo (8GB recomendado)                      |
| GPU       | NVIDIA con CUDA (opcional, acelera procesamiento) |
| SO        | Windows 10/11, Linux, macOS                       |

---

## 🐳 Docker

El proyecto incluye soporte completo para Docker con:

- **GPU NVIDIA**: Habilitado por defecto (comentar si no hay GPU)
- **Volúmenes persistentes**: Cache de modelos Whisper y HuggingFace
- **Health checks**: Monitoreo automático del servicio
- **Auto-restart**: Reinicio automático en caso de fallo

---

## 📋 Modelos Utilizados

| Modelo                                        | Propósito                         |
| --------------------------------------------- | --------------------------------- |
| OpenAI Whisper (small)                        | Transcripción de audio en español |
| Helsinki-NLP/opus-mt-es-en                    | Traducción español → inglés       |
| daveni/twitter-xlm-roberta-emotion-es         | Análisis emocional en español     |
| j-hartmann/emotion-english-distilroberta-base | Análisis emocional en inglés      |

---

## 📝 Notas Importantes

1. **Primera ejecución**: Descarga ~1-2GB de modelos automáticamente
2. **Audio mínimo**: 2-3 segundos para análisis correcto
3. **GPU**: Detecta CUDA automáticamente para acelerar procesamiento

---

## 📄 Licencia

MIT License
