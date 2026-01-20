# 🎭 Voz-a-Texto Emocional

Sistema avanzado de transcripción y análisis emocional de audio con interfaz visual interactiva, desarrollado con FastAPI y modelos de IA.

---

## ✨ Características Principales

| Característica               | Descripción                                            |
| ---------------------------- | ------------------------------------------------------ |
| 🎤 **Transcripción**         | OpenAI Whisper local + Cloud (OpenAI, Groq) en español |
| 😊 **Análisis Emocional**    | 4 categorías: Feliz, Enojado, Triste, Neutral          |
| 🔀 **Análisis Multi-Modal**  | Combina análisis de texto y tono de voz                |
| 👥 **Diarización**           | Identificación automática de múltiples hablantes       |
| 📊 **Dashboard Interactivo** | Métricas, gráficos Timeline y momentos destacados      |
| 📁 **Historial**             | Almacenamiento persistente de análisis anteriores      |
| 📤 **Exportación**           | JSON, CSV, SRT, VTT, TXT                               |
| 🐳 **Docker Ready**          | Despliegue containerizado con soporte GPU NVIDIA       |
| 🛡️ **Resiliencia**           | Circuit Breaker, Retry Logic y Graceful Degradation    |
| ✅ **Validación**            | Validación completa de audio, segmentos y parámetros   |

---

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Windows)

```bash
# Doble clic en:
run_system_v2.bat
```

Esto iniciará el API y abrirá el dashboard automáticamente.

### Opción 2: Manual

````bash
# 1. Activar entorno virtual
.venv\Scripts\activate

# 2. Instalar dependencias (primera vez)
pip install -r requirements.txt

### Opción 3: Docker (Recomendado para Producción)

**Requisitos previos:**
- Docker Desktop instalado y corriendo
- Puerto 8000 disponible

#### 🚀 Inicio Rápido

```bash
# 1. Construir e iniciar el contenedor
docker-compose up --build

# 2. Espera a ver el mensaje: "SISTEMA LISTO: http://127.0.0.1:8000"

# 3. Abre tu navegador en:
http://localhost:8000

# Iniciar en segundo plano (detached)
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Reiniciar el contenedor
docker-compose restart

# Detener el contenedor
docker-compose down

# Detener y eliminar volúmenes (limpieza completa)
docker-compose down -v

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
| **Historial**            | Acceso a análisis anteriores con búsqueda         |
| **Exportación**          | Descarga en múltiples formatos                    |

---

## 📁 Estructura del Proyecto

````

├── core/ # Módulos principales
│ ├── emotion_analysis.py # Análisis emocional multi-modal
│ ├── translation.py # Traducción ES→EN (Helsinki-NLP)
│ ├── audio_processing.py # Procesamiento de audio
│ ├── transcription.py # Transcripción local con Whisper
│ ├── transcription_cloud.py # Transcripción cloud (OpenAI, Groq)
│ ├── diarization.py # Diarización de hablantes
│ ├── model_manager.py # Gestión centralizada de modelos
│ ├── export_manager.py # Exportación a múltiples formatos
│ └── models.py # Carga de modelos Whisper
│
├── routes/ # Rutas API modulares
│ ├── history_routes.py # Historial de análisis
│ ├── export_routes.py # Exportación de datos
│ └── additional_routes.py # Transcripción cloud y sesiones
│
├── app_fastapi.py # API REST unificada (puerto 8000)
├── config.py # Configuración y mapeo de emociones
├── Validators.py # Validación de audio y parámetros
├── Resilience.py # Circuit Breaker y Retry Logic
│
├── dashboard.html # Dashboard web interactivo
├── run_system_v2.bat # Script de inicio (Windows)
├── run_system.sh # Script de inicio (Linux/Mac)
│
├── Dockerfile # Configuración Docker
├── docker-compose.yml # Orquestación de contenedores
├── requirements.txt # Dependencias Python
│
├── data/ # Archivos de datos e historial
├── history/ # Almacenamiento de historial
├── output/ # Archivos de salida
└── pruebas/ # Archivos de prueba

````

---

## 🔌 API Endpoints

### Transcripción y Análisis

| Método | Endpoint                    | Descripción                            |
| ------ | --------------------------- | -------------------------------------- |
| POST   | `/transcribe/full-analysis` | Análisis completo con emociones        |
| POST   | `/transcribe/with-provider` | Transcripción con proveedor específico |
| GET    | `/providers`                | Lista proveedores disponibles          |
| POST   | `/api-key`                  | Configura clave API para cloud         |
| POST   | `/validate-api-key`         | Valida clave API                       |
| GET    | `/estimate-cost`            | Estima costo de transcripción cloud    |

### Historial

| Método | Endpoint             | Descripción                       |
| ------ | -------------------- | --------------------------------- |
| GET    | `/history`           | Obtiene lista de análisis previos |
| GET    | `/history/{item_id}` | Obtiene un análisis específico    |
| POST   | `/history/save`      | Guarda nuevo análisis             |
| DELETE | `/history/{item_id}` | Elimina un análisis               |
| DELETE | `/history/clear`     | Limpia todo el historial          |

### Exportación

| Método | Endpoint          | Descripción                 |
| ------ | ----------------- | --------------------------- |
| POST   | `/export/json`    | Exporta a JSON              |
| POST   | `/export/csv`     | Exporta a CSV               |
| POST   | `/export/srt`     | Exporta subtítulos SRT      |
| POST   | `/export/vtt`     | Exporta subtítulos VTT      |
| POST   | `/export/txt`     | Exporta transcripción TXT   |
| POST   | `/export/summary` | Genera resumen del análisis |

### Sesiones

| Método | Endpoint                | Descripción                |
| ------ | ----------------------- | -------------------------- |
| POST   | `/session/store`        | Almacena nueva sesión      |
| GET    | `/session/{session_id}` | Obtiene sesión por ID      |
| PUT    | `/session/{session_id}` | Actualiza sesión existente |
| DELETE | `/session/{session_id}` | Elimina sesión             |
| GET    | `/sessions`             | Lista todas las sesiones   |
| PUT    | `/segment/update`       | Actualiza segmento         |
| POST   | `/speakers/merge`       | Fusiona hablantes          |

### Sistema

| Método | Endpoint             | Descripción                      |
| ------ | -------------------- | -------------------------------- |
| GET    | `/health`            | Estado básico del servidor       |
| GET    | `/health/detailed`   | Estado detallado con métricas    |
| POST   | `/admin/cleanup`     | Limpieza manual de memoria       |
| GET    | `/admin/model-stats` | Estadísticas de modelos cargados |

### Ejemplo de uso

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/full-analysis" \
  -F "file=@audio.mp3" \
  -F "audio_weight=0.4" \
  -F "lite_mode=false" \
  -F "enable_diarization=true"
````

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

| Modelo                                        | Propósito                          |
| --------------------------------------------- | ---------------------------------- |
| OpenAI Whisper (small)                        | Transcripción de audio en español  |
| Helsinki-NLP/opus-mt-es-en                    | Traducción español → inglés        |
| daveni/twitter-xlm-roberta-emotion-es         | Análisis emocional en español      |
| j-hartmann/emotion-english-distilroberta-base | Análisis emocional en inglés       |
| Resemblyzer VoiceEncoder                      | Embeddings de voz para diarización |

---

## 🔊 Proveedores de Transcripción

| Proveedor  | Descripción                           | Requiere API Key |
| ---------- | ------------------------------------- | ---------------- |
| **local**  | Whisper local (gratuito, usa GPU/CPU) | No               |
| **openai** | OpenAI Whisper API (cloud)            | Sí               |
| **groq**   | Groq API (cloud, rápido)              | Sí               |

---

## 📝 Notas Importantes

1. **Primera ejecución**: Descarga ~1-2GB de modelos automáticamente
2. **Audio mínimo**: 2-3 segundos para análisis correcto
3. **GPU**: Detecta CUDA automáticamente para acelerar procesamiento
4. **Historial**: Se almacena en `data/analysis_history.json` (máximo 100 entradas)
5. **CORS**: Configurado para desarrollo local, ajustar para producción

---

## 📄 Licencia

MIT License
