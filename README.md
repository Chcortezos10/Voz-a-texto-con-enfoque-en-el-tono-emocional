# 🎭 Voz-a-Texto Emocional V5

Sistema avanzado de transcripción y análisis emocional de audio con interfaz visual interactiva.

## ✨ Características Principales

- 🎤 **Transcripción** - OpenAI Whisper en español
- � **Análisis Emocional Simplificado** - 4 categorías: 😊 Feliz, 😠 Enojado, 😢 Triste, 😐 Neutral
- � **Análisis Multi-Modal** - Combina texto y tono de voz
- � **Dashboard Interactivo** - Métricas, gráficos y momentos destacados
- ⚡ **Optimizado para RAM** - Carga de modelos bajo demanda

---

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)

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

---

## 📊 Dashboard V5

El dashboard incluye:

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
├── core/                      # Módulos principales
│   ├── emotion_analysis.py    # Análisis emocional (4 categorías)
│   ├── translation.py         # Traducción ES→EN
│   ├── audio_processing.py    # Procesamiento de audio
│   └── models.py              # Carga de Whisper
│
├── app_fastapi.py             # API REST unificada (puerto 8000)
├── config.py                  # Configuración y mapeo de emociones
├── dashboard.html             # Dashboard V5 interactivo
├── run_system_v2.bat          # Script de inicio
└── requirements.txt           # Dependencias
```

---

## 🔌 API Endpoint

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

---

## ⚙️ Configuración de Emociones

Las emociones se simplifican a 4 categorías en `config.py`:

| Salida         | Emociones Incluidas             |
| -------------- | ------------------------------- |
| � **feliz**    | alegría, sorpresa, positividad  |
| 😠 **enojado** | ira, disgusto, rechazo          |
| 😢 **triste**  | tristeza, miedo, vulnerabilidad |
| 😐 **neutral** | neutral, otros                  |

---

## 🛠️ Requisitos

- Python 3.10+
- 4GB RAM mínimo (8GB recomendado)
- Windows 10/11

---

## 📄 Licencia

MIT License
