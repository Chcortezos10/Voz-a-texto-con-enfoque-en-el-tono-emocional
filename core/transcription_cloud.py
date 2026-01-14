"""
Transcripción usando OpenAI Whisper Cloud API
"""
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CloudWhisperTranscriber:
    """Cliente para transcripción con OpenAI Whisper Cloud"""
    
    def __init__(self, api_key: str):
        """
        Inicializa el cliente de OpenAI.
        
        Args:
            api_key: API key de OpenAI
        """
        # Import lazy para evitar errores si openai no está instalado
        import httpx
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(300.0, connect=60.0)
        )
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "es",
        prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Transcribe un archivo de audio usando Whisper Cloud.
        
        Args:
            audio_path: Ruta al archivo de audio
            language: Código de idioma (default: "es")
            prompt: Prompt opcional para mejorar transcripción
            temperature: Temperatura para sampling (0-1)
            
        Returns:
            Diccionario con transcripción y segmentos
        """
        try:
            logger.info(f"🌐 Transcribiendo con OpenAI Whisper Cloud: {audio_path}")
            
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    timestamp_granularities=["segment"]
                )
            
            # Convertir a formato estándar
            segments = []
            for seg in response.segments:
                segments.append({
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text']
                })
            
            result = {
                "text": response.text,
                "segments": segments,
                "language": response.language,
                "duration": response.duration
            }
            
            logger.info(f"✅ Transcripción completada: {len(segments)} segmentos, {result['duration']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en transcripción Cloud: {e}")
            raise


def transcribe_audio_cloud(
    audio_path: str,
    api_key: str,
    language: str = "es",
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Función helper para transcribir con Cloud Whisper.
    
    Args:
        audio_path: Ruta al archivo de audio
        api_key: API key de OpenAI
        language: Código de idioma
        prompt: Prompt opcional
        
    Returns:
        Diccionario con transcripción y segmentos
    """
    transcriber = CloudWhisperTranscriber(api_key)
    return transcriber.transcribe(audio_path, language=language, prompt=prompt)


def transcribe_with_timestamps_cloud(
    audio_path: str,
    api_key: str,
    language: str = "es"
) -> List[Dict[str, Any]]:
    """
    Transcribe audio con timestamps detallados usando Cloud Whisper.
    
    Args:
        audio_path: Ruta al archivo de audio
        api_key: API key de OpenAI
        language: Código de idioma
        
    Returns:
        Lista de segmentos con timestamps
    """
    result = transcribe_audio_cloud(audio_path, api_key, language)
    return result.get("segments", [])


def is_cloud_available(api_key: str) -> bool:
    """
    Verifica si la API de OpenAI está disponible.
    
    Args:
        api_key: API key de OpenAI
        
    Returns:
        True si la API está disponible, False en caso contrario
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=10.0)
        # Hacer una prueba simple
        models = client.models.list()
        return True
    except Exception as e:
        logger.warning(f"OpenAI API no disponible: {e}")
        return False


class TranscriptionCache:
    """Cache simple para transcripciones"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Inicializa el cache.
        
        Args:
            cache_dir: Directorio para guardar cache (default: temp)
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "whisper_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, audio_path: str) -> str:
        """
        Genera una clave de cache basada en el archivo.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            Clave de cache (hash del archivo)
        """
        import hashlib
        
        # Hash del contenido del archivo
        hasher = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            # Leer en chunks para archivos grandes
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una transcripción del cache.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            Transcripción cacheada o None
        """
        try:
            import json
            
            cache_key = self.get_cache_key(audio_path)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"📦 Transcripción encontrada en cache")
                return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Error leyendo cache: {e}")
            return None
    
    def set(self, audio_path: str, transcription: Dict[str, Any]):
        """
        Guarda una transcripción en el cache.
        
        Args:
            audio_path: Ruta al archivo de audio
            transcription: Datos de transcripción
        """
        try:
            import json
            
            cache_key = self.get_cache_key(audio_path)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Transcripción guardada en cache")
            
        except Exception as e:
            logger.warning(f"Error guardando cache: {e}")
    
    def clear(self):
        """Limpia todo el cache"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("🗑️ Cache limpiado")
        except Exception as e:
            logger.warning(f"Error limpiando cache: {e}")


# === Funciones de compatibilidad con additional_routes.py ===

from enum import Enum

class TranscriptionProvider(Enum):
    """Proveedores de transcripción disponibles"""
    LOCAL = "local"
    OPENAI = "openai"
    GROQ = "groq"


class TranscriptionService:
    """Servicio de transcripción con soporte para múltiples proveedores"""
    
    def __init__(self):
        self._api_keys = {}
    
    def set_openai_api_key(self, provider: TranscriptionProvider, api_key: str):
        """Configura API key de OpenAI"""
        self._api_keys[TranscriptionProvider.OPENAI] = api_key
    
    def set_groq_api_key(self, provider: TranscriptionProvider, api_key: str):
        """Configura API key de Groq"""
        self._api_keys[TranscriptionProvider.GROQ] = api_key
    
    def has_api_key(self, provider: TranscriptionProvider) -> bool:
        """Verifica si hay API key configurada para el proveedor"""
        return provider in self._api_keys and bool(self._api_keys.get(provider))
    
    def estimate_cost(self, duration_seconds: float, provider: TranscriptionProvider) -> Dict[str, Any]:
        """Estima el costo de transcripción"""
        costs = {
            TranscriptionProvider.LOCAL: 0.0,
            TranscriptionProvider.OPENAI: 0.006,  # $0.006 por minuto
            TranscriptionProvider.GROQ: 0.0001
        }
        
        minutes = duration_seconds / 60
        cost = costs.get(provider, 0.0) * minutes
        
        return {
            "provider": provider.value,
            "duration_seconds": duration_seconds,
            "estimated_cost_usd": round(cost, 4)
        }
    
    async def transcribe(
        self,
        audio_path: str,
        provider: TranscriptionProvider,
        language: str = "es",
        api_key: Optional[str] = None
    ) -> Any:
        """Transcribe audio con el proveedor especificado"""
        from dataclasses import dataclass
        
        @dataclass
        class TranscriptionResult:
            provider: str
            text: str
            segments: list
            language: str
            duration: float
            processing_time: float
            cost_estimate: float
        
        if provider == TranscriptionProvider.LOCAL:
            # Usar Whisper local
            from core.models import load_whisper_models
            import time
            
            start = time.time()
            models = load_whisper_models()
            whisper = models.get("whisper")
            result = whisper.transcribe(audio_path, language=language)
            processing_time = time.time() - start
            
            return TranscriptionResult(
                provider="local",
                text=result.get("text", ""),
                segments=result.get("segments", []),
                language=language,
                duration=result.get("duration", 0),
                processing_time=processing_time,
                cost_estimate=0.0
            )
        
        elif provider == TranscriptionProvider.OPENAI:
            import time
            key = api_key or self._api_keys.get(TranscriptionProvider.OPENAI)
            if not key:
                raise ValueError("No API key for OpenAI")
            
            start = time.time()
            transcriber = CloudWhisperTranscriber(key)
            result = transcriber.transcribe(audio_path, language=language)
            processing_time = time.time() - start
            
            duration = result.get("duration", 0)
            
            return TranscriptionResult(
                provider="openai",
                text=result.get("text", ""),
                segments=result.get("segments", []),
                language=language,
                duration=duration,
                processing_time=processing_time,
                cost_estimate=round((duration / 60) * 0.006, 4)
            )
        
        elif provider == TranscriptionProvider.GROQ:
            raise NotImplementedError("Groq provider not yet implemented")
        
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Singleton instance
_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Obtiene la instancia del servicio de transcripción"""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service


def validate_api_key(provider: str, api_key: str) -> Dict[str, Any]:
    """Valida una API key para un proveedor"""
    if not api_key or len(api_key) < 10:
        return {"valid": False, "message": "API key demasiado corta"}
    
    if provider == "openai":
        if not api_key.startswith("sk-"):
            return {"valid": False, "message": "API key de OpenAI debe comenzar con 'sk-'"}
    elif provider == "groq":
        if not api_key.startswith("gsk_"):
            return {"valid": False, "message": "API key de Groq debe comenzar con 'gsk_'"}
    
    return {"valid": True, "message": "API key válida", "provider": provider}

