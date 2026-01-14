"""
Sistema de carga lazy de modelos para optimizar RAM
"""
import logging
import torch
from typing import Dict, Any, Optional
import gc

logger = logging.getLogger(__name__)

# Cache global de modelos
_models_cache: Dict[str, Any] = {}
_loading_lock = {}


def clear_model_cache():
    """Limpia el cache de modelos y libera memoria"""
    global _models_cache
    
    for model_name in list(_models_cache.keys()):
        del _models_cache[model_name]
    
    _models_cache = {}
    
    # Limpiar cache de PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()
    
    logger.info("🗑️ Cache de modelos limpiado")


def load_whisper_models(model_size: str = "tiny") -> Dict[str, Any]:
    """
    Carga Whisper con lazy loading y cache.
    
    Args:
        model_size: Tamaño del modelo (tiny, base, small, medium, large)
    
    Returns:
        Dict con modelo Whisper
    """
    global _models_cache
    
    cache_key = f"whisper_{model_size}"
    
    if cache_key in _models_cache:
        logger.info(f"✅ Usando Whisper en cache: {model_size}")
        return _models_cache[cache_key]
    
    try:
        logger.info(f"⏳ Cargando Whisper modelo: {model_size}")
        import whisper
        
        # Configurar para bajo uso de memoria
        device = "cpu"  # Forzar CPU para ahorrar VRAM
        
        model = whisper.load_model(
            model_size,
            device=device,
            download_root=None,
            in_memory=False  # No mantener en memoria si es posible
        )
        
        result = {"whisper": model}
        _models_cache[cache_key] = result
        
        logger.info(f"✅ Whisper cargado: {model_size} en {device}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error cargando Whisper: {e}")
        raise


def get_whisper_model(model_size: str = "tiny"):
    """Helper para obtener modelo Whisper"""
    models = load_whisper_models(model_size)
    return models["whisper"]


def unload_whisper_model():
    """Descarga el modelo Whisper para liberar memoria"""
    global _models_cache
    
    keys_to_remove = [k for k in _models_cache.keys() if k.startswith("whisper_")]
    
    for key in keys_to_remove:
        del _models_cache[key]
        logger.info(f"🗑️ Modelo descargado: {key}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()