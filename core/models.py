"""
Sistema de carga lazy de modelos para optimizar RAM
"""
import logging
import torch
from typing import Dict, Any, Optional
import gc
import sys
import os

# Añadir parent al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WHISPER_MODEL

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


def get_optimal_device() -> str:
    """
    Detecta el mejor dispositivo disponible (GPU/CPU).
    
    Returns:
        'cuda' si hay GPU disponible, 'cpu' en caso contrario
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    else:
        logger.info("No se detectó GPU, usando CPU")
        return "cpu"


def get_recommended_model_for_vram() -> str:
    """
    Recomienda el modelo óptimo según la VRAM disponible.
    
    RTX 4050 (~6GB): puede correr hasta medium cómodamente
    """
    if not torch.cuda.is_available():
        return "small"  # CPU: modelo conservador
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb >= 10:
        return "large-v3"  # 10GB+ → large
    elif vram_gb >= 5:
        return "medium"    # 5-10GB → medium (ideal para RTX 4050)
    elif vram_gb >= 2:
        return "small"     # 2-5GB → small
    else:
        return "base"      # <2GB → base


def load_whisper_models(model_size: str = None) -> Dict[str, Any]:
    # Usar config si no se especifica
    if model_size is None:
        model_size = WHISPER_MODEL
    """
    Carga Whisper con lazy loading y cache.
    Usa GPU automáticamente si está disponible.
    
    Args:
        model_size: Tamaño del modelo (tiny, base, small, medium, large, large-v3)
    
    Returns:
        Dict con modelo Whisper
    """
    global _models_cache
    
    cache_key = f"whisper_{model_size}"
    
    if cache_key in _models_cache:
        logger.info(f"Usando Whisper en cache: {model_size}")
        return _models_cache[cache_key]
    
    try:
        logger.info(f"Cargando Whisper modelo: {model_size}")
        import whisper
        
        # Detectar automáticamente el mejor dispositivo
        device = get_optimal_device()
        
        # Info de VRAM si es GPU
        if device == "cuda":
            recommended = get_recommended_model_for_vram()
            logger.info(f"Modelo recomendado para tu GPU: {recommended}")
        
        model = whisper.load_model(
            model_size,
            device=device,
            download_root=None,
            in_memory=True  # Mantener en memoria GPU para velocidad
        )
        
        result = {"whisper": model, "device": device}
        _models_cache[cache_key] = result
        
        logger.info(f"Whisper cargado: {model_size} en {device}")
        return result
        
    except Exception as e:
        logger.error(f"Error cargando Whisper: {e}")
        raise


def get_whisper_model(model_size: str = None):
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