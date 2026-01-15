"""
Gestor de modelos para la API Unificada Voz-a-Texto Emocional
"""
import gc
import torch
from typing import Optional, Dict, Any
from threading import Lock

class ModelManager:
    #gestor centralizado de modelos de carga y descarga
    def __init__(self):
        self._models:Dict[str,Any] = {}
        self._usage_count:Dict[str,Any] = {}
        self.lock = Lock()
        self.max_models_loaded = max_models_loaded
        logger.info(f"Max models loaded: {self.max_models_loaded}")
    
    def get_model(self, model_name:str) -> Any:
        with self.lock:
            if model_name not in self._models:
                self._usage_count[model_name] += 1
                return self._models[model_name]

            #si se llega al limite descarga el modelo menos util
            if len(self._models) >= self.max_models_loaded:
                self._unload_least_used()

            #carga de modelos
            logger.info(f"Cargando modelo: {model_name}")
            try:
                self._models[model_name] = loader_func()
                self._usage_count[model_name] = 1
                import time
                self._last_access[model_name] = time.time()
                logger.info(f"✔ Modelo {model_name} cargado")
                return self._models[model_name]
            except Exception as e:
                logger.error(f"Error al cargar modelo {model_name}: {e}")
                raise
    
    def _unload_least_used(self):
        """Descarga el modelo con menor uso"""
        if not self._models:
            return
            
        # Encontrar el modelo con menor uso
        least_used_model = min(self._usage_count, key=self._usage_count.get)

        logger.info(f"Descargando modelo: {least_used_model}")
        del self._models[least_used_model]
        del self._usage_count[least_used_model]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self, model_name:str):
        """Descarga todos los modelos"""
        with self.lock:
            if model_name in self._models:
                logger.info(f"Descargando modelo: {model_name}")
                del self._models[model_name]
                del self._usage_count[model_name]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def clear_all(self):
        """Descarga todos los modelos"""
        with self.lock:
            self._models.clear()
            self._usage_count.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("MEMORIA LIMPIADA")

    def get_stats(self):
        with self.lock:
            return {
                "models_loaded": len(self._models),
                "max_models_loaded": self.max_models_loaded,
                "usage_count": self._usage_count,
                "last_access": self._last_access
            }

_model_manager: Optional[ModelManager] = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(max_models_loaded=MAX_MODELS_LOADED)
    return _model_manager

def cleanup_all_models():
    global _model_manager
    if _model_manager:
        _model_manager.clear_all()