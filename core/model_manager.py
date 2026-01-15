"""
Gestor inteligente de modelos para optimizar el uso de memoria.
Implementa carga bajo demanda y limpieza automática de modelos.
"""
import gc
import logging
import threading
from typing import Dict, Any, Callable, Optional
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestor de modelos con límite de memoria."""
    
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()
    
    def __init__(self, max_models: int = 2):
        self.max_models = max_models
        self.models: OrderedDict[str, Any] = OrderedDict()
        self._model_lock = threading.Lock()
        self._load_count = 0
        self._eviction_count = 0
    
    def get_model(self, name: str, loader: Callable[[], Any]) -> Any:
        """Obtiene o carga un modelo."""
        with self._model_lock:
            if name in self.models:
                self.models.move_to_end(name)
                return self.models[name]
            
            while len(self.models) >= self.max_models:
                old_name, old_model = self.models.popitem(last=False)
                logger.info(f"Evicting model: {old_name}")
                self._cleanup_model(old_model)
                self._eviction_count += 1
            
            logger.info(f"Loading model: {name}")
            model = loader()
            self.models[name] = model
            self._load_count += 1
            
            return model
    
    def _cleanup_model(self, model: Any) -> None:
        """Limpia memoria de un modelo."""
        try:
            if hasattr(model, 'model'):
                del model.model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error cleaning model: {e}")
    
    def cleanup_all(self) -> None:
        """Limpia todos los modelos."""
        with self._model_lock:
            for name, model in list(self.models.items()):
                self._cleanup_model(model)
            self.models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("All models cleaned")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del manager."""
        with self._model_lock:
            return {
                "loaded_models": list(self.models.keys()),
                "model_count": len(self.models),
                "max_models": self.max_models,
                "load_count": self._load_count,
                "eviction_count": self._eviction_count
            }


_model_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager(max_models: int = 2, max_models_loaded: int = None) -> ModelManager:
    """Obtiene o crea instancia del ModelManager."""
    global _model_manager
    
    effective_max = max_models_loaded if max_models_loaded is not None else max_models
    
    with _manager_lock:
        if _model_manager is None:
            _model_manager = ModelManager(max_models=effective_max)
        return _model_manager


def cleanup_all_models() -> None:
    """Limpia todos los modelos cargados."""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup_all()


def reset_model_manager() -> None:
    """Resetea el manager."""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup_all()
    _model_manager = None
