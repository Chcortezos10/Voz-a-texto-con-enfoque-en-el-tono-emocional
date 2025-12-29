# Modulo  de resilencia para manejo de errores 
#incluye retry logic, circuit braker y degradacion graceful

import asyncio
import functools
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Callable, Any, Dict, TypeVar

logger = logging.getLogger(__name__)
T= TypeVar('T')

class CircuitState(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    success_threshold: int = 2

@dataclass
class CircuitBreakerState:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

class CircuitBreaker:
    _instance: Dict[str, 'CircuitBreaker'] = {}

    def __init__(self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    @classmethod
    def get_or_create(cls,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> 'CircuitBreaker':
    #obtiene o crea un circuit breaker por nombre (singleton por nombre)
        if name not in cls._instances:
            cls._instances[name] = cls(name, config)
        return cls._instances[name]
    
    @property
    def is_available(self) -> bool:
        #retorna true si el circuit breaker esta abierto
        if self._state.state == CircuitBreakerState.CLOSED:
            return True
        if self._state.state == CircuitBreakerState.HALF_OPEN:
            elapsed = time.time()-self._state.last_failure_time
            if elapsed >= self.config.reset_timeout:
                self._state.state = CircuitState.HALF_OPEN
                self._state.success_count = 0
                logger.info(f"CircuitBreaker [{self.name}]: OPEN -> HALF_OPEN")
                return True
            return False
        return False
    return True

    def record_success(self)->None:
        #registra la llamada existosa
        self._state.last_success_time = time.time()
        self._state.failure_count =0 

        if self._state.state == CircuitBreakerState.HALF_OPEN:
            self._state.success_count += 1
            if self._state.success_count >= self.config.success_threshold:
                self._state.state = CircuitState.CLOSED
                logger.info(f"CircuitBreaker [{self.name}]: HALF_OPEN -> CLOSED (recovered)")

    def record_failure(self)->None:
        #registra la llamada fallida
        self._state.last_failure_time = time.time()
        self._state.failure_count += 1
        if self._state.state == CircuitBreakerState.CLOSED:
            if self._state.failure_count >= self.config.failure_threshold:
                self._state.state = CircuitState.OPEN
                logger.info(f"CircuitBreaker [{self.name}]: CLOSED -> OPEN (failed)")
        