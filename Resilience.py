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

    def call(self,func:callable[...,T],*Args,Fallback:Optional[Callable[...,T]] = None,**kwargs)->T:
        #ejecuta la funcion protegida por el circuit breaker
        if not self.is_available:
            if Fallback:
                logger.warning(f"CircuitBreaker[{self.name}]: OPEN -> FALLBACK")
                return Fallback(*Args,**kwargs)
            raise CircuitBreakerError(f"CircuitBreaker [{self.name}]: Abierto")

        try:
            result = func(*args,**kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            if Fallback:
                looger.warning(f"Circuitbreakear [{self.name}]:Error {e}-usando fallback")
                return Fallback(*Args,**kwargs)
            raise

        async def call_async(self,func:Callable[...,T],*args,fallback:Optional[Callable[...,T]] = None,**kwargs)->T:
            #version asincrona de call
            if not self.is_available:
                if fallback:
                    logger.warning(f"CircuitBreaker[{self.name}]: OPEN -> FALLBACK")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args,**kwargs)
                    return fallback(*args,**kwargs)
                raise CircuitBreakerError(f"CircuitBreaker [{self.name}]: Abierto")
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args,**kwargs)
                else:
                    result = func(*args,**kwargs)
                return result
            except Exception as e:
                self.record_failure(e)
                if fallback:
                    looger.warning(f"Circuitbreakear [{self.name}]:Error {e}-usando fallback")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args,**kwargs)
                    return fallback(*args,**kwargs)
                raise

    def get_status(self)->Dict[str,Any]:
        #retorna el estado actual del circuit breaker
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "is_available": self.is_available   
        }

class CircuitBreakerError(Exception):
    #excepcion personalizada para errores del circuit breaker
    pass

@dataclass
class RetryConfig:
    #configuracion para el retry logic
    max_retries:int = 3 
    base_delay:float = 1.0
    max_delay:float = 30.0
    exponential_base:float = 2.0
    jitter:bool = True
    randomization_factor:float = 0.1

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None):

#decorador para reintento con backoff
    def decorator(func:Callable[...,T])->Callable[...,T]:
        @functools.wraps(func)
        def wrapper(*args,**kwargs)->T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args,**kwargs)
                except exceptions as e:
                    last_exception = e 
                    if attempt  == max_retries:
                        logger.error(f"retry agotado para{func.__name__} despues de {max_retries} intentos")
                        raise 

                    #calcular el delay con backoff exponencial
                    delay = min(base_delay *(exponential_base**attempt),max_delay)

                    #anadir jitter(+-20%)
                    import random
                    delay *= (0.8 + random.random() * 0.4)

                    logger.warning( 
                        f"Retry {attempt + 1}/{max_retries} para {func.__name__}: "
                        f"{e}. Esperando {delay:.2f}s")
                    
                    if on_retry:
                        on_retry(attempt+1,e)
                    time.sleep(delay)
                raise last_exception
            return wrapper
        return decorator

def retry_with_backoff_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)):

# version async del decorador de reintentos
    def decorator(func:Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args,**kwargs)->T:
            import random
            last_exception = None 
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args,**kwargs)
                except exceptions as e:
                    last_exception = e 
                    if attempt  == max_retries:
                        logger.error(f"retry agotado para{func.__name__} despues de {max_retries} intentos")
                        raise 

                    #calcular el delay con backoff exponencial
                    delay = min(base_delay *(exponential_base**attempt),max_delay)

                    #anadir jitter(+-20%)
                    import random
                    delay *= (0.8 + random.random() * 0.4)

                    logger.warning( 
                        f"Retry {attempt + 1}/{max_retries} para {func.__name__}: "
                        f"{e}. Esperando {delay:.2f}s")
                    
                    if on_retry:
                        on_retry(attempt+1,e)
                    await asyncio.sleep(delay)
                raise last_exception
            return wrapper
        return decorator    

@dataclass
class FallbackChain:
    #cadena de fallbacks para la degradacion progresiva
    handlers: list = field(default_factory=list)

    def add(self,name:str,handler:callable)->None:
        self.handlers.append((name,handler))
        return self
    
    def execute(self,*args,**kwargs)->Any:
        #ejecutar la cadena hasta que uno tenga exito 
        last_error = None
        for name,handler in self.handlers:
            try:
                result = handler(*args,**kwargs)
                logger.debug(f"Fallback {name} exitoso")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Fallback {name} fallido: {e}")
                continue
        raise RuntimeError(f"Todos los fallbacks fallidos para {last_error}")
    async def execute_async(self,*args,**kwargs)->Any:
        #ejecutar la cadena hasta que uno tenga exito 
        last_error = None
        for name,handler in self.handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args,**kwargs)
                else:
                    result = handler(*args,**kwargs)
                logger.debug(f"Fallback {name} exitoso")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Fallback {name} fallido: {e}")
                continue
        raise RuntimeError(f"Todos los fallbacks fallidos para {last_error}")

class GracefulDegradation:
    #maneja degradacion cuando los serivxios fallan 
    #proporciona valores por defecto o alternativas 

    return {"top_emotion": "neutral",
            "top_score": 0.5,
            "confidence": 0.0,
            "emotions": {"neutral": 1.0},
            "source": "fallback",
            "degraded": True}
    
    def default_translation(text:str)->str:
        return { "text": "",
            "segments": [],
            "duration": 0,
            "degraded": True}
    
    @staticmethod
    def wrap_with_defaults(
        func: Callable[..., T],
        default_value: T,
        log_error: bool = True
    ) -> Callable[..., T]:
    #envuelve una funcion para retornar un valor por defecto si falla
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error al ejecutar {func.__name__}: {e}")
                return default_value
        return wrapper

# Instancias globales de circuit breakers para modelos
WHISPER_BREAKER = CircuitBreaker.get_or_create(
    "whisper",
    CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
)

EMOTION_TEXT_BREAKER = CircuitBreaker.get_or_create(
    "emotion_text",
    CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
)

EMOTION_AUDIO_BREAKER = CircuitBreaker.get_or_create(
    "emotion_audio",
    CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
)

TRANSLATION_BREAKER = CircuitBreaker.get_or_create(
    "translation",
    CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
)

def get_all_breakers_statues()->Dict[str,dict]:
    #retorna el estado de todos los circuit breakers
    return {
        name : breaker.get_status()
        for name, breaker in CircuitBreaker._instances.items()
    }