#configuracion del ensamble de modelos para la mejorar la la calidad de la deteccion de emociones

from dataclasses import dataclass,field
from typing import List,Dict,Optional
from enum import Enum

class ModelPriority(Enum):
    #prioridad de los modelos
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class TextModelConfig:
    #configuracion del modelo de texto
    name:str
    weight:float
    language:str
    priority: ModelPriority = ModelPriority.HIGH
    max_length:int = 512
    use_fp16:bool = True

@dataclass
class AudioModelConfig:
    #configuracion del modelo de audio
    name:str
    weight:float
    language:str
    priority: ModelPriority = ModelPriority.HIGH
    use_fp16:bool = True

@dataclass

class EnsembleConfig:
    #configuracion del ensamble
    text_models:List[TextModelConfig] = field(default_factory=lambda:[
        TextModelConfig(
            name="daveni/twitter-xlm-roberta-emotion-es",
            weight=0.30,
            language="es",
            priority=ModelPriority.CRITICAL
        ),
        TextModelConfig(
            name="j-hartmann/emotion-english-distilroberta-base",
            weight=0.20,
            language="en",
            priority=ModelPriority.HIGH
        ),
        TextModelConfig(
            name="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            weight=0.15,
            language="en",
            priority=ModelPriority.MEDIUM
        ),
    ])
    audio_models:List[AudioModelConfig] = field(default_factory=lambda:[
        AudioModelConfig(
            name="ssuperb/wav2vec2-base-superb-er",
            weight=0.20,
            priority=ModelPriority.HIGH
        ),
        AudioModelConfig(
            name="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            weight=0.15,
            priority=ModelPriority.MEDIUM
        ),
    ])

    #peso del analiss prosodico
    prosodic_weight:float = 0.15

    #optimizaciones
    use_fp16:bool = True
    batch_size:int = 8
    parallel_excecution:bool = True
    max_workers:int = 3

    #cache
    enable_cache:bool = True
    cache_size:int = 1000

    #ajsute de fusion 
    neutral_suppression:float = 0.12
    active_emotion_boost:float = 2.0
    min_confidence_threshold:float = 0.10

    #suavizado temporal
    temporal_smoothing:bool = True
    smoothing_window:int = 3
    smoothing_alpha:float = 0.65

#mapeo extraordinario de emociones incluye 28 categorias 

MAPEO_EMOCIONAL_EXTENDIDO = {
    #EMOCIONES BAS3E
    "joy": "feliz",
    "happiness": "feliz",
    "happy": "feliz",
    "hap": "feliz",
    "alegría": "feliz",
    "excitement": "feliz",
    "amusement": "feliz",
    "love": "feliz",
    "optimism": "feliz",
    "pride": "feliz",
    "relief": "feliz",
    "surprise": "feliz",
    "sorpresa": "feliz",
    "admiration": "feliz",
    "approval": "feliz",
    "caring": "feliz",
    "desire": "feliz",
    "gratitude": "feliz",
    
    "anger": "enojado",
    "angry": "enojado",
    "ang": "enojado",
    "ira": "enojado",
    "disgust": "enojado",
    "disgusto": "enojado",
    "annoyance": "enojado",
    "disapproval": "enojado",
    
    "sadness": "triste",
    "sad": "triste",
    "tristeza": "triste",
    "fear": "triste",
    "fea": "triste",
    "miedo": "triste",
    "grief": "triste",
    "remorse": "triste",
    "disappointment": "triste",
    "embarrassment": "triste",
    "nervousness": "triste",
    
    "neutral": "neutral",
    "neu": "neutral",
    "others": "neutral",
    "otros": "neutral",
    "realization": "neutral",
    "curiosity": "neutral",
    "confusion": "neutral",

}

#configuracion por defecto
DEFAULT_ENSEMBLE_CONFIG = EnsembleConfig()

def get_ensemble_config(present:str="balanceado")-> EnsembleConfig:
    #obtine configuracion segun el perfil

    if present == "speed":
        return EnsembleConfig(
                        text_models=[
                TextModelConfig(
                    name="cardiffnlp/twitter-roberta-base-emotion-multilingual-latest",
                    weight=0.60,
                    language="multi",
                    priority=ModelPriority.CRITICAL
                ),
            ],
            audio_models=[
                AudioModelConfig(
                    name="superb/wav2vec2-base-superb-er",
                    weight=0.25,
                    priority=ModelPriority.CRITICAL
                ),
            ],
            prosody_weight=0.15,
            early_stop_confidence=0.85,
            batch_size=16,
        )
    
    elif present == "presicion":
        return EnsembleConfig(
text_models=[TextModelConfig(
                    name="daveni/twitter-xlm-roberta-emotion-es",
                    weight=0.25,
                    language="es",
                    priority=ModelPriority.CRITICAL
                ),
                TextModelConfig(
                    name="j-hartmann/emotion-english-distilroberta-base",
                    weight=0.20,
                    language="en",
                    priority=ModelPriority.CRITICAL
                ),
                TextModelConfig(
                    name="SamLowe/roberta-base-go_emotions",
                    weight=0.15,
                    language="en",
                    priority=ModelPriority.HIGH
                ),
            ],
            audio_models=[
                AudioModelConfig(
                    name="superb/wav2vec2-base-superb-er",
                    weight=0.15,
                    priority=ModelPriority.CRITICAL
                ),
                AudioModelConfig(
                    name="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    weight=0.10,
                    priority=ModelPriority.HIGH
                ),
            ],
            prosody_weight=0.15,
            early_stop_confidence=None,
            batch_size=4,
        )
    else:
        return DEFAULT_ENSEMBLE_CONFIG