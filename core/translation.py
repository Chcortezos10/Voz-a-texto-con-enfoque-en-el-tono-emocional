"""
Módulo de traducción Español -> Inglés.
Utiliza Helsinki-NLP/opus-mt-es-en para traducción de alta calidad.
"""
import logging
from functools import lru_cache
from typing import Optional, List
import torch
from transformers import MarianMTModel, MarianTokenizer

from config import TRANSLATION_MODEL, DEVICE

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Determina el dispositivo a usar (GPU/CPU)."""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


@lru_cache(maxsize=1)
def load_translation_model():
    """
    Carga el modelo de traducción ES->EN con caché.
    
    Returns:
        Tuple de (modelo, tokenizer)
    """
    device = get_device()
    logger.info(f"Cargando modelo de traducción en {device}...")
    
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)
    
    if device == "cuda":
        model = model.to(device)
    
    logger.info("Modelo de traducción cargado")
    return model, tokenizer


def translate_es_to_en(text: str, max_length: int = 512) -> str:
    """
    Traduce texto de español a inglés.
    
    Args:
        text: Texto en español a traducir
        max_length: Longitud máxima de tokens
        
    Returns:
        Texto traducido al inglés
    """
    if not text or not text.strip():
        return ""
    
    try:
        model, tokenizer = load_translation_model()
        device = get_device()
        
        # Tokenizar
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generar traducción
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decodificar
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation
    
    except Exception as e:
        logger.error(f"Error en traducción: {e}")
        return text  # Retornar texto original si falla


def translate_segments(segments: List[dict]) -> List[dict]:
    """
    Traduce una lista de segmentos con texto en español.
    
    Args:
        segments: Lista de diccionarios con campo 'text'
        
    Returns:
        Lista de segmentos con campo 'text_en' añadido
    """
    for seg in segments:
        text = seg.get("text", "")
        if text:
            seg["text_en"] = translate_es_to_en(text)
        else:
            seg["text_en"] = ""
    return segments


def translate_batch(texts: List[str], max_length: int = 512) -> List[str]:
    """
    Traduce un lote de textos de forma eficiente.
    
    Args:
        texts: Lista de textos en español
        max_length: Longitud máxima de tokens
        
    Returns:
        Lista de textos traducidos
    """
    if not texts:
        return []
    
    try:
        model, tokenizer = load_translation_model()
        device = get_device()
        
        # Filtrar textos vacíos
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        if not non_empty_texts:
            return ["" for _ in texts]
        
        # Tokenizar batch
        inputs = tokenizer(
            non_empty_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generar traducciones
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decodificar
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Reconstruir lista completa
        result = ["" for _ in texts]
        for idx, trans in zip(non_empty_indices, translations):
            result[idx] = trans
        
        return result
    
    except Exception as e:
        logger.error(f"Error en traducción batch: {e}")
        return texts  # Retornar textos originales si falla
