# este modulo se encargara de validar la entrada del audio, formatos y parametros antes del procesamiento

import logging
import os
import tempfile
import wave 
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

@dataclass
class AudioValidationResult:
    #resultado de la validacion
    is_valid: bool
    duration_sec: float
    sample_rate: int
    channels: int
    format_info: str
    warnings: list
    errors: list

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0 
    
class AudioValidator:

    MIN_DURATION_SEC = 0.5      # Mínimo medio segundo
    MAX_DURATION_SEC = 600      # Máximo 10 minutos
    MIN_SAMPLE_RATE = 8000      # Mínimo 8kHz
    MAX_SAMPLE_RATE = 48000     # Máximo 48kHz
    TARGET_SAMPLE_RATE = 16000  # Objetivo para modelos
    MIN_RMS_THRESHOLD = 0.001   # Umbral mínimo de energía (detectar silencio)
    MAX_FILE_SIZE_MB = 100      # Máximo 100MB
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.flac', '.webm'}

    def __init__(self,
        min_duration: float = MIN_DURATION_SEC,
        max_duration: float = MAX_DURATION_SEC,
        max_file_size_mb: float = MAX_FILE_SIZE_MB):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_file_size_mb = max_file_size_mb
    
    def validate_audio(self, file_path: Union[str, Path]) -> AudioValidationResult:
        # valdia que un archivo desde su ruta
        path = Path(file_path)
        warnings = []
        errors = []
        
        #1._ verifiacion de la existencia del archivo
        if not path.exists():
            return AudioValidationResult(is_valid=False,
                duration_sec=0, sample_rate=0, channels=0,
                format_info="", warnings=[], 
                errors=[f"Archivo no encontrado: {path}"])
        #2._ verificacion extension 

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            errors.append(f"Formato no soportado: {ext}. Usar: {self.SUPPORTED_FORMATS}" )
        
        #3._ verificacion del tamaño del archivo
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            errors.append(f"Archivo muy grande: {file_size_mb:.1f}MB (máx: {self.max_file_size_mb}MB)")
        elif file_size_mb > self.max_file_size_mb * 0.8:
            warnings.append(f"Archivo grande: {file_size_mb:.1f}MB - puede ser lento")

        if errors:
            return AudioValidationResult(is_valid=False,
                duration_sec=0, sample_rate=0, channels=0,
                format_info=ext, warnings=warnings, errors=errors)
        
        #4._cargar y validar el audio

        try:
            return self._validate_audio_content(str(path),warnings,errors)
        except Exception as e:
            errors.append(f"Error al cargar el audio: {str(e)}")
            return AudioValidationResult(is_valid=False,
                duration_sec=0, sample_rate=0, channels=0,
                format_info=ext, warnings=warnings, errors=errors)
        
    def _validate_audio_content(self, file_path: str, warnings: list, errors: list) -> AudioValidationResult:
        #leer metadata con soundfile
        info = sf.info(file_path)
        duration = info.duration
        sample_rate = info.samplerate
        channels = info.channels
        format_info = f"{info.format} / {info.subtype}"
        #validar duracion
        if duration < self.min_duration:
            errors.append(f"Audio muy corto: {duration:.2f}s (mín: {self.min_duration}s)")
        elif duration > self.max_duration:
            errors.append(f"Audio muy largo: {duration:.1f}s (máx: {self.max_duration}s)")
        elif duration > self.max_duration * 0.8:
            warnings.append(f"Audio largo: {duration:.1f}s - procesamiento lento")

        #validar sample rate
        if sample_rate < self.MIN_SAMPLE_RATE:
            errors.append(f"Sample rate muy bajo: {sample_rate}Hz (mín: {self.MIN_SAMPLE_RATE}Hz)")
        elif sample_rate != self.TARGET_SAMPLE_RATE:
            warnings.append(f"Sample rate {sample_rate}Hz será convertido a {self.TARGET_SAMPLE_RATE}Hz")           

        #validar canales (advertencia, no error - librosa convierte automáticamente)
        if channels != 1:
            warnings.append(f"Audio estéreo ({channels} canales) será convertido a mono")
        
        #8._ validar energia 
        if not errors:
            try:
                data,_ = sf.read(file_path, dtype='float32')
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                
                rms = np.sqrt(np.mean(data ** 2))
                if rms < self.MIN_RMS_THRESHOLD:
                    warnings.append(f"Audio con muy poca energía (RMS={rms:.4f}) - posible silencio")
                
                # Detectar clipping
                peak = np.max(np.abs(data))
                if peak > 0.99:
                    warnings.append("Audio posiblemente clippeado (peak > 0.99)")
                    
            except Exception as e:
                warnings.append(f"No se pudo analizar energía: {e}")
        
        is_valid = len(errors) == 0
        return AudioValidationResult(is_valid=is_valid,
            duration_sec=duration, sample_rate=sample_rate, channels=channels,
            format_info=format_info, warnings=warnings, errors=errors)

class SegmentValidator:
    MIN_SEGMENT_DURATION = 0.1   # 100ms mínimo
    MAX_SEGMENT_DURATION = 30.0  # 30s máximo
    MIN_TEXT_LENGTH = 1  # mínimo de caracteres

    @staticmethod
    def validate_segment(start: float,
        end: float,
        text: str,
        audio_duration: float
        ) -> Tuple[bool, list]:
        issues = []
        
        if start < 0:
            issues.append(f"start negativo: {start}")
        if end <= start:
            issues.append(f"end ({end}) <= start ({start})")
        if end > audio_duration + 0.5:  # 0.5s de tolerancia
            issues.append(f"end ({end}) > duracion audio ({audio_duration})")

        #validar duracion 
        duration = end - start
        if duration < SegmentValidator.MIN_SEGMENT_DURATION:
            issues.append(f"Segmento muy corto: {duration:.3f}s")
        if duration > SegmentValidator.MAX_SEGMENT_DURATION:
            issues.append(f"Segmento muy largo: {duration:.1f}s")
        
        # Validar texto
        if not text or len(text.strip()) < SegmentValidator.MIN_TEXT_LENGTH:
            issues.append("Texto vacío o muy corto")
        
        return len(issues) == 0, issues

    @staticmethod
    def sanitize_text(start: float,
        end: float,
        text: str,
        audio_duration: float
    ) -> Tuple[float, float, str]:

    #corregir tiempos
        start_fixed = max(0.0, start)
        end_fixed = min(audio_duration, max(end, start_fixed + 0.1))
        # Asegurar que end > start
        if end_fixed <= start_fixed:
            end_fixed = start_fixed + 0.1
        
        # Limpiar texto
        text_fixed = text.strip() if text else ""
        
        return start_fixed, end_fixed, text_fixed

class ParametersValidator:
    #validador de parametros de api
    @staticmethod
    def validate_audio_weight(audio_weight: float) -> Tuple[float, Optional[str]]:
        warning = None
        if audio_weight < 0:
            audio_weight = 0.0
            warning = "audio_weight negativo, ajustado a 0.0"
        elif audio_weight > 1:
            audio_weight = 1.0
            warning = "audio_weight > 1, ajustado a 1.0"
        
        return audio_weight, warning
    @staticmethod
    def validate_request_params(lite_mode: bool,
        audio_weight: float
    ) -> Tuple[float, list, bool]:
        warnings = []
        audio_weight, weight_warning = ParametersValidator.validate_audio_weight(audio_weight)
        if weight_warning:
            warnings.append(weight_warning)

        # Coherencia: lite_mode implica audio_weight = 0
        if lite_mode and audio_weight > 0:
            audio_weight = 0.0
            warnings.append("lite_mode implica audio_weight = 0, ajustado a 0.0")
        
        return audio_weight, warnings, lite_mode
    
    def validate_audio_file(file_path: Union[str, Path]) -> AudioValidationResult:
        #funcion de conveniencia para validar un archivo de audio.
        validator = AudioValidator()
        return validator.validate_audio_file(file_path)
    
    def is_audio_valid_for_processing(file_path: Union[str, Path]) -> Tuple[bool, str]:
        result = validate_audio_file(file_path)
        if result.is_valid:
            msg = f"audio valido: {result.duration_sec:.1f}s, {result.sample_rate}hz"
            if result.warnings:
                msg += f" - advertencias: {len(result.warnings)}"
            return True, msg
        else:
            return False, f"audio invalido: {result.errors}"

def validate_audio_energy(
    audio: np.ndarray, 
    sr: int, 
    min_rms: float = 0.001
) -> tuple:
    # Validacion rapida de energia parecida a lo que se hace dentro de AudioValidator
    rms = np.sqrt(np.mean(audio**2))
    if rms < min_rms:
         return False, f"Energía muy baja (RMS={rms:.4f})"
    
    # Check clipping
    if np.max(np.abs(audio)) > 0.99:
        return True, "Posible clipping detectado"
        
    return True, None