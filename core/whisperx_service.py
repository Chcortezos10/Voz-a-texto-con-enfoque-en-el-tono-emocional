"""
WhisperX Service — Solo transcripción y alineación.
La diarización se maneja por separado en core/pyannote_diarizer.py
"""
import whisperx
import gc
import torch
import logging
import os
import config

logger = logging.getLogger(__name__)


class WhisperXService:
    def __init__(self, device=None, compute_type="int8"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.metadata = None
        self.model_name = config.WHISPER_MODEL

    def load_models(self):
        if self.model is None:
            logger.info(f"Loading WhisperX model ({self.model_name}/{self.compute_type})...")
            self.model = whisperx.load_model(
                self.model_name, self.device, compute_type=self.compute_type
            )

    def process_audio(self, audio_file, batch_size=4):
        """
        Transcribe y alinea el audio. NO hace diarización.
        La diarización se ejecuta por separado con PyannoteDiarizer.
        """
        try:
            self.load_models()

            # 1. Transcribe
            logger.info("WhisperX: Transcribiendo audio...")
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=batch_size)
            logger.info(f"WhisperX: Transcripción completada. Idioma: {result.get('language', 'unknown')}, "
                       f"{len(result.get('segments', []))} segmentos")

            # 2. Align
            logger.info("WhisperX: Alineando segmentos...")
            if self.align_model is None:
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=result["language"], device=self.device
                )

            result = whisperx.align(
                result["segments"], self.align_model, self.metadata,
                audio, self.device, return_char_alignments=False
            )
            logger.info(f"WhisperX: Alineación completada. {len(result.get('segments', []))} segmentos alineados")

            return result

        except Exception as e:
            logger.error(f"Error in WhisperX process: {e}")
            raise e

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        if self.align_model:
            del self.align_model
            self.align_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
