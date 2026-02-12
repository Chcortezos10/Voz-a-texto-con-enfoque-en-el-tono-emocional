import whisperx
import gc
import torch
import logging
import os

logger = logging.getLogger(__name__)

class WhisperXService:
    def __init__(self, device="cuda", compute_type="int8"):
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.align_model = None
        self.metadata = None
        self.diarize_model = None
        self.hf_token = os.getenv("HF_TOKEN")

    def load_models(self):
        if self.model is None:
            logger.info("Loading WhisperX model (medium/int8)...")
            self.model = whisperx.load_model("medium", self.device, compute_type=self.compute_type)

        if self.diarize_model is None and self.hf_token:
            logger.info("Loading Diarization pipeline...")
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

    def process_audio(self, audio_file, batch_size=16):
        self.load_models()
        
        # 1. Transcribe
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=batch_size)
        
        # 2. Align
        if self.align_model is None:
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
        
        result = whisperx.align(
            result["segments"], self.align_model, self.metadata, audio, self.device, return_char_alignments=False
        )

        # 3. Diarize
        if self.diarize_model:
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return result
        
    def cleanup(self):
        del self.model
        del self.align_model
        del self.diarize_model
        gc.collect()
        torch.cuda.empty_cache()
