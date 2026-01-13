import asyncio
import os
import io
import logging
from unittest.mock import MagicMock, AsyncMock
from fastapi import UploadFile

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFICATION")

# Mock dependencies if strictly needed (e.g. if CUDA fails)
# For now we assume the environment is capable or we will catch exceptions.

async def verify_validators():
    logger.info("--- Testing Validators ---")
    from Validators import validate_audio_energy
    import numpy as np
    
    # Test silence
    silence = np.zeros(16000)
    is_valid, msg = validate_audio_energy(silence, 16000)
    if is_valid is False and "baja" in msg:
        logger.info("✔ validate_audio_energy detected silence correctly.")
    else:
        logger.error(f"✘ validate_audio_energy failed silence detection: {is_valid}, {msg}")

async def verify_resilience():
    logger.info("--- Testing Resilience ---")
    from Resilience import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
    
    # Create a test breaker
    breaker = CircuitBreaker("test_breaker", CircuitBreakerConfig(failure_threshold=2, reset_timeout=1))
    
    # function that fails
    def fail_func():
        raise ValueError("Intentional Failure")
    
    # Fail 2 times
    try: breaker.call(fail_func)
    except: pass
    try: breaker.call(fail_func)
    except: pass
    
    # Should be open now
    if not breaker.is_available:
         logger.info("✔ CircuitBreaker opened after threshold.")
    else:
         logger.error("✘ CircuitBreaker failed to open.")
         
    # Fallback test
    res = breaker.call(fail_func, fallback=lambda: "fallback_ok")
    if res == "fallback_ok":
        logger.info("✔ CircuitBreaker fallback works.")
    else:
        logger.error(f"✘ CircuitBreaker fallback failed: {res}")

async def verify_api_integration():
    logger.info("--- Testing API Integration ---")
    from app_fastapi import transcribe_full_analysis, app
    
    file_path = "pruebas/test#2.mp3"
    if not os.path.exists(file_path):
        import shutil
        # If test#2.mp3 doesn't exist, try to find any small file or create dummy
        # But user said @[pruebas] is there.
        # Let's list absolute path
        abs_path = os.path.abspath(os.path.join(os.getcwd(), "pruebas", "test#2.mp3"))
        if not os.path.exists(abs_path):
             logger.error(f"Test file not found at {abs_path}. Skipping API test.")
             return

    # Mock UploadFile
    # We need to read the real file into a BytesIO or SpooledTemporaryFile to simulate UploadFile
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    file_obj = io.BytesIO(file_content)
    upload_file = UploadFile(filename="test#2.mp3", file=file_obj)
    
    # Call the endpoint directly
    # Note: We need to pass args that match the signature
    # file: UploadFile, lite_mode: bool, audio_weight: float, enable_diarization: bool, num_speakers: Optional[int]
    
    try:
        logger.info("Calling transcribe_full_analysis (this may take time)...")
        # We set lite_mode=True to speed it up and avoid OOM if env is small
        # But we want to test integration, so maybe lite_mode=False is better if possible?
        # Let's try lite_mode=True first to verify flow, as user asked to "mira que el api funcione"
        # If we use lite_mode=True, we avoid translation/diarization overhead maybe?
        # User wants "genera pruebas... mira que el api funcione correctly"
        # Let's enable diarization to test that flow too, but maybe on a short segment?
        # The file is 480KB mp3, likely short.
        
        result = await transcribe_full_analysis(
            file=upload_file,
            lite_mode=True, 
            audio_weight=0.5,
            enable_diarization=True,
            num_speakers=None
        )
        
        if isinstance(result, dict) and result.get("status") == "success":
             logger.info("✔ API Response status is success")
             logger.info(f"✔ Transcription length: {len(result.get('transcription', ''))}")
             logger.info(f"✔ Speakers detected: {result.get('diarization', {}).get('num_speakers')}")
        else:
             logger.error(f"✘ API Response failed or invalid format: {result}")
             
    except Exception as e:
        logger.error(f"✘ API Integration test raised exception: {e}")
        import traceback
        traceback.print_exc()

async def main():
    logger.info("STARTING VERIFICATION SUITE")
    await verify_validators()
    await verify_resilience()
    await verify_api_integration()
    logger.info("VERIFICATION SUITE COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())
