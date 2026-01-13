import asyncio
import os
import io
import sys
import traceback
from fastapi import UploadFile

async def test_api():
    print('='*60)
    print('INICIANDO PRUEBAS DE API')
    print('='*60)
    
    # Test 1: Load modules
    print('\n[TEST 1] Cargando módulos...')
    try:
        from app_fastapi import transcribe_full_analysis, load_models
        print('✔ Módulos cargados correctamente')
    except Exception as e:
        print(f'✘ Error cargando módulos: {e}')
        traceback.print_exc()
        return
    
    # Test 2: Load models
    print('\n[TEST 2] Cargando modelos...')
    try:
        models = load_models()
        print(f'✔ Modelos cargados: {list(models.keys())}')
    except Exception as e:
        print(f'✘ Error cargando modelos: {e}')
        traceback.print_exc()
        return
    
    # Test 3: Check test file
    file_path = 'pruebas/test#2.mp3'
    print(f'\n[TEST 3] Verificando archivo de prueba: {file_path}')
    if not os.path.exists(file_path):
        print(f'✘ Archivo no encontrado: {file_path}')
        return
    print(f'✔ Archivo existe, tamaño: {os.path.getsize(file_path)} bytes')
    
    # Test 4: Call API with lite_mode
    print('\n[TEST 4] Llamando API (lite_mode=True, diarization=False)...')
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_obj = io.BytesIO(file_content)
        upload_file = UploadFile(filename='test#2.mp3', file=file_obj)
        
        result = await transcribe_full_analysis(
            file=upload_file,
            lite_mode=True,
            audio_weight=0.5,
            enable_diarization=False,
            num_speakers=None
        )
        
        if isinstance(result, dict) and result.get('status') == 'success':
            print('✔ API retornó success')
            print(f'  - Transcripción: {result.get("transcription", "")[:100]}...')
            print(f'  - Segmentos: {len(result.get("segments", []))}')
        else:
            print(f'✘ API falló: {result}')
            
    except Exception as e:
        print(f'✘ Error en API: {e}')
        traceback.print_exc()
    
    # Test 5: Call API with full mode and diarization
    print('\n[TEST 5] Llamando API (lite_mode=False, diarization=True)...')
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_obj = io.BytesIO(file_content)
        upload_file = UploadFile(filename='test#2.mp3', file=file_obj)
        
        result = await transcribe_full_analysis(
            file=upload_file,
            lite_mode=False,
            audio_weight=0.5,
            enable_diarization=True,
            num_speakers=2
        )
        
        if isinstance(result, dict) and result.get('status') == 'success':
            print('✔ API retornó success')
            print(f'  - Transcripción: {result.get("transcription", "")[:100]}...')
            print(f'  - Segmentos: {len(result.get("segments", []))}')
            print(f'  - Speakers detectados: {result.get("diarization", {}).get("num_speakers")}')
            print(f'  - Emoción global: {result.get("global_emotions", {}).get("top_emotion")}')
        else:
            print(f'✘ API falló: {result}')
            
    except Exception as e:
        print(f'✘ Error en API: {e}')
        traceback.print_exc()
    
    print('\n' + '='*60)
    print('PRUEBAS COMPLETADAS')
    print('='*60)

if __name__ == '__main__':
    asyncio.run(test_api())
