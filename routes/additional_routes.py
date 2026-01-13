"""
Rutas adicionales para transcripción
en la nube (cloud)
-exportacion de datos
-gesition de sesiones
-transcripcion en la nube
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
import io

from core.transcription_cloud import (
    get_transcription_service,
    TranscriptionProvider,
    validate_api_key
)
from core.export_manager import (
    get_export_manager,
    ExportData
)

logger = logging.getLogger(__name__)

_sessions_store: Dict[str, Dict[str, Any]] = {}

router = APIRouter()

#Trascripcion en la nube 

@router.post("/config/api_key",tags=["configuracion"])
async def set_api_key(provider: str = Form(...), api_key: str = Form(...)) -> Dict[str, Any]:
    """
    Configura la clave de API para el proveedor de transcripción en la nube.
    """
    if provider not in ["openai", "groq"]:
        raise HTTPException(status_code=400, detail="Proveedor no válido. Use 'openai' o 'groq'")

    validation = validate_api_key(provider, api_key)
    
    if not validation.get("valid"):
        return {
            "status": "error",
            "message": validation.get("message", "API key inválida"),
            "provider": provider
        }

    #configurar el servicio de transcripcion

    service = get_transcription_service()

    if provider == "openai":
        service.set_openai_api_key(TranscriptionProvider.OPENAI, api_key)
    elif provider == "groq":
        service.set_groq_api_key(TranscriptionProvider.GROQ, api_key)
    
    return {
        "status": "success",
        "message": "Clave de API configurada correctamente",
        "provider": provider
    }

@router.post("/config/validate_api_key",tags=["configuracion"])
async def validate_api_key_endpoint(provider: str = Form(...), api_key: str = Form(...)) -> Dict[str, Any]:
    """
    Valida la clave de API para el proveedor de transcripción en la nube.
    """
    if provider not in ["openai", "groq"]:
        raise HTTPException(status_code=400, detail="Proveedor no válido. Use 'openai' o 'groq'")
    
    result = validate_api_key(provider, api_key)
    return result

@router.get("/config/api_key",tags=["configuracion"])
async def estimate_cost(
    duration_seconds: float = Query(..., description="Duración del audio en segundos"),
    provider: str = Query("local", description="Proveedor: local, openai, groq")
) -> Dict[str, Any]:
    """
    Estima el costo de la transcripción en la nube.
    """
    provied_map = {
        "local": TranscriptionProvider.LOCAL,
        "openai": TranscriptionProvider.OPENAI,
        "groq": TranscriptionProvider.GROQ
    }

    if provider not in ["local", "openai", "groq"]:
        raise HTTPException(status_code=400, detail="Proveedor no válido. Use 'local', 'openai' o 'groq'")
    
    service = get_transcription_service()
    estimate = service.estimate_cost(duration_seconds, provied_map[provider])
    return estimate

@router.get("/config/estimate_cost", tags=["configuracion"])
async def get_providers() -> Dict[str, Any]:
    """
    Obtiene el servicio de transcripción en la nube.
    """
    service = get_transcription_service()
    return {
        "providers": [
            {
                "id": "local",
                "name": "Local (Whisper)",
                "description": "Procesamiento local. Gratis pero más lento.",
                "cost_per_minute": 0.0,
                "speed": "1x tiempo real",
                "available": True,
                "requires_key": False
            },
            {
                "id": "openai",
                "name": "OpenAI Whisper API",
                "description": "API de OpenAI. Rápido y preciso.",
                "cost_per_minute": 0.006,
                "speed": "~2x más rápido que tiempo real",
                "available": service.has_api_key(TranscriptionProvider.OPENAI),
                "requires_key": True
            },
            {
                "id": "groq",
                "name": "Groq (Ultra Rápido)",
                "description": "25x más rápido que tiempo real. Casi gratis.",
                "cost_per_minute": 0.0001,
                "speed": "~25x más rápido que tiempo real",
                "available": service.has_api_key(TranscriptionProvider.GROQ),
                "requires_key": True
            }
        ]
    }


#exportacion de datos

@router.post("/export/str", tags=["export"])
async def export_str(segments: str = Form(...),
    include_emotions: bool = Form(True),
    include_speaker: bool = Form(True),
    filename: str = Form("subtitulos")
) -> Response:
    """
    Exporta los segmentos de la transcripción en formato STR.
    """

    try:
        segments_list = json.loads(segments)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_str(segments_list, include_emotions, include_speaker)

    return Response(content=content, media_type="text/plain",headers={"Content-Disposition": f"attachment; filename={filename}.str"})

@router.post("/export/vtt", tags=["export"])
async def export_vtt(
    segments: str = Form(...),
    include_emotions: bool = Form(True),
    include_speaker: bool = Form(True),
    filename: str = Form("subtitulos")
) -> Response:

    """
    Exporta los segmentos de la transcripción en formato VTT.
    """

    try:
        segments_list = json.loads(segments)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_vtt(segments_list, include_emotions, include_speaker)

    return Response(content=content, media_type="text/vtt",headers={"Content-Disposition": f"attachment; filename={filename}.vtt"})

@router.post("/export/csv", tags=["export"])
async def export_csv(
    segments: str = Form(...),
    include_all_emotions: bool = Form(True),
    filename: str = Form("subtitulos")
) -> Response:
    """
    Exporta los segmentos de la transcripción en formato CSV.
    """

    try:
        segments_list = json.loads(segments)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_csv(segments_list, include_all_emotions)

    return Response(content=content, media_type="text/csv",headers={"Content-Disposition": f"attachment; filename={filename}.csv"})

@router.post("/export/excel", tags=["export"])
async def export_excel(
    segments: str = Form(...),
    global_emotions: str = Form("{}"),
    speaker_stats: str = Form("{}"),
    metadata: str = Form("{}"),
    filename: str = Form("analisis_emocional")
) -> Response:
    """
    Exporta los segmentos de la transcripción en formato Excel.
    """

    try:
        data= ExportData(
            segments=json.loads(segments),
            global_emotions=json.loads(global_emotions),
            speaker_stats=json.loads(speaker_stats),
            metadata=json.loads(metadata),
            filename=filename
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_excel(data)

    return Response(content=content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ,headers={"Content-Disposition": f"attachment; filename={filename}.xlsx"})

@router.post("/export/pdf", tags=["export"])
async def export_pdf(
    segments: str = Form(...),
    global_emotions: str = Form("{}"),
    speaker_stats: str = Form("{}"),
    metadata: str = Form("{}"),
    filename: str = Form("analisis_emocional")
) -> Response:
    """
    Exporta los segmentos de la transcripción en formato PDF.
    """

    try:
        data= ExportData(
            segments=json.loads(segments),
            global_emotions=json.loads(global_emotions),
            speaker_stats=json.loads(speaker_stats),
            metadata=json.loads(metadata),
            filename=filename
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_pdf(data)

    return Response(content=content, media_type="application/pdf",headers={"Content-Disposition": f"attachment; filename={filename}.pdf"})

@router.post("/export/json", tags=["export"])
async def export_json(
    segments: str = Form(...),
    global_emotions: str = Form("{}"),
    speaker_stats: str = Form("{}"),
    metadata: str = Form("{}"),
    filename: str = Form("analisis_emocional")
) -> Response:
    """
    Exporta los segmentos de la transcripción en formato JSON.
    """

    try:
        data= ExportData(
            segments=json.loads(segments),
            global_emotions=json.loads(global_emotions),
            speaker_stats=json.loads(speaker_stats),
            metadata=json.loads(metadata),
            filename=filename
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()
    content = exporter.export_json(data)

    return Response(content=content, media_type="application/json",headers={"Content-Disposition": f"attachment; filename={filename}.json"})

#alamacenamiento termporal de las sesiones 

@router.post("/store_session", tags=["sessions"])
async def store_session(namer:str = None, data:str = None) -> Dict[str,Any]:

    """
    Almacena una sesión en el almacenamiento temporal.
    """
    session_id = str(uuid.uuid4())
    time_stamp = datetime.now()

    try:
        session_data=json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de datos inválido")
    

    session = {
        "id": session_id,
        "name": namer or f"Sesión {time_stamp.strftime('%Y-%m-%d %H:%M')}",
        "created_at": time_stamp.isoformat(),
        "updated_at": time_stamp.isoformat(),
        "data": session_data
    }

    _sessions_store[session_id] = session
    return {"status": "success",
        "session_id": session_id,
        "name": session["name"],
        "created_at": session["created_at"]}


@router.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Obtiene una sesión por ID.
    """
    if session_id not in _sessions_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    return _sessions_store[session_id]



@router.put("/sessions/{session_id}", tags=["Sessions"])
async def update_session(
    session_id: str,
    name: str = Form(None),
    data: str = Form(None)
) -> Dict[str, Any]:
    """
    Actualiza una sesion existente.
    """
    if session_id not in _sessions_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    session = _sessions_store[session_id]
    
    if name:
        session["name"] = name
    
    if data:
        try:
            session["data"] = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="JSON inválido")
    
    session["updated_at"] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "session_id": session_id,
        "updated_at": session["updated_at"]
    }

@router.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str) -> Dict[str, Any]:
    """
    Elimina una sesion existente.
    """
    if session_id not in _sessions_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    del _sessions_store[session_id]
    
    return {"status": "success", "message": "Sesión eliminada exitosamente"}

@router.get("/sessions", tags=["Sessions"])
async def list_sessions(
    limit: int = Query(20, ge=1, le=100)
) -> Dict[str, Any]:
    """
    Lista todas las sesiones.
    """
    sessions = sorted(
        _sessions_store.values(),
        key=lambda x: x["updated_at"],
        reverse=True
    )[:limit]
    
    return {
        "sessions": [
            {
                "id": s["id"],
                "name": s["name"],
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
                "segments_count": len(s["data"].get("segments", []))
            }
            for s in sessions
        ],
        "total": len(_sessions_store)
    }

@router.post("/transcription/update-segment", tags=["Transcription"])
async def update_segment(
    session_id: str = Form(...),
    segment_index: int = Form(...),
    text: str = Form(None),
    speaker_label: str = Form(None),
    speaker_id: int = Form(None)
) -> Dict[str, Any]:
    """
    Actualiza un segmento de transcripcin.
    Permite corregir texto o reasignar hablante.
    """
    if session_id not in _sessions_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    session = _sessions_store[session_id]
    segments = session["data"].get("segments", [])
    
    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(status_code=400, detail="Índice de segmento inválido")
    
    segment = segments[segment_index]
    
    if text is not None:
        segment["text_es"] = text
        segment["text"] = text
        segment["_edited"] = True
    
    if speaker_label is not None:
        segment["speaker_label"] = speaker_label
        segment["_speaker_edited"] = True
    
    if speaker_id is not None:
        segment["speaker_id"] = speaker_id
    
    session["updated_at"] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "segment_index": segment_index,
        "updated_segment": segment
    }


@router.post("/transcription/merge-speakers", tags=["Transcription"])
async def merge_speakers(
    session_id: str = Form(...),
    source_speaker_id: int = Form(...),
    target_speaker_id: int = Form(...),
    target_label: str = Form(None)
) -> Dict[str, Any]:
    """
    Fusiona dos hablantes en uno.

    """
    if session_id not in _sessions_store:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    session = _sessions_store[session_id]
    segments = session["data"].get("segments", [])
    
    merged_count = 0
    for segment in segments:
        if segment.get("speaker_id") == source_speaker_id:
            segment["speaker_id"] = target_speaker_id
            if target_label:
                segment["speaker_label"] = target_label
            merged_count += 1
    
    session["updated_at"] = datetime.now().isoformat()
    
    return {
        "status": "success",
        "merged_segments": merged_count,
        "source_speaker_id": source_speaker_id,
        "target_speaker_id": target_speaker_id
    }


#End point para la transcripcion
@router.post("/transcribe/with-provider", tags=["Transcription"])
async def transcribe_with_provider(
    file: UploadFile = File(...),
    provider: str = Form("local"),
    api_key: str = Form(None),
    language: str = Form("es")
) -> Dict[str, Any]:
    """
    Transcribe audio usando el proveedor especificado.
    """
    import tempfile
    import aiofiles
    
    provider_map = {
        "local": TranscriptionProvider.LOCAL,
        "openai": TranscriptionProvider.OPENAI,
        "groq": TranscriptionProvider.GROQ
    }
    
    if provider not in provider_map:
        raise HTTPException(status_code=400, detail="Proveedor no valido")
    
    # Verificar API key para proveedores cloud
    if provider in ["openai", "groq"] and not api_key:
        service = get_transcription_service()
        if not service.has_api_key(provider_map[provider]):
            raise HTTPException(
                status_code=400,
                detail=f"Se requiere API key para {provider}"
            )
    
    # Guardar archivo temporal
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribir
        service = get_transcription_service()
        result = await service.transcribe(
            audio_path=tmp_path,
            provider=provider_map[provider],
            language=language,
            api_key=api_key
        )
        
        return {
            "status": "success",
            "provider": result.provider,
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "duration": result.duration,
            "processing_time": result.processing_time,
            "cost_estimate": result.cost_estimate
        }
        
    except Exception as e:
        logger.error(f"Error en transcripción: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
