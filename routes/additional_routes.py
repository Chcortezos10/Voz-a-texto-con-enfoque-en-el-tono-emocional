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

@router.post("/export/srt", tags=["export"])
async def export_srt_form(segments: str = Form(...),
    include_emotions: bool = Form(True),
    include_speaker: bool = Form(True),
    filename: str = Form("subtitulos")
) -> Response:
    try:
        segments_list = json.loads(segments)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato de segmentos inválido")

    exporter = get_export_manager()

    data_dict = {"segments":segments_list}
    content = exporter.export_srt(data_dict)
    
    return Response(content=content, media_type="text/plain",headers={"Content-Disposition": f"attachment; filename={filename}.srt"})

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

    data_dict = {"segments":segments_list}
    content = exporter.export_vtt(data_dict)

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

    data_dict = {"segments":segments_list}
    content = exporter.export_csv(data_dict)

    return Response(content=content, media_type="text/csv",headers={"Content-Disposition": f"attachment; filename={filename}.csv"})

@staticmethod
def export_pdf(data: 'ExportData') -> bytes:
    """
    Exporta transcripción a formato PDF.
    Requiere instalar: pip install reportlab
    
    Args:
        data: Objeto ExportData con todos los datos
        
    Returns:
        Bytes del archivo PDF
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Título
        title = Paragraph("Analisis de Transcripcion con Emociones", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Resumen
        summary_data = [
            ["Duracion Total", f"{data.metadata.get('total_duration', 0):.2f}s"],
            ["Emocion Dominante", data.global_emotions.get("top_emotion", "neutral")],
            ["Segmentos", str(len(data.segments))]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Segmentos
        seg_data = [["Tiempo", "Hablante", "Emoción", "Texto"]]
        for seg in data.segments[:50]:  # Limitar a 50 para no saturar
            seg_data.append([
                f"{seg.get('start', 0):.1f}s",
                seg.get("speaker_label", "")[:15],
                seg.get("emotion", "")[:10],
                seg.get("text_es", "")[:60]
            ])
        
        seg_table = Table(seg_data)
        seg_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        elements.append(seg_table)
        
        doc.build(elements)
        buffer.seek(0)
        
        return buffer.read()
        
    except ImportError:
        logger.error("reportlab no instalado Ejecutar: pip install reportlab")
        raise
    except Exception as e:
        logger.error(f"Error exportando PDF: {e}")
        raise


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
