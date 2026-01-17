from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])

history_file = Path("data/analysis_history.json")

def load_history() -> List[Dict[str, Any]]:
    try:
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando historial: {e}")
    return []

def save_history_to_file(history: List[Dict[str, Any]]):
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error guardando historial: {e}")

@router.get("/")
async def get_history(limit: int = 50):
    try:
        history = load_history()
        sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)
        return sorted_history[:limit]
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{item_id}")
async def get_history_item(item_id: str):
    try:
        history = load_history()
        for item in history:
            if item.get("id") == item_id:
                return item
        raise HTTPException(status_code=404, detail="Item no encontrado")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_to_history(data: Dict[str, Any]):
    try:
        history = load_history()
        
        dominant_emotion = (
            data.get("dominant_emotion") or 
            data.get("global_emotions", {}).get("top_emotion") or
            "neutral"
        )
        
        duration = (
            data.get("duration") or 
            data.get("metadata", {}).get("total_duration") or
            data.get("metadata", {}).get("audio_duration") or
            0
        )
        
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "filename": data.get("filename", "unknown"),
            "dominant_emotion": dominant_emotion,
            "num_segments": data.get("num_segments", len(data.get("segments", []))),
            "duration": duration,
            "global_emotions": data.get("global_emotions", {}),
            "segments": data.get("segments", []),
            "transcription": data.get("transcription", ""),
            "diarization": data.get("diarization", {}),
            "metadata": data.get("metadata", {}),
            "emotion_timeline": data.get("emotion_timeline", []),
            "speaker_stats": data.get("speaker_stats", {})
        }
        
        history.append(entry)
        
        if len(history) > 100:
            history.pop(0)

        save_history_to_file(history)
        logger.info(f"Historial guardado: {entry['id']} - {entry['filename']}")
        
        return {"status": "success", "saved_id": entry["id"]}
    except Exception as e:
        logger.error(f"Error guardando en historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_history():
    save_history_to_file([])
    logger.info("Historial limpiado")
    return {"status": "success", "message": "Historial limpiado"}

@router.delete("/{item_id}")
async def delete_history_item(item_id: str):
    try:
        history = load_history()
        initial_count = len(history)
        history = [item for item in history if item.get("id") != item_id]
        
        if len(history) == initial_count:
            raise HTTPException(status_code=404, detail="Item no encontrado")
        
        save_history_to_file(history)
        logger.info(f"Item eliminado del historial: {item_id}")
        
        return {"status": "success", "message": "Item eliminado"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando item: {e}")
        raise HTTPException(status_code=500, detail=str(e))