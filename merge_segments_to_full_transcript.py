# merge_segments_to_full_transcript.py
"""
Fusiona segmentos de transcripción consecutivos del mismo hablante.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

from config import PROJECT_ROOT


def merge_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fusiona segmentos consecutivos del mismo hablante en párrafos.
    
    Args:
        segments: Lista de segmentos con speaker, start, end, text
        
    Returns:
        Lista de segmentos fusionados
    """
    # Asegurarse de que los segmentos estén ordenados por tiempo
    segments = sorted(segments, key=lambda s: s.get("start", 0.0))
    
    merged = []
    for seg in segments:
        speaker = (
            seg.get("speaker") or 
            seg.get("speaker_label") or 
            seg.get("speaker_id") or 
            "speaker_0"
        )
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        
        if not merged:
            merged.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "texts": [text]
            })
        else:
            last = merged[-1]
            if speaker == last["speaker"]:
                # Extender rango temporal y añadir texto
                last["end"] = max(last["end"], end)
                if text:
                    last["texts"].append(text)
            else:
                merged.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "texts": [text]
                })
    
    return merged


def format_transcription(merged_blocks: List[Dict[str, Any]]) -> str:
    """
    Formatea bloques fusionados en texto con etiquetas de hablante.
    
    Args:
        merged_blocks: Lista de bloques fusionados
        
    Returns:
        Texto formateado con etiquetas [speaker]
    """
    lines = []
    for block in merged_blocks:
        speaker = block["speaker"]
        # Concatenar textos con espacio
        paragraph = " ".join(t for t in block["texts"] if t).strip()
        if paragraph:
            lines.append(f"[{speaker}] {paragraph}")
        else:
            lines.append(f"[{speaker}] (sin texto)")
    
    return "\n\n".join(lines)


def process_transcription(input_json: Path, output_txt: Path) -> None:
    """
    Procesa un archivo JSON de transcripción y genera salida fusionada.
    
    Args:
        input_json: Ruta al archivo JSON de entrada
        output_txt: Ruta al archivo TXT de salida
    """
    if not input_json.exists():
        print(f"Error: No se encontró {input_json}")
        return
    
    data = json.loads(input_json.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    
    merged = merge_segments(segments)
    transcription = format_transcription(merged)
    
    output_txt.write_text(transcription, encoding="utf-8")
    print(f"Transcripción guardada en: {output_txt}")


if __name__ == "__main__":
    input_file = PROJECT_ROOT / "transcription_sequence_streamlit.json"
    output_file = PROJECT_ROOT / "transcription_labeled_full_merged.txt"
    
    process_transcription(input_file, output_file)
