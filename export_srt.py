# export_srt.py
"""
Exporta transcripción etiquetada a formato SRT (subtítulos).
"""
import json
from pathlib import Path

from config import PROJECT_ROOT


def fmt(t: float) -> str:
    """Formatea tiempo en segundos a formato SRT (HH:MM:SS,mmm)."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_to_srt(input_json: Path, output_srt: Path) -> None:
    """
    Exporta transcripción JSON a formato SRT.
    
    Args:
        input_json: Ruta al archivo JSON con la transcripción
        output_srt: Ruta de salida para el archivo SRT
    """
    data = json.loads(input_json.read_text(encoding="utf-8"))
    lines = []
    
    for i, seg in enumerate(data["segments"], 1):
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"].strip()
        cue = f"{i}\n{fmt(start)} --> {fmt(end)}\n[{speaker}] {text}\n"
        lines.append(cue)
    
    output_srt.write_text("\n".join(lines), encoding="utf-8")
    print(f"SRT guardado en: {output_srt}")


if __name__ == "__main__":
    input_file = PROJECT_ROOT / "transcription_sequence_changepoint.json"
    output_file = PROJECT_ROOT / "transcription.srt"
    
    if not input_file.exists():
        print(f"Error: No se encontró {input_file}")
    else:
        export_to_srt(input_file, output_file)
