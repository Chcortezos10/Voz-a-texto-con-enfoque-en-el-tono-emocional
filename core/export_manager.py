"""
Gestor de exportación de transcripciones en múltiples formatos
"""
import json
import csv
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from io import StringIO
from pathlib import Path

logger = logging.getLogger(__name__)


class ExportManager:
    """Gestor para exportar transcripciones en diferentes formatos"""
    
    @staticmethod
    def export_json(data: Dict[str, Any], pretty: bool = True) -> str:
        """
        Exporta datos a formato JSON.
        
        Args:
            data: Datos de transcripción
            pretty: Si True, formatea con indentación
            
        Returns:
            String JSON
        """
        try:
            if pretty:
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error exportando JSON: {e}")
            raise
    
    @staticmethod
    def export_csv(data: Dict[str, Any]) -> str:
        """
        Exporta transcripción a formato CSV.
        
        Args:
            data: Datos de transcripción
            
        Returns:
            String CSV
        """
        try:
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "Tiempo Inicio (s)",
                "Tiempo Fin (s)",
                "Duración (s)",
                "Hablante",
                "Emoción",
                "Intensidad",
                "Texto Español",
                "Texto Inglés"
            ])
            
            # Rows
            segments = data.get("segments", [])
            for seg in segments:
                writer.writerow([
                    round(seg.get("start", 0), 2),
                    round(seg.get("end", 0), 2),
                    round(seg.get("duration", 0), 2),
                    seg.get("speaker_label", "Hablante 1"),
                    seg.get("emotion", "neutral"),
                    round(seg.get("intensity", 0), 2),
                    seg.get("text_es", ""),
                    seg.get("text_en", "")
                ])
            
            # Summary
            writer.writerow([])
            writer.writerow(["=== RESUMEN ==="])
            writer.writerow(["Duración Total", round(data.get("metadata", {}).get("total_duration", 0), 2)])
            writer.writerow(["Número de Hablantes", data.get("diarization", {}).get("num_speakers", 1)])
            writer.writerow(["Emoción Dominante", data.get("global_emotions", {}).get("top_emotion", "neutral")])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error exportando CSV: {e}")
            raise
    
    @staticmethod
    def export_srt(data: Dict[str, Any]) -> str:
        """
        Exporta transcripción a formato SRT (subtítulos).
        
        Args:
            data: Datos de transcripción
            
        Returns:
            String SRT
        """
        try:
            srt_content = ""
            segments = data.get("segments", [])
            
            for i, seg in enumerate(segments, 1):
                start_time = ExportManager._format_srt_time(seg.get("start", 0))
                end_time = ExportManager._format_srt_time(seg.get("end", 0))
                text = seg.get("text_es", "")
                speaker = seg.get("speaker_label", "Hablante 1")
                emotion = seg.get("emotion", "neutral")
                
                srt_content += f"{i}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"[{speaker}] ({emotion}) {text}\n\n"
            
            return srt_content
            
        except Exception as e:
            logger.error(f"Error exportando SRT: {e}")
            raise
    
    @staticmethod
    def export_txt(data: Dict[str, Any]) -> str:
        """
        Exporta transcripción a formato TXT plano.
        
        Args:
            data: Datos de transcripción
            
        Returns:
            String TXT
        """
        try:
            txt_content = "=" * 80 + "\n"
            txt_content += "TRANSCRIPCIÓN CON ANÁLISIS EMOCIONAL\n"
            txt_content += "=" * 80 + "\n\n"
            
            # Metadata
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            txt_content += f"Fecha: {now}\n"
            txt_content += f"Duración: {round(data.get('metadata', {}).get('total_duration', 0), 2)}s\n"
            txt_content += f"Hablantes: {data.get('diarization', {}).get('num_speakers', 1)}\n"
            txt_content += f"Emoción dominante: {data.get('global_emotions', {}).get('top_emotion', 'neutral')}\n\n"
            txt_content += "=" * 80 + "\n\n"
            
            # Segments
            segments = data.get("segments", [])
            for seg in segments:
                speaker = seg.get("speaker_label", "Hablante 1")
                emotion = seg.get("emotion", "neutral")
                start = round(seg.get("start", 0), 1)
                end = round(seg.get("end", 0), 1)
                text = seg.get("text_es", "")
                
                txt_content += f"[{start}s - {end}s] {speaker} ({emotion}):\n"
                txt_content += f"{text}\n\n"
            
            return txt_content
            
        except Exception as e:
            logger.error(f"Error exportando TXT: {e}")
            raise
    
    @staticmethod
    def export_vtt(data: Dict[str, Any]) -> str:
        """
        Exporta transcripción a formato WebVTT.
        
        Args:
            data: Datos de transcripción
            
        Returns:
            String VTT
        """
        try:
            vtt_content = "WEBVTT\n\n"
            segments = data.get("segments", [])
            
            for seg in segments:
                start_time = ExportManager._format_vtt_time(seg.get("start", 0))
                end_time = ExportManager._format_vtt_time(seg.get("end", 0))
                text = seg.get("text_es", "")
                speaker = seg.get("speaker_label", "Hablante 1")
                
                vtt_content += f"{start_time} --> {end_time}\n"
                vtt_content += f"<v {speaker}>{text}\n\n"
            
            return vtt_content
            
        except Exception as e:
            logger.error(f"Error exportando VTT: {e}")
            raise
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """
        Formatea segundos a formato SRT (HH:MM:SS,mmm).
        
        Args:
            seconds: Tiempo en segundos
            
        Returns:
            Tiempo formateado
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """
        Formatea segundos a formato WebVTT (HH:MM:SS.mmm).
        
        Args:
            seconds: Tiempo en segundos
            
        Returns:
            Tiempo formateado
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    
    @staticmethod
    def save_to_file(content: str, file_path: str):
        """
        Guarda contenido a un archivo.
        
        Args:
            content: Contenido a guardar
            file_path: Ruta del archivo
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Exportado a: {file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando archivo: {e}")
            raise


def export_transcription(
    data: Dict[str, Any],
    output_format: str = "json",
    output_path: Optional[str] = None
) -> str:
    """
    Exporta transcripción al formato especificado.
    
    Args:
        data: Datos de transcripción
        output_format: Formato de salida (json, csv, srt, txt, vtt)
        output_path: Ruta opcional para guardar archivo
        
    Returns:
        Contenido exportado como string
    """
    manager = ExportManager()
    
    if output_format.lower() == "json":
        content = manager.export_json(data)
    elif output_format.lower() == "csv":
        content = manager.export_csv(data)
    elif output_format.lower() == "srt":
        content = manager.export_srt(data)
    elif output_format.lower() == "txt":
        content = manager.export_txt(data)
    elif output_format.lower() == "vtt":
        content = manager.export_vtt(data)
    else:
        raise ValueError(f"Formato no soportado: {output_format}")
    
    # Guardar si se especifica ruta
    if output_path:
        manager.save_to_file(content, output_path)
    
    return content


from dataclasses import dataclass

@dataclass
class ExportData:
    """Datos para exportación"""
    segments: List[Dict[str, Any]]
    global_emotions: Dict[str, Any]
    speaker_stats: Dict[str, Any]
    metadata: Dict[str, Any]
    filename: str = "export"


# Singleton instance
_export_manager: Optional[ExportManager] = None


def get_export_manager() -> ExportManager:
    """Obtiene la instancia del gestor de exportación"""
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager
