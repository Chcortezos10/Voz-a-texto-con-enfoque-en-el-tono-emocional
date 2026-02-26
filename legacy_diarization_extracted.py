from typing import List, Dict, Any, Optional

def merge_consecutive_same_speaker(
    segments_with_text: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Fusiona segmentos consecutivos del mismo hablante.
    """
    if not segments_with_text: return []
    merged = []
    current = segments_with_text[0].copy()
    
    for nxt in segments_with_text[1:]:
        # Comparar speaker (int o str)
        s1 = current.get("speaker", 0)
        s2 = nxt.get("speaker", 0)
        # Normalizar a int si es posible para comparar
        try:
             if isinstance(s1, str) and "_" in s1: s1 = int(s1.split("_")[1])
        except: pass
        try:
             if isinstance(s2, str) and "_" in s2: s2 = int(s2.split("_")[1])
        except: pass

        if s1 == s2:
            # Merge
            current["end"] = max(current["end"], nxt["end"])
            txt1 = current.get("text_es") or current.get("text", "")
            txt2 = nxt.get("text_es") or nxt.get("text", "")
            full_txt = (txt1 + " " + txt2).strip()
            
            current["text_es"] = full_txt
            current["text"] = full_txt
        else:
            merged.append(current)
            current = nxt.copy()
            
    merged.append(current)
    return merged



def format_labeled_transcription(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        spk = s.get("speaker", 0)
        # Parse speaker if string
        if isinstance(spk, str) and spk.startswith("speaker_"):
            try: spk = int(spk.split("_")[1])
            except: spk = 0
            
        text = s.get("text_es") or s.get("text", "")
        if text:
            lines.append(f"[Hablante {int(spk)+1}]: {text}")
    return " ".join(lines)


