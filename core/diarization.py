from typing import List, Dict, Any

def merge_consecutive_same_speaker(
    segments_with_text: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Fusiona segmentos consecutivos del mismo hablante.
    """
    if not segments_with_text:
        return []

    merged = []
    # Make a deep copy to avoid modifying original if needed, 
    # but here we just copy the dict structure
    current = segments_with_text[0].copy()
    
    for nxt in segments_with_text[1:]:
        # Comparar speaker (int o str)
        s1 = current.get("speaker", 0)
        s2 = nxt.get("speaker", 0)
        
        # Normalizar a int si es posible para comparar (handle "speaker_01")
        try:
             if isinstance(s1, str) and "_" in s1: s1 = int(s1.split("_")[-1])
        except: pass
        try:
             if isinstance(s2, str) and "_" in s2: s2 = int(s2.split("_")[-1])
        except: pass

        if s1 == s2:
            # Merge
            current["end"] = max(current["end"], nxt["end"])
            txt1 = current.get("text_es") or current.get("text", "")
            txt2 = nxt.get("text_es") or nxt.get("text", "")
            full_txt = (txt1 + " " + txt2).strip()
            
            # Update both text fields to be safe
            current["text_es"] = full_txt
            current["text"] = full_txt
        else:
            merged.append(current)
            current = nxt.copy()
            
    merged.append(current)
    return merged

def format_labeled_transcription(segments: List[Dict]) -> str:
    """
    Formatea la transcripción con etiquetas [Hablante N].
    """
    lines = []
    for s in segments:
        spk = s.get("speaker", 0)
        # Parse speaker if string "speaker_01" -> 1
        if isinstance(spk, str) and "_" in spk:
            try: 
                spk = int(spk.split("_")[-1])
            except: 
                spk = 0
        elif isinstance(spk, str):
            try:
                spk = int(spk)
            except:
                spk = 0
            
        text = s.get("text_es") or s.get("text", "")
        if text:
            # Legacy format: [Hablante N]: Text
            lines.append(f"[Hablante {int(spk)+1}]: {text}")
            
    return " ".join(lines)
