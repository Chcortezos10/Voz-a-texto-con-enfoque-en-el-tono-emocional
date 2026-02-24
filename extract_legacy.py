import subprocess
import re

try:
    # Get content from git
    content = subprocess.check_output(['git', 'show', '3e9f4b0~1:core/diarization.py']).decode('utf-8', errors='ignore')
    
    # Extract functions
    functions = {}
    current_func = None
    lines = content.split('\n')
    
    captured_lines = []
    
    for line in lines:
        if line.startswith('def '):
            if current_func:
                functions[current_func] = '\n'.join(captured_lines)
            current_func = line.split('(')[0].replace('def ', '').strip()
            captured_lines = [line]
        elif current_func:
            captured_lines.append(line)
            
    if current_func:
        functions[current_func] = '\n'.join(captured_lines)
        
    # Save specific functions
    target_funcs = ['merge_consecutive_same_speaker', 'format_labeled_transcription']
    
    with open('legacy_diarization_extracted.py', 'w', encoding='utf-8') as f:
        f.write("from typing import List, Dict, Any, Optional\n\n")
        for func_name in target_funcs:
            if func_name in functions:
                f.write(functions[func_name] + "\n\n")
            else:
                f.write(f"# {func_name} not found in legacy file\n")
                
    print(" extraction complete")
    
except Exception as e:
    print(f"Error: {e}")
