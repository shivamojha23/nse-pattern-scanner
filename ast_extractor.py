import ast
import os

def run():
    with open("pattern_scanner.py", "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)

    funcs = {}
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.ClassDef):
            pass

    config_assigns = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'get_nifty_list':
            break
        if isinstance(node, ast.Assign):
            # Check if any target is uppercase
            is_config = False
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    is_config = True
                    break
            if is_config:
                config_assigns.append(ast.get_source_segment(src, node))
            
    config_code = "\n".join(config_assigns)
    
    os.makedirs("detectors", exist_ok=True)
    with open("detectors/__init__.py", "w") as f:
        pass

    core_imports = """import numpy as np
import pandas as pd
import math
import datetime
import os
import sys
from scipy.signal import find_peaks
from scipy.stats import linregress
"""

    core_code = f"""{core_imports}

{config_code}

{funcs['smooth_prices']}

{funcs['compute_quality_score']}

{funcs['refine_peak']}

{funcs['refine_trough']}
"""
    with open("detectors/core.py", "w", encoding="utf-8") as f:
        f.write(core_code)

    def write_detector(filename, func_name):
        code = f"""{core_imports}from .core import *\n\n{funcs[func_name]}\n"""
        with open(f"detectors/{filename}.py", "w", encoding="utf-8") as f:
            f.write(code)

    write_detector("cup_and_handle", "detect_cup_and_handle")
    write_detector("bull_flag", "detect_bull_flag")
    write_detector("bear_flag", "detect_bear_flag")
    write_detector("pennant", "detect_pennant")
    write_detector("head_and_shoulders", "detect_head_and_shoulders")
    write_detector("double_top", "detect_double_top")
    write_detector("double_bottom", "detect_double_bottom")

    # Now rewrite pattern_scanner.py to replace these things.
    # Actually, if we just delete these nodes from the AST and unparse it?
    # Unparse destroys comments.
    
    # We can just delete the text of these functions!
    to_delete = [
        funcs['detect_cup_and_handle'],
        funcs['detect_bull_flag'],
        funcs['detect_bear_flag'],
        funcs['detect_pennant'],
        funcs['detect_head_and_shoulders'],
        funcs['detect_double_top'],
        funcs['detect_double_bottom'],
        funcs['smooth_prices'],
        funcs['compute_quality_score'],
        funcs['refine_peak'],
        funcs['refine_trough'],
    ]
    
    new_src = src
    for code in to_delete:
        new_src = new_src.replace(code, "")
        
    # Delete the config block by replacing the first occurrence of CONFIGURATION to GET THE WATCHLIST with imports
    import re
    config_pattern = r"# =============================================================================\s*#  CONFIGURATION.*?# =============================================================================\s*#  1\. GET THE WATCHLIST"
    
    imports = """# =============================================================================
#  DETECT ENGINE IMPORTS
# =============================================================================
from detectors.core import *
from detectors.cup_and_handle import detect_cup_and_handle
from detectors.bull_flag import detect_bull_flag
from detectors.bear_flag import detect_bear_flag
from detectors.pennant import detect_pennant
from detectors.head_and_shoulders import detect_head_and_shoulders
from detectors.double_top import detect_double_top
from detectors.double_bottom import detect_double_bottom

# =============================================================================
#  1. GET THE WATCHLIST"""

    new_src = re.sub(config_pattern, imports, new_src, flags=re.DOTALL)
    
    with open("pattern_scanner.py", "w", encoding="utf-8") as f:
        f.write(new_src)
        
    print("AST Extraction and Rewrite Complete.")

if __name__ == "__main__":
    run()
