import os
import re

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. Remove structural_cache=None from signature
    src = re.sub(r",\s*structural_cache=None", "", src)

    # 2. Remove the cache fast-path block
    if "Slow Path: Full Detection" in src:
        # Most files
        # It starts after `if len(prices) < ...:\n        return []\n`
        # and ends at `# Slow Path: Full Detection`
        
        # Find the return [] block
        start_match = re.search(r"if (?:n|len\(prices\)) < \d+:\n\s*return \[\]\n", src)
        if start_match:
            start_idx = start_match.end()
            end_idx = src.find("# Slow Path: Full Detection")
            if end_idx != -1:
                # Keep any vol_sma20 initialization that might be before Cache Hit
                # Wait, in pennant.py, vol_sma20 is defined between return [] and Cache Hit
                cache_hit_str = "# Fast Path: Cache Hit"
                cache_idx = src.find(cache_hit_str, start_idx, end_idx)
                if cache_idx != -1:
                    src = src[:cache_idx] + "\n    " + src[end_idx:]
                else:
                    src = src[:start_idx] + "\n    " + src[end_idx:]

    # 3. Remove the cache saving at the bottom
    save_pattern1 = r"\n\s*if structural_cache is not None and isinstance\(structural_cache, dict\):\n\s*structural_cache\[.*?\] =.*?\n"
    src = re.sub(save_pattern1, "\n", src)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(src)

for file in os.listdir("detectors"):
    if file.endswith(".py") and file != "core.py" and file != "__init__.py" and file != "cup_and_handle.py":
        process_file(os.path.join("detectors", file))

print("Cache removed from 6 files")
