import re

with open('detectors/cup_and_handle.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Remove parameter
src = re.sub(r",\s*structural_cache=None", "", src)

# 2. Remove top cache block
start_match = re.search(r"\n\s*cache_hit = False", src)
end_match = re.search(r"\n\s*if not cache_hit:\n", src)

if start_match and end_match:
    src = src[:start_match.start()] + "\n" + src[end_match.end():]

# 3. Outdent the block and remove cache save block
lines = src.split('\n')
new_lines = []
outdent_mode = False
skip_save = False

for line in lines:
    if '# ── Step 1: Smooth prices and find peaks/troughs ──' in line:
        outdent_mode = True
        
    if 'candidates = []' in line:
        outdent_mode = False
        
    if 'if isinstance(structural_cache, dict) and dates is not None:' in line:
        skip_save = True
        continue
        
    if skip_save:
        if ']' in line:
            skip_save = False
        continue

    if outdent_mode and line.startswith('    '):
        # Outdent by 4 spaces
        new_lines.append(line[4:])
    else:
        new_lines.append(line)

with open('detectors/cup_and_handle.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("cup_and_handle fixed")
