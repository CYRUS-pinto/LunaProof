"""
patch_builder_modules.py
Replaces REAL_DATA_SRC, CROSS_SRC and INFER_SRC in build_v4_notebook.py
with the definitive v4 versions that match local lunaproof/*.py files.
"""
import json, ast

# Read the actual v4 module files from disk
with open("lunaproof/real_data.py", "r", encoding="utf-8") as f:
    REAL_DATA_SRC = f.read()

with open("lunaproof/cross_dataset.py", "r", encoding="utf-8") as f:
    CROSS_SRC = f.read()

with open("lunaproof/inference.py", "r", encoding="utf-8") as f:
    INFER_SRC = f.read()

# Validate all three
for name, src in [("real_data", REAL_DATA_SRC), ("cross_dataset", CROSS_SRC), ("inference", INFER_SRC)]:
    try:
        ast.parse(src)
        print(f"  {name}: AST OK")
    except SyntaxError as e:
        print(f"  {name}: SYNTAX ERROR line {e.lineno}: {e.msg}")
        raise

# Read builder
with open("build_v4_notebook.py", "r", encoding="utf-8") as f:
    builder = f.read()

# Build the replacement blocks using json.dumps for safe embedding
RD_LINE  = f"REAL_DATA_SRC = {json.dumps(REAL_DATA_SRC)}"
CD_LINE  = f"CROSS_SRC = {json.dumps(CROSS_SRC)}"
INF_LINE = f"INFER_SRC = {json.dumps(INFER_SRC)}"

import re

# Replace each block — they start with the variable name and span multiple lines
def replace_block(text, varname, new_line):
    # Find start: "VARNAME = (" or "VARNAME = \""
    start = text.find(f"\n{varname} = ")
    if start == -1:
        print(f"  WARNING: {varname} not found in builder")
        return text
    # Find end: a blank line after the closing paren/quote
    # Look for the next top-level variable assignment or blank line after content
    chunk_start = start + 1  # skip leading \n
    # Find next variable assignment at column 0
    pattern = re.compile(r'\n(?=[A-Z_]+ =|\n# )', re.MULTILINE)
    m = pattern.search(text, start + len(varname) + 5)
    if m:
        chunk_end = m.start()
    else:
        chunk_end = len(text)
    old_block = text[chunk_start:chunk_end]
    print(f"  Replacing {varname} ({len(old_block)} chars -> {len(new_line)} chars)")
    return text[:chunk_start] + new_line + "\n" + text[chunk_end:]

builder = replace_block(builder, "REAL_DATA_SRC", RD_LINE)
builder = replace_block(builder, "CROSS_SRC", CD_LINE)
builder = replace_block(builder, "INFER_SRC", INF_LINE)

with open("build_v4_notebook.py", "w", encoding="utf-8") as f:
    f.write(builder)

print("\nBuilder modules patched. Rebuilding notebook...")
import subprocess, sys
result = subprocess.run([sys.executable, "build_v4_notebook.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
if result.returncode != 0:
    print("BUILD FAILED")
else:
    print("BUILD OK")
