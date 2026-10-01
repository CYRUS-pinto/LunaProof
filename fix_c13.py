"""fix_c13.py — Patch the Cell 13 colab block to use _in_colab pattern."""
with open("build_v4_notebook.py", "r", encoding="utf-8") as f:
    content = f.read()

OLD = (
    "try:\n"
    "    from google.colab import files; files.download(ZN); print('[Colab] Browser download triggered.')\n"
    "except ImportError:\n"
    "    print(f'[Local] -> {os.path.abspath(ZN)}')\"\"\""
)

NEW = (
    "_in_colab = False\n"
    "try:\n"
    "    from google.colab import files as _colab_files\n"
    "    _in_colab = True\n"
    "except ImportError:\n"
    "    pass\n"
    "if _in_colab:\n"
    "    _colab_files.download(ZN)\n"
    "    print('[Colab] Browser download triggered.')\n"
    "else:\n"
    "    print(f'[Local] -> {os.path.abspath(ZN)}')\"\"\""
)

if OLD in content:
    content = content.replace(OLD, NEW)
    with open("build_v4_notebook.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("PATCHED OK")
else:
    # Try to find it
    idx = content.find("from google.colab import files; files.download(ZN)")
    print(f"NOT FOUND at expected location. Colab line found at: {idx}")
    # Show surrounding context
    if idx > 0:
        print(repr(content[idx-200:idx+200]))
