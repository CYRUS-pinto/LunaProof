"""smoke_test.py — Run all notebook cells locally to confirm zero errors"""
import json, sys, os

os.chdir("c:/Users/Cyrus/Downloads/!Main")

import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed

if __name__ == "__main__":
    with open("LunaProof_Unified_Master_v3.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"Notebook: {len(nb['cells'])} total cells ({len(code_cells)} code)")
    print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
    print()

    g = {}
    errors = 0

    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        label = src.split("\n")[0].strip().lstrip("#").strip()[:55]

        if i == 11:  # Cell 10: plt.show() — skip interactive display, still run save
            src = src.replace("plt.show()", "# plt.show()  [headless]")
        if i == 12:  # Cell 11: files.download — will gracefully except
            pass

        try:
            exec(src, g)
            print(f"  Cell {i+1:02d}: EXEC OK  [{label}]")
        except Exception as e:
            print(f"  Cell {i+1:02d}: EXEC ERROR  [{label}]")
            print(f"           {type(e).__name__}: {e}")
            errors += 1

    print()
    if errors == 0:
        print(f"ALL {len(code_cells)} CELLS EXECUTED SUCCESSFULLY — Zero errors")
    else:
        print(f"FAILED: {errors} cell(s) raised errors")
        sys.exit(1)

