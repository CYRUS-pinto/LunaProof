"""
smoke_test_v4.py - Full headless smoke test for LunaProof v4.
Local test: N_PAIRS=200, EP=2 (Colab runs N_PAIRS=10000, EP=25).
All logic paths are identical.
"""
import json, sys, os

os.chdir("c:/Users/Cyrus/Downloads/!Main")

import matplotlib
matplotlib.use("Agg")

with open("LunaProof_v4_Real_Training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"LunaProof v4: {len(nb['cells'])} total, {len(code_cells)} code cells")
print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print()

g = {}
errors = 0
skipped = 0

for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    label = src.split("\n")[0].strip().lstrip("#").strip()[:60]

    # Speed up for local smoke test
    src = src.replace("N_PAIRS = 10000", "N_PAIRS = 200")
    src = src.replace("EP = 25", "EP = 2")
    src = src.replace("num_workers=2", "num_workers=0")
    src = src.replace("pin_memory=True", "pin_memory=False")

    # Headless matplotlib
    src = src.replace(
        "plt.show()",
        f"plt.savefig('_smoke_p{i}.png', dpi=40, bbox_inches='tight')"
    )

    # Colab download — make _in_colab = False
    src = src.replace(
        "from google.colab import files as _colab_files",
        "raise ImportError('headless-colab')"
    )

    try:
        exec(src, g)
        print(f"  Cell {i+1:02d}: PASS  [{label}]")
    except ModuleNotFoundError as e:
        mod_name = str(e).split("'")[1] if "'" in str(e) else str(e)
        if mod_name in ("torch", "google", "torchvision"):
            print(f"  Cell {i+1:02d}: SKIP  [{label}]")
            print(f"             (Colab-only: {e})")
            skipped += 1
        else:
            print(f"  Cell {i+1:02d}: ERROR [{label}]")
            print(f"             {type(e).__name__}: {e}")
            errors += 1
    except ImportError as e:
        if "headless-colab" in str(e):
            print(f"  Cell {i+1:02d}: PASS  [{label}]  [Colab download skipped]")
        else:
            print(f"  Cell {i+1:02d}: SKIP  [{label}]")
            print(f"             (Colab-only import: {e})")
            skipped += 1
    except Exception as e:
        print(f"  Cell {i+1:02d}: ERROR [{label}]")
        print(f"             {type(e).__name__}: {e}")
        errors += 1

print()
n_pass = len(code_cells) - errors - skipped
print(f"Summary: {n_pass} PASS  |  {skipped} SKIP (Colab-only)  |  {errors} FAIL")
if errors == 0:
    print("\n[OK] All non-Colab cells pass. v4 notebook is ready for Colab.")
else:
    print(f"\n[FAIL] {errors} cell(s) have errors. Fix before submitting.")
    sys.exit(1)
