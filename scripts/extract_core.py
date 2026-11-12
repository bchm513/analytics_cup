"""Usage (from the project folder):  python scripts/extract_core.py

Build src/spacecreation/core.py from the notebook: every import, class, function and plain constant, in notebook
order, with nothing that runs analysis. Re-run this after changing the notebook so the dashboard uses the same code."""
import ast, sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "space_creation.ipynb"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "src" / "spacecreation" / "core.py"
nb = nbformat.read(NB, 4)
SKIP_NAMES = {"MATCH_IDS", "ROOT", "DATA_DIR", "CACHE_DIR", "FIG_DIR", "OUT_DIR", "REVIEW_DIR", "SHEET", "BUILD_REVIEW",
              "POSE_FILES", "POSE_SAMPLE", "SHOWCASE_DIR", "REVIEW", "SEED", "R", "RUNS", "OCC", "PASSES", "LINE",
              "LINE_DEF", "LAG", "POSE", "SHOW", "TAXONOMY", "DIR_MEDOIDS", "RUNTYPE_ACC", "EMPTY", "POSE_BY",
              "POSE_COLS_RUNS", "POSE_COLS_PASSES", "REBUILD_BASELINE", "RERUN_PIPELINE", "RERUN_MATCHES",
              "REBUILD_REVIEW", "REVIEW_NAME", "N_REVIEW"}
ALLOWED_CALLS = {"Grid", "range", "set", "dict", "tuple", "list"}
consts, out, seen_imports = set(), [], set()


def safe_value(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else None
            if name not in ALLOWED_CALLS:
                return False
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in consts and n.id not in ALLOWED_CALLS and n.id not in {"np", "True", "False", "None"}:
                return False
    return True


for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = "\n".join(l for l in c.source.splitlines() if not l.strip().startswith("%"))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if seg not in seen_imports and not (isinstance(node, ast.ImportFrom) and node.module == "huggingface_hub"):
                seen_imports.add(seg); out.append(seg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # decorators are part of the segment only from their line; include them explicitly
            deco = "".join("@" + ast.get_source_segment(src, d) + "\n" for d in node.decorator_list)
            out.append(deco + seg)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for t in node.targets:
                if isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
            if names and all(n.isupper() or n.startswith("_") is False and n.upper() == n for n in names) \
                    and not set(names) & SKIP_NAMES and safe_value(node.value):
                out.append(seg); consts.update(names)
header = '''"""Space-creation analysis core — generated from notebooks/space_creation.ipynb by scripts/extract_core.py.
Do not edit by hand: change the notebook and regenerate. Module-level data (metas, events_all, mins, ep, RUNS, ...)
is filled in by spacecreation.loader.init()."""
'''
body = "\n\n".join(out)
fut = "from __future__ import annotations"
body = body.replace(fut + "\n", "").replace(fut, "")
OUT.write_text(header + fut + "\n\n" + body + "\n", encoding="utf-8")
print(len(out), "blocks;", len(consts), "constants")
