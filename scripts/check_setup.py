"""
Check that the environment and SkillCorner data are ready.

    python scripts/check_setup.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ok = True


def report(passed: bool, msg: str, fix: str = ""):
    global ok
    ok &= passed
    print(f"[{'ok' if passed else '!!'}] {msg}")
    if not passed and fix:
        print(f"     -> {fix}")


report(sys.version_info[:2] >= (3, 10), f"Python {sys.version.split()[0]}",
       "Use Python 3.10+ (the README uses 3.12).")
report(sys.prefix != sys.base_prefix, "Running inside a virtual environment",
       "Activate it: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate")

for pkg in ("pandas", "numpy", "matplotlib", "pyarrow"):
    try:
        importlib.import_module(pkg)
        report(True, f"{pkg} installed")
    except ImportError:
        report(False, f"{pkg} missing", "python -m pip install -r requirements.txt")

from src.config import find_data_dir  # noqa: E402

try:
    data_dir = find_data_dir()
    report(True, f"SkillCorner data found at {data_dir}")
except FileNotFoundError as e:
    report(False, "SkillCorner data not found", str(e).splitlines()[-1])
    sys.exit(1)

from src.loading import list_matches  # noqa: E402

ids = list_matches()
report(len(ids) > 0, f"{len(ids)} matches listed in matches.json")

pointers, missing = [], []
for mid in ids:
    f = data_dir / "matches" / str(mid) / f"{mid}_tracking_extrapolated.jsonl"
    if not f.exists():
        missing.append(mid)
    elif f.stat().st_size < 1024 and f.read_text(errors="ignore").startswith("version https://git-lfs"):
        pointers.append(mid)
report(not missing, f"tracking files present for {len(ids) - len(missing)}/{len(ids)} matches")
report(not pointers, "tracking files are real data, not Git LFS pointers",
       "Run `git lfs install` then `git lfs pull` inside the opendata folder.")

print("\nAll good, ready to go." if ok else "\nFix the items marked !! above.")
sys.exit(0 if ok else 1)
