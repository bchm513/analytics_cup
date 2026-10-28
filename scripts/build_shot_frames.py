"""
Extract every shot's tracking frame, oriented so the shooting team attacks toward +x.

    python scripts/build_shot_frames.py

Writes data/processed/shots.parquet and data/processed/shot_frames.parquet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.loading import build_shot_frames, check_goals  # noqa: E402

shots, frames = build_shot_frames()
t = shots[shots["has_tracking"]]
goals = check_goals()

print(f"\n{len(shots)} shots ({int(shots['is_goal'].sum())} goals) from {shots['match_id'].nunique()} matches")
print(f"with a tracking frame:      {len(t)} ({len(t) / len(shots):.0%})")
print(f"orientation check passed:   {t['orientation_ok'].mean():.1%}")
print(f"keeper inferred (no GK id): {int(t['gk_inferred'].sum())}")
print(f"ball missing at shot frame: {t['ball_x'].isna().mean():.1%}")
print(f"defenders extrapolated:     {t['n_defenders_extrapolated'].sum() / t['n_defenders'].sum():.1%}")
print(f"matches where shot goals + own goals != score: {int((~goals['matches_score']).sum())} "
      f"of {len(goals)} (goals with no matching shot event)")
print(f"\nSaved to {PROCESSED_DIR}")
