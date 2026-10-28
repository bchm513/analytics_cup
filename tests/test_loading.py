"""
Smoke tests: run with `python -m pytest`. Skipped automatically if the SkillCorner data
isn't set up yet.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import data_available
from src.shape import frame_shape

needs_data = pytest.mark.skipif(not data_available(), reason="SkillCorner data not found")


def test_frame_shape_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=float)
    s = frame_shape(sq)
    assert s["area"] == pytest.approx(100)
    assert s["width"] == 10 and s["depth"] == 10 and s["line_height"] == 0


@needs_data
def test_matches_and_metadata():
    from src.loading import list_matches, load_match_meta
    ids = list_matches()
    assert len(ids) > 0
    meta = load_match_meta(ids[0])
    assert set(meta["home_direction"]) == {1, 2}
    assert meta["players"]["is_gk"].sum() >= 2


@needs_data
def test_tracking_loads():
    from src.loading import BALL_ID, list_matches, load_tracking
    trk = load_tracking(list_matches()[0], step=100)
    assert not trk.empty
    assert {"frame", "player_id", "x", "y", "is_detected", "team", "role"} <= set(trk.columns)
    assert (trk["player_id"] == BALL_ID).any()


@needs_data
def test_shot_frames_oriented():
    from src.loading import extract_shot_frames, list_matches
    shots, frames = extract_shot_frames(list_matches()[0])
    assert len(shots) > 0
    t = shots[shots["has_tracking"]]
    # the defending keeper should almost always be in the half being attacked
    assert t["orientation_ok"].mean() > 0.9
    assert set(frames["role"]) >= {"shooter", "defender", "defending_gk"}
