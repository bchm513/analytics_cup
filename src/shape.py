"""
Team shape measures from one frame of tracking, for one team's outfield players.

All functions expect coordinates oriented so the team being measured defends the goal
at -x. src.loading.orient(df, match_id, team_id) does exactly that: it makes team_id
attack toward +x, so its own goal is at -x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _convex_hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of 2D points (monotone chain + shoelace)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def frame_shape(xy: np.ndarray) -> dict:
    """
    Shape of a set of outfield players (n x 2 array, team defending the goal at -x).

    width        spread across the pitch (y range)
    depth        distance from the deepest to the most advanced player (x range)
    area         convex hull area (m^2), a common compactness measure
    centroid_x/y average position
    line_height  x of the deepest outfield player (the last line), relative to the center
    """
    if len(xy) < 3:
        return dict(width=np.nan, depth=np.nan, area=np.nan, centroid_x=np.nan,
                    centroid_y=np.nan, line_height=np.nan, n_players=len(xy))
    return dict(
        width=float(np.ptp(xy[:, 1])),
        depth=float(np.ptp(xy[:, 0])),
        area=_convex_hull_area(xy),
        centroid_x=float(xy[:, 0].mean()),
        centroid_y=float(xy[:, 1].mean()),
        line_height=float(xy[:, 0].min()),
        n_players=len(xy),
    )


def team_shape(tracking: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Per-frame shape of one team's outfield players.

    `tracking` is long-format tracking (load_tracking, with_players=True) already
    oriented with orient(tracking, match_id, team_id), so team_id defends the goal at -x.
    """
    d = tracking[(tracking["team_id"] == team_id) & ~tracking["is_gk"].fillna(False).astype(bool)]
    rows = []
    for frame, g in d.groupby("frame", sort=True):
        r = frame_shape(g[["x", "y"]].to_numpy(dtype=float))
        r["frame"] = frame
        r["share_detected"] = float(g["is_detected"].fillna(False).astype(bool).mean())
        rows.append(r)
    return pd.DataFrame(rows)
