"""
Pitch drawing for SkillCorner coordinates: meters, origin at the center of the pitch.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

LINE = "#9a9a9a"
HOME, AWAY, BALL = "#2388c8", "#e4572e", "#222222"


def draw_pitch(ax=None, length: float = 105, width: float = 68, half: bool = False,
               color: str = LINE, lw: float = 1.0):
    """
    Draw pitch markings. half=True shows only the attacking half (x >= 0), which is
    the view you want after orienting play toward +x.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.5) if half else (10.5, 6.8))
    L, W = length / 2, width / 2
    kw = dict(color=color, lw=lw, zorder=1)
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], **kw)
    ax.plot([0, 0], [-W, W], **kw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=1))
    box = dict(fill=False, ec=color, lw=lw, zorder=1)
    for s in (-1, 1):
        goal_line = s * L
        # penalty area, six-yard box and goal; each extends from the goal line toward the center
        ax.add_patch(Rectangle((goal_line if s < 0 else goal_line - 16.5, -20.16), 16.5, 40.32, **box))
        ax.add_patch(Rectangle((goal_line if s < 0 else goal_line - 5.5, -9.16), 5.5, 18.32, **box))
        ax.add_patch(Rectangle((goal_line - 1.5 if s < 0 else goal_line, -3.66), 1.5, 7.32, **box))
        ax.scatter([s * (L - 11)], [0], s=4, c=color, zorder=1)
        ax.add_patch(Arc((s * (L - 11), 0), 18.3, 18.3, theta1=(127 if s > 0 else -53),
                         theta2=(233 if s > 0 else 53), ec=color, lw=lw, zorder=1))
    ax.set_xlim((-2 if half else -L - 3), L + 3)
    ax.set_ylim(-W - 2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_frame(frame_df, ax=None, length: float = 105, width: float = 68, half: bool = False,
               color_by: str = "team", label: bool = False, title: str | None = None):
    """
    Plot one frame of long-format tracking (from load_tracking or shot_frames).

    Players are colored by `color_by` (e.g. 'team' or 'role'). Extrapolated players
    (is_detected False) are drawn hollow, so it's clear which positions are estimates.
    """
    ax = draw_pitch(ax, length, width, half)
    df = frame_df[frame_df["player_id"] != -1]
    ball = frame_df[frame_df["player_id"] == -1]
    groups = sorted(df[color_by].dropna().unique())
    palette = {"home": HOME, "away": AWAY, "shooter": "#00a82f", "teammate": HOME,
               "defender": AWAY, "defending_gk": "#7b2d26"}
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, g in enumerate(groups):
        d = df[df[color_by] == g]
        c = palette.get(g, fallback[i % len(fallback)])
        det = d["is_detected"].fillna(False).astype(bool)
        ax.scatter(d.loc[det, "x"], d.loc[det, "y"], s=70, c=c, ec="white", lw=0.8, zorder=3, label=str(g))
        ax.scatter(d.loc[~det, "x"], d.loc[~det, "y"], s=70, facecolors="none", ec=c, lw=1.6, zorder=3)
        if label and "short_name" in d:
            for r in d.itertuples():
                ax.annotate(str(r.short_name), (r.x, r.y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if len(ball):
        ax.scatter(ball["x"], ball["y"], s=30, c=BALL, zorder=4, label="ball")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=8,
              title="hollow = extrapolated", title_fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
