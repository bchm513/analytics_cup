"""
Loading utilities for SkillCorner's A-League 2024/25 open data.

Everything here reads only the files shipped in SkillCorner's opendata repository.
Coordinates are in meters with the origin at the center of the pitch; x runs along
the long side, y along the short side.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, find_data_dir

BALL_ID = -1                       # player_id used for the ball in long-format tables
FPS = 10                           # tracking, events and phases all run at 10 frames/second
_FRAME_RE = re.compile(r'"frame":\s*(\d+)')


# ---------------------------------------------------------------- matches and metadata

def _match_dir(match_id: int) -> Path:
    return find_data_dir() / "matches" / str(match_id)


def list_matches() -> list[int]:
    """All match ids in the dataset."""
    with open(find_data_dir() / "matches.json", encoding="utf-8") as f:
        return [m["id"] for m in json.load(f)]


def match_table() -> pd.DataFrame:
    """One row per match: date, teams, score, pitch size."""
    rows = []
    for mid in list_matches():
        meta = load_match_meta(mid)
        rows.append({k: meta[k] for k in ("match_id", "date", "home_team", "away_team",
                                          "home_score", "away_score", "pitch_length", "pitch_width")})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=32)
def load_match_meta(match_id: int) -> dict:
    """
    Pitch size, teams, score, attacking direction per period and a players table.

    home_direction maps period -> +1 if the home team attacks toward +x in that period,
    -1 otherwise ('left_to_right' = attacking toward +x, checked against keeper positions).
    """
    with open(_match_dir(match_id) / f"{match_id}_match.json", encoding="utf-8") as f:
        m = json.load(f)

    players = pd.DataFrame([{
        "player_id": p["id"],
        "team_id": p["team_id"],
        "short_name": p["short_name"],
        "number": p["number"],
        "role": p["player_role"]["acronym"],
        "position_group": p["player_role"].get("position_group"),
        "minutes_played": ((p.get("playing_time") or {}).get("total") or {}).get("minutes_played"),
        "goals": p.get("goal", 0),
        "own_goals": p.get("own_goal", 0),
    } for p in m["players"]])
    players["is_gk"] = players["role"].eq("GK")
    players["team"] = np.where(players["team_id"].eq(m["home_team"]["id"]), "home", "away")

    sides = m["home_team_side"]
    home_direction = {i + 1: (1 if s == "left_to_right" else -1) for i, s in enumerate(sides)}

    return {
        "match_id": m["id"],
        "date": m["date_time"],
        "pitch_length": m["pitch_length"],
        "pitch_width": m["pitch_width"],
        "home_team_id": m["home_team"]["id"],
        "away_team_id": m["away_team"]["id"],
        "home_team": m["home_team"]["short_name"],
        "away_team": m["away_team"]["short_name"],
        "home_score": m["home_team_score"],
        "away_score": m["away_team_score"],
        "home_direction": home_direction,
        # own goals credited to each side (scored by the opponent's players)
        "home_own_goals_for": int(players.loc[players["team"] == "away", "own_goals"].sum()),
        "away_own_goals_for": int(players.loc[players["team"] == "home", "own_goals"].sum()),
        "players": players,
    }


def attack_direction(meta: dict, team_id: int, period: int) -> int:
    """+1 if team_id attacks toward +x in this period, -1 otherwise."""
    d = meta["home_direction"][int(period)]
    return d if team_id == meta["home_team_id"] else -d


# ---------------------------------------------------------------- tracking

def iter_tracking(match_id: int, frames: set[int] | None = None, step: int = 1,
                  periods: tuple[int, ...] | None = None):
    """
    Stream raw tracking frames (dicts) from a match's .jsonl file.

    frames   only parse these frame numbers (fast: other lines are skipped unparsed)
    step     keep every `step`-th frame, e.g. step=10 gives one frame per second
    periods  keep only these periods; frames before kick-off (period None) are dropped
    """
    path = _match_dir(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if frames is not None or step > 1:
                hit = _FRAME_RE.search(line)
                if not hit:
                    continue
                n = int(hit.group(1))
                if frames is not None and n not in frames:
                    continue
                if step > 1 and n % step:
                    continue
            fr = json.loads(line)
            if fr.get("period") is None or (periods and fr["period"] not in periods):
                continue
            yield fr


def frames_to_long(frame_iter) -> pd.DataFrame:
    """
    Flatten tracking frames to one row per object per frame.
    The ball is included with player_id = BALL_ID and its height in z.
    """
    rows = []
    for fr in frame_iter:
        f, per, ts = fr["frame"], fr.get("period"), fr.get("timestamp")
        group = (fr.get("possession") or {}).get("group")
        for p in fr.get("player_data") or []:
            rows.append((f, per, ts, group, p["player_id"], p["x"], p["y"], np.nan, p.get("is_detected")))
        b = fr.get("ball_data") or {}
        if b.get("x") is not None:
            rows.append((f, per, ts, group, BALL_ID, b["x"], b["y"], b.get("z"), b.get("is_detected")))
    df = pd.DataFrame(rows, columns=["frame", "period", "timestamp", "possession_group",
                                     "player_id", "x", "y", "z", "is_detected"])
    df["is_detected"] = df["is_detected"].astype("boolean")
    return df


def load_tracking(match_id: int, frames: set[int] | None = None, step: int = 1,
                  periods: tuple[int, ...] | None = None, with_players: bool = True) -> pd.DataFrame:
    """
    Long-format tracking for a match. A full match at step=1 is ~1.3M rows and takes
    a little while to parse; use step=5 or step=10 for exploration.

    with_players=True joins team, role and name from the match metadata.
    """
    df = frames_to_long(iter_tracking(match_id, frames, step, periods))
    if with_players and not df.empty:
        players = load_match_meta(match_id)["players"]
        df = df.merge(players[["player_id", "team_id", "team", "short_name", "role", "is_gk"]],
                      on="player_id", how="left")
        df.loc[df["player_id"] == BALL_ID, ["team", "role"]] = "ball"
    return df


def orient(df: pd.DataFrame, match_id: int, team_id: int, period_col: str = "period") -> pd.DataFrame:
    """
    Rotate coordinates 180 degrees where needed so `team_id` always attacks toward +x.
    (Rotating rather than mirroring keeps left and right consistent.)
    """
    meta = load_match_meta(match_id)
    out = df.copy()
    sign = out[period_col].map(lambda p: attack_direction(meta, team_id, p))
    out["x"] = out["x"] * sign
    out["y"] = out["y"] * sign
    return out


# ---------------------------------------------------------------- events, phases, aggregates

def load_events(match_id: int) -> pd.DataFrame:
    """
    SkillCorner Game Intelligence dynamic events. Four event types:
    player_possession, passing_option, off_ball_run, on_ball_engagement.
    """
    return pd.read_csv(_match_dir(match_id) / f"{match_id}_dynamic_events.csv", low_memory=False)


def load_phases(match_id: int) -> pd.DataFrame:
    """Phases of play, with in- and out-of-possession phase types (e.g. low_block, high_block)."""
    return pd.read_csv(_match_dir(match_id) / f"{match_id}_phases_of_play.csv")


def load_all(loader, match_ids: list[int] | None = None) -> pd.DataFrame:
    """Run load_events or load_phases for every match and stack the results."""
    return pd.concat([loader(m) for m in (match_ids or list_matches())], ignore_index=True)


def list_aggregates() -> list[Path]:
    """Season-level aggregate CSVs (physical, off-ball runs, passing), if present."""
    folder = find_data_dir() / "aggregates"
    return sorted(folder.glob("*.csv")) if folder.exists() else []


def load_aggregate(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------- shots

def load_shots(match_id: int) -> pd.DataFrame:
    """
    Shots = player possessions that end in a shot; the shot is taken at frame_end.

    lead_to_goal is True for every shot in a phase that ends in a goal (a saved shot
    followed by a scored rebound are both flagged), so is_goal keeps only the last
    flagged shot per team and phase of play.
    """
    ev = load_events(match_id)
    s = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "shot")]
    keep = ["event_id", "match_id", "period", "frame_start", "frame_end", "minute_start",
            "second_start", "phase_index", "team_id", "team_shortname", "player_id", "player_name",
            "player_position", "lead_to_goal", "game_state", "team_in_possession_phase_type",
            "overall_pressure_end", "possession_epv_for_end", "possession_epv_against_end"]
    s = s[[c for c in keep if c in s.columns]].copy()
    s["shot_frame"] = s["frame_end"].astype(int)
    flagged = s[s["lead_to_goal"].fillna(False).astype(bool)]
    scored = flagged.sort_values("frame_end").groupby(["team_id", "phase_index"]).tail(1).index
    s["is_goal"] = s.index.isin(scored)
    s["shot_id"] = s["match_id"].astype(str) + "_" + s["event_id"].astype(str)
    return s.reset_index(drop=True)


def extract_shot_frames(match_id: int, max_frame_gap: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every shot, pull its tracking frame, label roles (shooter, teammate, defender,
    defending_gk, ball) and orient so the shooting team attacks toward +x.

    If the exact frame is missing, the nearest frame within `max_frame_gap` is used and
    the gap recorded. Returns (shots, shot_frames).
    """
    meta = load_match_meta(match_id)
    shots = load_shots(match_id)
    if shots.empty:
        return shots, pd.DataFrame()

    wanted = {f + d for f in shots["shot_frame"] for d in range(-max_frame_gap, max_frame_gap + 1)}
    trk = load_tracking(match_id, frames=wanted, with_players=False)
    available = np.sort(trk["frame"].unique()) if not trk.empty else np.array([], dtype=int)

    players = meta["players"].set_index("player_id")
    half_len = meta["pitch_length"] / 2

    shot_rows, frame_parts = [], []
    for s in shots.itertuples(index=False):
        row = s._asdict()
        gap = None
        if available.size:
            i = int(np.abs(available - s.shot_frame).argmin())
            gap = int(available[i] - s.shot_frame)
        if gap is None or abs(gap) > max_frame_gap:
            row.update(tracking_frame=None, frame_gap=None, has_tracking=False)
            shot_rows.append(row)
            continue
        used = s.shot_frame + gap

        fr = trk[trk["frame"] == used].copy()
        sign = attack_direction(meta, s.team_id, s.period)
        fr["x"] = fr["x"] * sign
        fr["y"] = fr["y"] * sign
        fr["team_id"] = fr["player_id"].map(players["team_id"])
        fr["is_gk"] = fr["player_id"].map(players["is_gk"]).fillna(False).astype(bool)
        fr["role"] = np.select(
            [fr["player_id"].eq(BALL_ID), fr["player_id"].eq(s.player_id),
             fr["team_id"].eq(s.team_id), fr["is_gk"]],
            ["ball", "shooter", "teammate", "defending_gk"], default="defender")

        # A substitute keeper is listed with role SUB, so fall back to the defender
        # nearest the goal center when no keeper is identified.
        gk_inferred = False
        if not fr["role"].eq("defending_gk").any():
            d = fr[fr["role"] == "defender"]
            if not d.empty:
                fr.loc[np.hypot(d["x"] - half_len, d["y"]).idxmin(), "role"] = "defending_gk"
                gk_inferred = True

        def first(role, col):
            v = fr.loc[fr["role"] == role, col]
            return float(v.iloc[0]) if len(v) else np.nan

        defenders = fr["role"] == "defender"
        row.update(
            tracking_frame=used, frame_gap=gap, has_tracking=True, gk_inferred=gk_inferred,
            gk_x=first("defending_gk", "x"),
            orientation_ok=bool(first("defending_gk", "x") > 0),   # keeper should be in the attacked half
            ball_x=first("ball", "x"), ball_y=first("ball", "y"),
            shooter_x=first("shooter", "x"), shooter_y=first("shooter", "y"),
            n_defenders=int(defenders.sum()),
            n_defenders_extrapolated=int((defenders & ~fr["is_detected"].fillna(False).astype(bool)).sum()),
        )
        shot_rows.append(row)
        fr.insert(0, "shot_id", s.shot_id)
        frame_parts.append(fr)

    shots_out = pd.DataFrame(shot_rows)
    shots_out["pitch_length"] = meta["pitch_length"]
    shots_out["pitch_width"] = meta["pitch_width"]
    frames_out = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    return shots_out, frames_out


def build_shot_frames(match_ids: list[int] | None = None, save: bool = True,
                      verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shot frames for every match; saved to data/processed/ as parquet."""
    all_shots, all_frames = [], []
    for mid in match_ids or list_matches():
        shots, frames = extract_shot_frames(mid)
        all_shots.append(shots)
        all_frames.append(frames)
        if verbose:
            print(f"  {mid}: {len(shots)} shots")
    shots = pd.concat(all_shots, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        shots.to_parquet(PROCESSED_DIR / "shots.parquet", index=False)
        frames.to_parquet(PROCESSED_DIR / "shot_frames.parquet", index=False)
    return shots, frames


def check_goals(match_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Goals from shot events plus own goals (from match.json) vs the final score.
    A mismatch usually means a goal with no matching shot event in the dynamic events.
    """
    rows = []
    for mid in match_ids or list_matches():
        meta, shots = load_match_meta(mid), load_shots(mid)
        g = shots.groupby("team_id")["is_goal"].sum()
        rows.append({"match_id": mid,
                     "home_shot_goals": int(g.get(meta["home_team_id"], 0)),
                     "home_own_goals_for": meta["home_own_goals_for"],
                     "home_score": meta["home_score"],
                     "away_shot_goals": int(g.get(meta["away_team_id"], 0)),
                     "away_own_goals_for": meta["away_own_goals_for"],
                     "away_score": meta["away_score"]})
    df = pd.DataFrame(rows)
    df["matches_score"] = (
        (df.home_shot_goals + df.home_own_goals_for == df.home_score) &
        (df.away_shot_goals + df.away_own_goals_for == df.away_score))
    return df
