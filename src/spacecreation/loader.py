"""Load everything the notebook already computed and cached, so the dashboard starts in seconds.

    from spacecreation import loader
    core = loader.init("path/to/analytics-cup-football")

Needs one full run of notebooks/space_creation.ipynb first (it writes data/cache/expected_positions.pkl and
data/outputs/runs_enriched.pkl, results_<id>.pkl, passes.csv).
"""
from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from . import core


class _NotebookUnpickler(pickle.Unpickler):
    """Objects pickled from the notebook point at __main__; resolve them in core instead."""

    def find_class(self, module, name):
        if module == "__main__":
            return getattr(core, name)
        return super().find_class(module, name)


def _attacking_half(runs: pd.DataFrame) -> pd.DataFrame:
    """Same labels as the notebook's taxonomy cell, for when only runs_full.pkl exists."""
    if "attacking_half" in runs:
        return runs
    dm = np.stack(runs.defence_map.values)
    tot = dm.sum(1)
    runs["opp_half_share"] = np.where(tot > 0, dm[:, core.COARSE.points[:, 0] > core.HALF_X].sum(1)
                                      / np.maximum(tot, 1e-12), np.nan)
    runs["created_opp_half"] = runs.created.clip(lower=0) * runs.opp_half_share.fillna(0)
    runs["attacking_half"] = np.where(runs.opp_half_share.notna(), runs.opp_half_share >= 0.5, runs.x_end > core.HALF_X)
    if "run_type" not in runs:
        runs["run_type"] = np.where(runs.kind == "carry", "carry (on ball)", runs.sk_event_subtype.fillna("unlabelled"))
    return runs


def init(root, match_ids=None, verbose=True):
    root = Path(root).resolve()
    core.ROOT = root
    core.DATA_DIR = root / "data" / "opendata" / "data"
    core.CACHE_DIR = root / "data" / "cache"
    core.OUT_DIR = root / "data" / "outputs"
    core.FIG_DIR = root / "figures"
    if not (core.DATA_DIR / "matches").is_dir():
        raise FileNotFoundError(f"No SkillCorner data at {core.DATA_DIR}")

    mids = list(match_ids) if match_ids else core.list_matches()
    core.match_ids = mids
    core.events_all = pd.concat([core.load_events(m) for m in mids], ignore_index=True)
    core.metas = {m: core.load_meta(m, core.events_all[core.events_all.match_id == m]) for m in mids}
    core.phases_all = {m: core.load_phases(m) for m in mids}
    core.players_all = pd.concat([core.metas[m].players.assign(match_id=m) for m in mids], ignore_index=True)
    core.team_names = {k: v for m in core.metas.values() for k, v in m.team_names.items()}
    mins = core.players_all.groupby("player_id").agg(name=("name", "first"), team_id=("team_id", "first"),
                                                     role=("role", lambda s: s.mode().iat[0]),
                                                     minutes_tip=("minutes_tip", "sum"), matches=("match_id", "nunique"))
    mins["team"] = mins.team_id.map(core.team_names)
    core.mins = mins
    core.value_surface = core.fit_value_surface(core.events_all)

    ep_path = core.CACHE_DIR / "expected_positions.pkl"
    if not ep_path.exists():
        raise FileNotFoundError(f"{ep_path} not found — run the notebook once first")
    with open(ep_path, "rb") as fh:
        cached = _NotebookUnpickler(fh).load()
    core.ep = cached["ep"] if isinstance(cached, dict) else cached

    out = core.OUT_DIR
    runs_path = out / "runs_enriched.pkl" if (out / "runs_enriched.pkl").exists() else out / "runs_full.pkl"
    with open(runs_path, "rb") as fh:
        core.RUNS = _attacking_half(_NotebookUnpickler(fh).load())
    core.PASSES = pd.read_csv(out / "passes.csv") if (out / "passes.csv").exists() else pd.DataFrame()
    if verbose:
        print(f"Loaded {len(mids)} matches, {len(core.RUNS):,} movements ({runs_path.name}), "
              f"{len(core.PASSES):,} passes")
    return core


class ContextCache:
    """Keep the last few matches' tracking in memory (each is ~50–100 MB)."""

    def __init__(self, size=3):
        self.size, self._d = size, OrderedDict()

    def get(self, mid):
        mid = int(mid)
        if mid in self._d:
            self._d.move_to_end(mid)
            return self._d[mid]
        ctx = core.build_ctx(mid)
        self._d[mid] = ctx
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return ctx
