"""
Paths used across the project.

The SkillCorner data folder (the one containing matches.json) is found in this order:
  1. the SKILLCORNER_DATA environment variable
  2. data/opendata/data inside this repository (the default setup in the README)
  3. a sibling folder named skillcorner-opendata, i.e. ../skillcorner-opendata/data
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"


def _candidates() -> list[Path]:
    env = os.environ.get("SKILLCORNER_DATA")
    paths = [Path(env)] if env else []
    return paths + [ROOT / "data" / "opendata" / "data",
                    ROOT.parent / "skillcorner-opendata" / "data"]


def find_data_dir() -> Path:
    for c in _candidates():
        if (c / "matches.json").exists():
            return c
    tried = "\n  ".join(str(c) for c in _candidates())
    raise FileNotFoundError(
        "SkillCorner data not found. Looked for matches.json in:\n  " + tried +
        "\nSee README.md > Setup, or set the SKILLCORNER_DATA environment variable."
    )


def data_available() -> bool:
    try:
        find_data_dir()
        return True
    except FileNotFoundError:
        return False
