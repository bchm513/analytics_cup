# PySport Analytics Cup 2.0: Football

*USA edition, SkillCorner A-League 2024/25 data. Working title; project in progress.*

> The final README will be at most 1,000 words with at most two figures.

## Data

This project uses only the data provided for the Analytics Cup:
[SkillCorner's A-League 2024/25 open data](https://github.com/SkillCorner/opendata).
Data courtesy of SkillCorner. No data is included in this repository.

## Setup

Requires Python 3.12 and [Git LFS](https://git-lfs.com/), since the tracking files are stored with LFS.

```bash
git clone <this-repo-url>
cd <repo-folder>

# SkillCorner data, cloned into data/ (ignored by git)
git lfs install
git clone https://github.com/SkillCorner/opendata.git data/opendata

# Environment
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# Check everything is in place
python scripts/check_setup.py
```

The data can also live outside the repository: set the `SKILLCORNER_DATA` environment variable
to its `data` folder (the one containing `matches.json`), or place the clone next to this
repository as `skillcorner-opendata`.

## Run

```bash
python scripts/build_shot_frames.py   # oriented tracking frames for every shot -> data/processed/
python -m pytest                      # smoke tests
```

## Project structure

```
src/
  config.py       paths and data location
  loading.py      matches, metadata, tracking, events, phases, aggregates, shot frames
  pitch.py        pitch and frame plotting
  shape.py        team shape: width, depth, compactness, line height
scripts/          runnable steps
notebooks/        exploration (01-05) and sandbox/ for experiments
tests/            smoke tests
```

## License

MIT, see [LICENSE.md](LICENSE.md).
