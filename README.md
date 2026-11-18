# Space Creation in Build-Up Play
 
**PySport Analytics Cup 2.0, football track** · SkillCorner A-League 2024/25 tracking, events and body-pose data
 
Who makes space, who finds it, and who feeds it. This project measures the space that off-ball runs and carries open
for teammates, by comparing where defenders are with where they *should* be, and shows who could use that space and
where they should go. It has two notebooks and an interactive dashboard.
 
---
 
## What's in the project
 
```
analytics-cup-football/
├── notebooks/
│   ├── space_creation.ipynb   definitions and pipeline: everything the dashboard needs
│   └── space_analysis.ipynb   analysis on top of the defined spaces (reads saved results)
├── src/spacecreation/
│   ├── core.py                analysis code, generated from space_creation.ipynb (don't edit by hand)
│   └── loader.py              loads the notebook's cached outputs into core
├── app/
│   ├── dashboard.py           the interactive dashboard (Dash + Plotly)
│   └── assets/                drawing and playback in the browser (space.js) and styling (style.css)
├── scripts/
│   └── extract_core.py        regenerates src/spacecreation/core.py from the notebook
├── data/
│   ├── opendata/              SkillCorner open-data repo (cloned; not committed)
│   ├── cache/                 parsed tracking, expected-position baseline, pose (not committed)
│   └── outputs/               per-match results, runs_enriched.pkl, passes.csv, tables (not committed)
├── figures/                   case studies, showcase, review GIFs, xThreat comparison
├── docs/  tests/
├── requirements.txt
└── LICENSE.md
```
 
---
 
## Setup (once)
 
### 1. Data
 
Clone SkillCorner's open data into `data/opendata` and fetch the tracking files with Git LFS:
 
```
cd "C:\path\to\analytics-cup-football\data"
git clone https://github.com/SkillCorner/opendata.git
cd opendata
git lfs install
git lfs pull
```
 
Each `matches/<id>/<id>_tracking_extrapolated.jsonl` should be around 100 MB. A file of about 130 bytes is only an LFS
pointer: run `git lfs pull` again.
 
**Body pose (optional, two matches):** download `raw/1925299.jsonl.zip` and `raw/1996435.jsonl.zip` from the
`SkillCorner/opendata-bodypose` dataset on Hugging Face into `data/opendata/data/bodypose/raw/` (about 600 MB each;
leave them zipped). Without them the analysis notebook falls back to the 12-second sample.
 
### 2. Environment
 
Use the Anaconda environment (Anaconda Prompt, or select the `anaconda3` interpreter in VS Code) and install the extras:
 
```
pip install -r requirements.txt
```
 
or, at minimum:
 
```
pip install dash plotly nbformat nbconvert huggingface_hub
```
 
numpy, pandas, scipy, scikit-learn and matplotlib come with Anaconda.
 
---
 
## Running it
 
From the Anaconda Prompt, in the project folder (keep the quotes; the path has spaces):
 
```
cd "C:\path\to\analytics-cup-football"
```
 
**1. Definitions notebook** (all the scoring; 30–45 min on a rescore, a few minutes when cached)
 
```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\space_creation.ipynb
```
 
**2. Sync the dashboard's code with the notebook**
 
```
python scripts\extract_core.py
```
 
**3. Analysis notebook** (reads saved results; nothing is recomputed)
 
```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\space_analysis.ipynb
```
 
**4. Dashboard**
 
```
python app\dashboard.py
```
 
Open http://127.0.0.1:8050 and stop it with Ctrl+C. Options: `--port 8060`, `--root path\to\analytics-cup-football`.
 
Running cell by cell in VS Code or Jupyter (`jupyter lab`) works the same way, in this order. Add `--allow-errors` to
the nbconvert commands to keep going past a failing cell.
 
---
 
## The two notebooks
 
**`space_creation.ipynb` (definitions)**
 
| Section | What it does |
|---|---|
| 1 | Settings and caching switches |
| 2–3 | Load matches, tracking and events; data audit (on-camera detection) |
| 4 | Value surface fitted on SkillCorner xThreat |
| 5 | Pitch control (Fernández & Bornn) |
| 6 | Expected positions: each role's offset from his team's block, given the ball and phase, fitted without the match being scored, never within 4 m of a teammate |
| 7 | Movements: off-ball runs (speed ≥ `RUN_MIN_SPEED`) and carries; keepers excluded |
| 8 | Space created: defenders pulled out of position by an *abnormal* movement (one that stands out from the team's own shift), whose deviation grows, whose spot isn't still held; never the keeper, never space behind the keeper |
| 9–11 | Occupier, feeder and space-behind-the-line metrics; the per-match pipeline |
| 12 | Taxonomy; saves `data/outputs/runs_enriched.pkl` |
| 13 | Shared helpers (definitions only): beneficiaries, scenes, team plays, lens overlap |
 
**`space_analysis.ipynb` (analysis on top)**
 
| Section | What it does |
|---|---|
| 1 | Setup and review settings (`REVIEW_NAME`, `REBUILD_REVIEW`) |
| 2 | Designed vs real space; what works against which defensive phase |
| 3 | Body orientation (pose) |
| 4 | Space created vs SkillCorner run xThreat: agreement vs pure decoys (`figures/xthreat/`) |
| 5 | Validation against SkillCorner Game Intelligence |
| 6 | Leaderboards and team comparison |
| 7 | Case studies, showcase and "more like this" (`figures/cases/`, `figures/showcase/`) |
| 8 | Blind review sample and scoring (`figures/<REVIEW_NAME>/review.html`) |
 
---
 
## The dashboard
 
- **Two tabs.** *Team plays* groups every movement by one team in the same moment of play (until the first pass);
  *Individual movements* lists runs and carries one by one. Pick a match (the two pose games are listed first), filter,
  and click any row.
- **Playback** runs in the browser: play, pause, step, scrub, speed, and jump to start / peak / prime / pass.
- **Layers:** who controls the pitch, space opened for teammates, where teammates should go (best suggestion; toggle to
  show all), defenders pulled out of position (expected spot → actual), lines from movers to the defenders they pull,
  prime space, designed space, vacated-spot cover, every defender's expected spot. Separate toggles for attacking
  players, defending players, ball and movement paths.
- **Click a player:** attackers show where they can get before the quickest defender; defenders show how far they are
  from their expected spot. Click again to clear.
- The table under the pitch shows who should go where at the current frame.
---
 
## Settings and caching
 
The main settings are in Section 1 of `space_creation.ipynb`.
 
| Setting | Current | Meaning |
|---|---|---|
| `RUN_MIN_SPEED` | 2.24 m/s (5 mph) | speed to count as running (m/s; km/h ÷ 3.6) |
| `ABNORMAL_MIN_M` | 3.0 m | movement relative to the team's own shift needed to count as abnormal |
| `DEV_GROWTH_MIN` | 1.0 m | how much further from his expected spot a defender must get |
| `OCCUPIED_R` | 5.0 m | a pulled defender's spot is still held if a teammate is this close |
| `BEN_EXCLUDE_GROUPS` | GK, CB | never suggested as the player to run into the space |
| `TEAMPLAY_JOIN_S` | 1.0 s | a movement joins a team play if it starts within this of the play's end |
| `PIPELINE_VERSION` | v6 | results are cached per version |
| `BASELINE_VERSION` | shape-v1 | the expected-position model is cached per version |
 
The beneficiary weights (`W_SPACE`, `W_VALUE`, `W_FREE`, `W_REACH`, `OPEN_SCALE_M`) are in Section 13. They only
affect the visuals and need no rescore.
 
**What to rerun after a change**
 
| Changed | Do this |
|---|---|
| Visuals, beneficiary weights, taxonomy | Run the notebook (cached, minutes), `extract_core.py`, restart the dashboard |
| Run speed, drag rules, scoring code | `RERUN_PIPELINE = True` once (or a new `PIPELINE_VERSION`), then as above |
| Expected positions, roles, phases, match list | `REBUILD_BASELINE = True` and `RERUN_PIPELINE = True` once |
| New blind-review sample | A new `REVIEW_NAME` in the analysis notebook (keeps the old answers) |
 
Set the switches back to `False` afterwards. Restarting the kernel is safe: everything slow is cached on disk.
`MATCH_IDS = [1925299, 1996435]` runs only the pose games as a quick test; the baseline is then fitted on one other
game only, so use all 20 matches for real results.
 
---
 
## Troubleshooting
 
| Problem | Fix |
|---|---|
| `UnicodeDecodeError ... 'charmap'` | Run with Anaconda's Python, or `python -X utf8 ...` |
| `ModuleNotFoundError` (dash, plotly) | `pip install dash plotly` in the environment you run with |
| Dashboard list is empty | Pick a match that was scored; untick "Pulled a defender"; read the count line under the list |
| Dashboard shows old behaviour | `python scripts\extract_core.py`, restart, hard-refresh the browser (Ctrl+F5) |
| A notebook can't find the data | Run it from `notebooks/` (nbconvert does this automatically) |
| Tracking files ~130 bytes | Git LFS pointers: `git lfs pull` in `data/opendata` |
 
---
 
## Data and references
 
- SkillCorner open data, A-League 2024/25: tracking (broadcast, 10 fps), Dynamic Events, phases of play, body pose.
- Fernández & Bornn (2018), *Wide Open Spaces*: pitch control and space value.
- Mendes-Neves et al. (2025), *A Neighbor-based Approach to Pitch Ownership Models in Soccer*: uncertainty and model
  disagreement behind the defensive line.
Keep these out of git (`.gitignore`): `data/opendata/`, `data/cache/`, `data/outputs/`, `figures/review*/`.