# Space Creation Explorer (dashboard)

An interactive version of the notebook's visuals: pick a play, play / pause / scrub it frame by frame, switch layers
on and off, and click players.

## Setup (once)

1. Run `notebooks/space_creation.ipynb` top to bottom once. The dashboard reads what it caches:
   `data/cache/expected_positions.pkl`, `data/cache/<id>_tracking.npz`, `data/outputs/runs_enriched.pkl`,
   `data/outputs/passes.csv`.
2. Install the two extra packages:
   ```
   pip install dash plotly
   ```

## Run

From the `analytics-cup-football` folder:
```
python app/dashboard.py
```
Open http://127.0.0.1:8050. Options: `--port 8060`, `--root path/to/analytics-cup-football`, `--debug`.

## How it stays fast

When you pick a play, the server computes every frame once (usually 1–5 seconds; a small spinner shows next to the
title) and sends the whole play to the browser. Playing, pausing, scrubbing, switching layers and following a player
then all run in the browser (`assets/space.js`), so there's no wait per frame. Clicking an attacker fetches his
"where can I get to first" map for the whole play in one short request.

## What you can do

- **Pick a play:** choose a match, filter movements (off-ball runs / carries, opponent's half, pulled a defender,
  build-up phases) and click any cell in a row. Plays are sorted by space created in the opponent's half.
- **Jump to any moment:** type a clock like `P1 12:34` and press Go for free exploration (no movement selected).
- **Play / pause / step:** ▶ plays the play; ⏮ ⏭ step one frame; the speed menu sets playback speed; the buttons
  `start`, `peak`, `prime`, `pass` jump to the key moments, which are also marked on the slider.
- **Layers:** pitch control, space created (live), designed space, prime space (overlap of designed and real),
  pulled defenders (expected spot → actual), who should go where (green arrows and stars), vacated-spot cover
  (6 m circle: green clear, purple overload, grey covered), and every defender's expected spot.
- **Click a player:** attackers show a purple map of where they can get before the quickest defender (weighted by
  value) and their share of the opened space; defenders show how far they are from their expected spot. Click again
  to clear.
- After the pass, the play keeps running but nothing model-based is drawn.

## Keeping it in sync with the notebook

The analysis code in `src/spacecreation/core.py` is generated from the notebook, so the dashboard and the notebook
always use the same logic. After changing the notebook, regenerate it:
```
python scripts/extract_core.py
```
(`pip install nbformat` if it isn't installed.) Don't edit `core.py` by hand.

## Files

- `app/dashboard.py` — the Dash app: layout, and the server callbacks that precompute a whole play
- `app/assets/space.js` — drawing and playback in the browser
- `app/assets/style.css` — the look (Barlow type, tactics-board pitch)
- `src/spacecreation/core.py` — analysis functions and constants, generated from the notebook
- `src/spacecreation/loader.py` — loads the notebook's cached outputs into `core`
- `scripts/extract_core.py` — regenerates `core.py`
