# Setup guide (Windows)

Step-by-step instructions for getting this project running locally and onto GitHub.
Run the commands in PowerShell from the project folder unless noted otherwise.

## 1. Put the project folder in place

Unzip the project folder wherever you keep your work, e.g.
`C:\Users\<you>\TSLR\Analytics Cup\analytics-cup-football`.

If you already cloned SkillCorner's data somewhere else, you don't need to download it again:
move that `opendata` folder into this project's `data` folder (create `data` if needed), so the
path is `analytics-cup-football\data\opendata\data\matches.json`.

## 2. Get the SkillCorner data (skip if you moved it in step 1)

```powershell
git lfs install
git clone https://github.com/SkillCorner/opendata.git data/opendata
```

## 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Your prompt now starts with `(.venv)`. Activate it again whenever you open a new terminal.

In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → choose the one in `.venv`. When a
notebook asks for a kernel, pick the same `.venv` Python.

## 4. Check the setup

```powershell
python scripts/check_setup.py
python -m pytest
```

Every line of the check should say `[ok]`. If the tracking files are flagged as Git LFS
pointers, run `git lfs pull` inside `data\opendata`.

## 5. Explore

Open the notebooks in order:

| Notebook | What it covers |
|---|---|
| `01_data_overview` | matches, players, dynamic events, phases of play, aggregates |
| `02_tracking` | tracking format, how much is extrapolated, frames, average positions |
| `03_defensive_events` | on-ball engagements, pressing chains, defensive blocks, line height, pressure |
| `04_team_shape` | width, depth, compactness and line height by defensive phase |
| `05_shots` | shot frames, defenders and keepers at the moment of the shot |
| `sandbox/scratch` | blank starting point for trying your own ideas |

`05_shots` builds `data/processed/` the first time it runs (about a minute), or run
`python scripts/build_shot_frames.py` beforehand.

## 6. Create the GitHub repository

The competition requires a public repository, so do this once.

1. **Configure git** (first time on this computer only):
   ```powershell
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
   Use the email attached to your GitHub account.

2. **Create an empty repository on GitHub.** Go to github.com → **New repository**.
   Name it (e.g. `analytics-cup-football`), set it to **Public**, and leave **all** the
   "Initialize this repository with" options unchecked (no README, .gitignore or license;
   this project already has them).

3. **Make the first commit locally:**
   ```powershell
   git init
   git add .
   git status
   ```
   Check the `git status` list before committing: nothing under `data/` or `.venv/` should
   appear. If it does, stop and check `.gitignore` is in the project folder.
   ```powershell
   git commit -m "Initial project structure"
   git branch -M main
   ```

4. **Connect and push** (copy the URL from your new GitHub repository page):
   ```powershell
   git remote add origin https://github.com/<your-username>/analytics-cup-football.git
   git push -u origin main
   ```
   The first push opens a browser window to sign in to GitHub (Git Credential Manager,
   included with Git for Windows).

5. **Refresh the GitHub page.** You should see the code, README and LICENSE.md, and no data.

## Day-to-day git

```powershell
git add .
git commit -m "Short description of what changed"
git push
```

Commit small and often. Before the deadline, test the submission from scratch: clone the
repository into a new folder, follow the README exactly, and confirm everything runs.
