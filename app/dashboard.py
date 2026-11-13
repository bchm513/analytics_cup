"""Space Creation Explorer — interactive dashboard.

Run from the project folder (after running the notebook through Section 12 once):
    python app/dashboard.py              # then open http://127.0.0.1:8050
    python app/dashboard.py --port 8060 --root path/to/analytics-cup-football

How it stays fast: when you pick a play, the server computes every frame once (a few seconds) and sends the whole play
to the browser. Playing, pausing, scrubbing, switching layers and selecting players then run in the browser
(assets/space.js) with no round trip per frame. Clicking an attacker fetches his "where can I get to first" map for the
whole play in one request.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from dash import ClientsideFunction, Dash, Input, Output, State, dash_table, dcc, html, no_update
from dash import ctx as dash_ctx

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
from spacecreation import loader  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(ROOT))
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()

core = loader.init(args.root)
CTX = loader.ContextCache(size=3) 
PLAYS: dict = {}


def name(pid):
    return str(core.mins["name"].get(pid, pid))


def match_label(mid):
    m = core.metas[mid]
    return f"{m.team_names[m.home_id]} v {m.team_names[m.away_id]}"


def enc(arr, scale=1.0):
    """0–1 map -> list of ints 0–100 (compact JSON)."""
    return np.clip(np.round(np.nan_to_num(np.asarray(arr, float) / scale) * 100), 0, 100).astype(np.int16).tolist()


def r2(v):
    return [round(float(x), 2) for x in v]


# ---------------------------------------------------------------------------------------------------------------------
# Server-side analysis (once per play)
# ---------------------------------------------------------------------------------------------------------------------
def analyse_play(idx):
    if idx in PLAYS:
        return PLAYS[idx]
    run = next(core.RUNS.loc[[idx]].itertuples())
    c = CTX.get(run.match_id)
    ov = core.overlap_analysis(c, run)
    peak, _ = core.space_peak(c, run)
    vac = core.vacated_check(c, run, peak=peak)
    pinfo = core.pass_info(c, run)
    gmax = max((float(m.max()) for m in ov["real"].values()), default=1e-12) or 1e-12
    gpeak = float(ov["real"][peak].sum()) if peak in ov["real"] else gmax
    play = dict(run=run, ov=ov, peak=peak, vac=vac, pinfo=pinfo, gmax=gmax, gpeak=gpeak, match=int(run.match_id))
    if len(PLAYS) >= 6:
        PLAYS.pop(next(iter(PLAYS)))
    PLAYS[idx] = play
    return play


def players_at(c, i, team):
    trk, meta = c.trk, c.meta
    st = core.frame_state(trk, i, team)
    num = dict(zip(meta.players.player_id, meta.players.number))
    out = []
    for cols, side in ((st["att"], "a"), (st["dfn"], "d")):
        for j in cols:
            pid = int(trk.player_ids[j])
            n = num.get(pid)
            out.append([pid, side, round(float(st["pos"][j, 0]), 2), round(float(st["pos"][j, 1]), 2),
                        "" if n is None or n != n else str(int(n)), round(float(np.linalg.norm(st["vel"][j])), 1)])
    return out, st


def stage_text(i, a, e, cut, peak):
    if cut is not None and i > cut:
        return "After the pass: the play continues, nothing is computed"
    s = "Before the movement" if i < a else ("During the movement" if i <= e else "After the movement")
    if a <= i < peak:
        s += f", space peaks in {(peak - i) / core.FPS:.1f} s"
    return s


def precompute_play(idx):
    play = analyse_play(idx)
    run, ov, vac, pinfo = play["run"], play["ov"], play["vac"], play["pinfo"]
    c = CTX.get(play["match"])
    trk, g = c.trk, c.grid
    team = int(run.team_id)
    a, e, cut, peak, prime = int(run.idx_start), int(run.idx_end), ov["cut"], play["peak"], int(ov["prime"])
    idxs = [int(i) for i in ov["idxs"]]
    j_run = int(np.flatnonzero(trk.player_ids == run.player_id)[0])
    s_sign = trk.direction.get((team, int(trk.period[a])), 1)
    path = (trk.pos[a:e + 1, j_run] * s_sign).round(2).tolist()

    frames = []
    for i in idxs:
        sc = ov["scenes"].get(i) or core.run_scene(c, run, i, with_created=True)
        live = a <= i and (cut is None or i <= cut)
        players, st = players_at(c, i, team)
        exp = c.expected(st["dfn"], "out", st["ball"], trk.phase[i], st["pos"])
        f = dict(i=i, clock=core.match_clock(trk, i), stage=stage_text(i, a, e, cut, peak),
                 players=players, ball=r2(st["ball"]), runner=r2(sc["runner"]),
                 exp=[[int(trk.player_ids[j]), round(float(x), 2), round(float(y), 2)]
                      for j, (x, y) in zip(st["dfn"], exp) if not np.isnan(x)],
                 pc=enc(sc["pc"]), live=bool(live))
        if live:
            f["created"] = enc(np.clip(sc["created"], 0, None) * c.value_grid, play["gmax"])
            if i in ov["designed"]:
                f["designed"] = enc(ov["designed"][i], ov["dmax"])
            if i in ov["O"]:
                f["prime"] = enc(ov["O"][i])
            dragged = set(run.dragged_ids)
            drag_ids = [int(trk.player_ids[k]) for k in st["dfn"] if trk.player_ids[k] in dragged
                        and c.role.get(trk.player_ids[k]) != "GK"]
            # [expected x, y, actual x, y, defender, the mover pulling him]
            f["drag"] = [r2(list(ex) + list(now)) + [pid, int(run.player_id)]
                         for ex, now, pid in zip(sc["drag_exp"], sc["drag_now"], drag_ids) if not np.isnan(ex[0])]
            gain = np.clip(sc["created"], 0, None) * c.value_grid
            if gain.sum() >= core.LIVE_MIN_FRAC * play["gpeak"]:
                ben = core.find_beneficiary(c, run, i, sc["created"])
                if ben is not None:
                    rk = ben["ranking"].head(5)
                    f["bens"] = [dict(pid=int(r.player_id), name=str(r.name), x0=round(float(sc["P"][int(r.col), 0]), 2),
                                      y0=round(float(sc["P"][int(r.col), 1]), 2), x1=round(float(r.tx), 2),
                                      y1=round(float(r.ty), 2), share=round(float(r.share), 3),
                                      t=round(float(r.time_s), 1), first=round(float(r.ease), 2),
                                      ctrl=round(float(r.control_at_spot), 2), dmin=round(float(r.nearest_def_m), 1))
                                 for r in rk.itertuples()]
            if i >= peak and vac.get("spot") is not None:
                f["cover"] = [round(float(vac["spot"][0]), 2), round(float(vac["spot"][1]), 2), core.COVER_R, vac["cover"]]
        if cut is not None and i >= cut and pinfo:
            f["pass"] = r2(pinfo["target"])
        frames.append(f)

    pos = {i: k for k, i in enumerate(idxs)}
    marks = {}
    for key, lab in ((a, "start"), (peak, "peak"), (prime, "prime"), (cut, "pass")):
        if key is not None and key in pos:
            k = pos[key]
            marks[k] = f"{marks[k]} / {lab}" if k in marks else lab
    timing = None
    if pinfo:
        timing = (("early" if pinfo["idx"] < ov["w0"] else ("on time" if pinfo["idx"] <= ov["w1"] else "late"))
                  if ov["pmax"] > 0 else "no prime space")
    kind = "Carry" if run.kind == "carry" else f"Off-ball run ({str(run.run_type).replace('_', ' ')})"
    summary = [f"**{name(run.player_id)}**, {core.team_names.get(team, '')}",
               f"{kind} at {core.match_clock(trk, a)}",
               "",
               f"Space created: **{run.created:.3f}** ({run.created_opp_half:.3f} in the opponent's half)",
               f"Space peaks at {core.match_clock(trk, peak)}",
               (f"Prime window: {core.match_clock(trk, ov['w0'])} to {core.match_clock(trk, ov['w1'])}"
                if ov["pmax"] > 0 else "No prime space (designed and real space never overlap)"),
               core.cover_text(vac) or "No vacated defender spot",
               (f"Ball played at {core.match_clock(trk, pinfo['idx'])}: {name(pinfo['passer'])} to "
                f"{name(pinfo['receiver'])}, {timing}, {'completed' if pinfo['completed'] else 'not completed'}"
                if pinfo else "No pass in this window")]
    return dict(kind="play", key=dict(match=play["match"], team=team, frames=idxs), title=f"{name(run.player_id)}: {kind}",
                grid=dict(xs=r2(g.xs), ys=r2(g.ys), nx=g.nx, ny=g.ny), frames=frames, marks=marks,
                start=pos.get(a, 0), runner_path=path, a_pos=pos.get(a, 0), e_pos=pos.get(e, len(idxs) - 1),
                movement="carry" if run.kind == "carry" else "run", runner=int(run.player_id),
                summary="  \n".join(summary), names={str(p): name(p) for p in core.mins.index},
                roles={str(p): str(r) for p, r in core.mins["role"].items()})


@lru_cache(maxsize=None)
def period_starts(mid):
    """First frame of each period, read straight from the cached tracking file (much faster than loading the match)."""
    path = core.CACHE_DIR / f"{mid}_tracking.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        fr, per = z["frames"], z["period"]
    else:
        trk = CTX.get(mid).trk
        fr, per = trk.frames, trk.period
    return {int(p): int(fr[per == p].min()) for p in np.unique(per)}


def clock_of(mid, frame, period):
    """Same match clock as core.match_clock, from a frame number and period."""
    secs = (int(frame) - period_starts(mid)[int(period)]) / core.FPS + (45 * 60 if int(period) == 2 else 0)
    return f"P{int(period)} {int(secs // 60):02d}:{int(secs % 60):02d}"


# ---------------------------------------------------------------------------------------------------------------------
# Team plays: every movement by one team in the same moment of play
# ---------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def team_plays_for_match(mid):
    R = core.RUNS[core.RUNS.match_id == mid]
    Pm = core.PASSES[core.PASSES.match_id == mid] if len(core.PASSES) else core.PASSES
    return core.build_team_plays(R, Pm)


def precompute_teamplay(mid, k):
    tp = team_plays_for_match(mid).loc[k]
    c = CTX.get(mid)
    trk, g = c.trk, c.grid
    team = int(tp.team_id)
    i_start, i_end = int(trk.idx(int(tp.frame_start))), int(trk.idx(int(tp.frame_end)))
    same = np.flatnonzero(trk.period == trk.period[i_start])
    lo = int(max(i_start - core.TEAMPLAY_LEAD_S * core.FPS, same[0]))
    hi = int(min(i_end + core.TEAMPLAY_BUFFER_S * core.FPS, same[-1]))
    idxs = list(range(lo, hi + 1))
    tpd = dict(team_id=team, match_id=int(mid), members=list(tp.members))
    sign = trk.direction.get((team, int(trk.period[i_start])), 1)

    scenes = {i: core.team_scene(c, tpd, i, with_created=i_start <= i <= i_end) for i in idxs}
    gains = {i: np.clip(sc["created"], 0, None) * c.value_grid for i, sc in scenes.items() if "created" in sc}
    gmax = max((float(v.max()) for v in gains.values()), default=1e-12) or 1e-12
    peak = max(gains, key=lambda i: gains[i].sum()) if gains else i_start
    gpeak = float(gains[peak].sum()) if gains else 0.0

    pass_row = None
    if tp.pass_frame is not None and len(core.PASSES):
        pr = core.PASSES[(core.PASSES.match_id == mid) & (core.PASSES.team_id == team)
                         & (core.PASSES.frame_pass == tp.pass_frame)]
        pass_row = pr.iloc[0] if len(pr) else None
    i_pass = int(trk.idx(int(tp.pass_frame))) if tp.pass_frame is not None else None

    paths = []
    pos = {i: n for n, i in enumerate(idxs)}
    abnormal = [m for m in tp.members if bool(core.RUNS.at[m, "abnormal"])] if "abnormal" in core.RUNS \
        else list(tp.members)
    for m in abnormal:                            # only movements that stand out from the team's own shift
        r = core.RUNS.loc[m]
        a, e = int(r.idx_start), int(r.idx_end)
        j = int(np.flatnonzero(trk.player_ids == r.player_id)[0])
        a_pos = pos.get(a, 0 if a < lo else None)
        if a_pos is None:
            continue
        paths.append(dict(pid=int(r.player_id), kind="carry" if r.kind == "carry" else "run", a_pos=pos.get(a, 0),
                          e_pos=pos.get(min(e, hi), len(idxs) - 1),
                          path=(trk.pos[max(a, lo):min(e, hi) + 1, j] * sign).round(2).tolist()))

    frames = []
    for i in idxs:
        sc = scenes[i]
        live = i_start <= i <= i_end
        players, st = players_at(c, i, team)
        exp = c.expected(st["dfn"], "out", st["ball"], trk.phase[i], st["pos"])
        stage = ("Before the play" if i < i_start else ("During the play" if i <= i_end
                 else "After the play: shown for context, nothing is computed"))
        if live and i < peak:
            stage += f", space peaks in {(peak - i) / core.FPS:.1f} s"
        f = dict(i=i, clock=core.match_clock(trk, i), stage=stage, players=players, ball=r2(st["ball"]),
                 exp=[[int(trk.player_ids[j]), round(float(x), 2), round(float(y), 2)]
                      for j, (x, y) in zip(st["dfn"], exp) if not np.isnan(x)],
                 pc=enc(sc["pc"]), live=bool(live), movers=sorted(sc["movers"]) if live else [])
        if live:
            f["created"] = enc(gains[i], gmax)
            f["drag"] = [r2([ex[0], ex[1], now[0], now[1]]) + [d, m] for d, ex, now, m in sc["drag"]]
            if gains[i].sum() >= core.LIVE_MIN_FRAC * gpeak and gpeak > 0:
                ben = core.team_play_beneficiaries(c, tpd, i, sc["created"], sc["movers"])
                if ben is not None:
                    rk = ben["ranking"].head(5)
                    f["bens"] = [dict(pid=int(r.player_id), name=str(r.name), x0=round(float(sc["P"][int(r.col), 0]), 2),
                                      y0=round(float(sc["P"][int(r.col), 1]), 2), x1=round(float(r.tx), 2),
                                      y1=round(float(r.ty), 2), share=round(float(r.share), 3),
                                      t=round(float(r.time_s), 1), first=round(float(r.ease), 2),
                                      ctrl=round(float(r.control_at_spot), 2), dmin=round(float(r.nearest_def_m), 1))
                                 for r in rk.itertuples()]
        if pass_row is not None and i_pass is not None and i >= i_pass:
            f["pass"] = [round(float(pass_row.target_x), 2), round(float(pass_row.target_y), 2)]
        frames.append(f)

    marks = {}
    for key, lab in ((i_start, "start"), (peak, "peak"), (i_pass if i_pass is not None and i_pass <= hi else i_end,
                                                          "pass" if i_pass is not None else "end")):
        if key in pos:
            n = pos[key]
            marks[n] = f"{marks[n]} / {lab}" if n in marks else lab
    counts = {}
    for m in abnormal:                       # one entry per player: "N. Blair (2 runs, 1 carry)"
        pid, kind = int(core.RUNS.at[m, "player_id"]), str(core.RUNS.at[m, "kind"])
        counts.setdefault(pid, {"run": 0, "carry": 0})["carry" if kind == "carry" else "run"] += 1
    movers_txt = ", ".join(
        f"{name(pid)} (" + ", ".join(f"{n} {w}{'s' if n > 1 else ''}" for w, n in c.items() if n) + ")"
        for pid, c in counts.items())
    n_pulled = len({int(d) for m in abnormal for d in core.RUNS.at[m, "dragged_ids"]})
    summary = [f"**{core.team_names.get(team, '')}**, {core.match_clock(trk, i_start)} to {core.match_clock(trk, i_end)}",
               (f"{len(counts)} player{'s' if len(counts) != 1 else ''} moving differently from the team: {movers_txt}"
                if counts else "Nobody moved differently from the team in this play"),
               "",
               f"Defenders pulled out of position: **{n_pulled}**",
               f"Space peaks at {core.match_clock(trk, peak)}",
               (f"Ball played at {core.match_clock(trk, i_pass)}: {name(int(pass_row.player_id))} to "
                f"{name(int(pass_row.target_id))}, {'completed' if pass_row.completed else 'not completed'}"
                if pass_row is not None else "No pass: the play ends without one"),
               "",
               "Movers are not suggested as runners into the space; nor are centre-backs or keepers."]
    return dict(kind="teamplay", key=dict(match=int(mid), team=team, frames=idxs),
                title=f"{core.team_names.get(team, '')}: team play, {len(abnormal)} movement"
                      f"{'s' if len(abnormal) != 1 else ''}",
                grid=dict(xs=r2(g.xs), ys=r2(g.ys), nx=g.nx, ny=g.ny), frames=frames, marks=marks,
                start=pos.get(i_start, 0), paths=paths, runner_path=[], a_pos=0, e_pos=0, movement=None, runner=None,
                summary="  \n".join(summary), names={str(p): name(p) for p in core.mins.index},
                roles={str(p): str(r) for p, r in core.mins["role"].items()})


def precompute_free(mid, i0):
    c = CTX.get(mid)
    trk, g = c.trk, c.grid
    team = int(trk.poss_team[i0]) if trk.poss_team[i0] != -1 else int(c.meta.home_id)
    same = np.flatnonzero(trk.period == trk.period[i0])
    idxs = list(range(int(max(i0 - 50, same[0])), int(min(i0 + 200, same[-1])) + 1))
    frames = []
    for i in idxs:
        players, st = players_at(c, i, team)
        pc = core.pitch_control(st["pos"][st["att"]], st["vel"][st["att"]], st["pos"][st["dfn"]],
                                st["vel"][st["dfn"]], st["ball"], g.points)
        exp = c.expected(st["dfn"], "out", st["ball"], trk.phase[i], st["pos"])
        frames.append(dict(i=i, clock=core.match_clock(trk, i), stage="Free exploration", players=players,
                           ball=r2(st["ball"]), pc=enc(pc), live=False,
                           exp=[[int(trk.player_ids[j]), round(float(x), 2), round(float(y), 2)]
                                for j, (x, y) in zip(st["dfn"], exp) if not np.isnan(x)]))
    return dict(kind="free", key=dict(match=int(mid), team=team, frames=idxs),
                title=f"{core.team_names.get(team, '')} in possession, attacking left to right",
                grid=dict(xs=r2(g.xs), ys=r2(g.ys), nx=g.nx, ny=g.ny), frames=frames, marks={},
                start=idxs.index(i0) if i0 in idxs else 0, runner_path=[], a_pos=0, e_pos=0, movement=None,
                runner=None, summary=f"Free exploration of {match_label(mid)} from {core.match_clock(trk, i0)}.",
                names={str(p): name(p) for p in core.mins.index},
                roles={str(p): str(r) for p, r in core.mins["role"].items()})


def reach_for(key, pid):
    """Where player pid can get before the quickest defender, value-weighted, for every frame of the play."""
    c = CTX.get(key["match"])
    trk = c.trk
    maps, peak = [], 1e-12
    for i in key["frames"]:
        st = core.frame_state(trk, i, key["team"])
        js = [j for j in st["att"] if trk.player_ids[j] == pid]
        if not js:
            maps.append(None)
            continue
        j = js[0]
        t_me = core.arrival_times(st["pos"][[j]], st["vel"][[j]], c.grid.points)[0]
        t_def = core.arrival_times(st["pos"][st["dfn"]], st["vel"][st["dfn"]], c.grid.points).min(0)
        m = 1 / (1 + np.exp(-(t_def - t_me) / core.ARRIVAL_SIGMA)) * c.value_grid
        maps.append(m)
        peak = max(peak, float(m.max()))
    return [None if m is None else enc(m, peak) for m in maps]


# ---------------------------------------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------------------------------------
LIST_TABLE_STYLE = dict(
    style_table={"width": "100%", "overflowX": "hidden"},
    style_cell={"overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap", "maxWidth": 0,
                "textAlign": "left", "padding": "6px 4px"},
    css=[{"selector": "table", "rule": "table-layout: fixed; width: 100%;"}],
)
# ---------------------------------------------------------------------------------------------------------------------
app = Dash(__name__, title="Space Creation Explorer", assets_folder=str(APP_DIR / "assets"))
match_opts = [{"label": match_label(m), "value": m} for m in core.match_ids]

PITCH_ITEMS = [("attackers", "Attacking players"), ("defenders", "Defending players"), ("ball", "Ball"),
               ("path", "Path of the movement")]
LAYERS = [("links", "Lines from movers to the defenders they pull"), ("pc", "Who controls the pitch"),
          ("bens_all", "Show every suggested run, not just the best"), ("created", "Space opened for teammates"), ("bens", "Where teammates should go"),
          ("drag", "Defenders pulled out of position"), ("prime", "Prime space (designed and open)"),
          ("designed", "Designed space (attack only)"), ("cover", "Is the vacated spot covered?"),
          ("expected", "Every defender's expected spot")]
DEFAULT_LAYERS = ["attackers", "defenders", "ball", "path", "links", "pc", "created", "bens", "drag", "cover"]

app.layout = html.Div(className="shell", children=[
    html.Header(className="topbar", children=[
        html.H1("Space Creation Explorer"),
        dcc.Dropdown(match_opts, match_opts[0]["value"] if match_opts else None, id="match", clearable=False,
                     className="match-select"),
    ]),
    html.Div(className="body", children=[
        html.Aside(className="rail", children=[
            dcc.Tabs(id="mode", value="team", className="mode-tabs", children=[
                dcc.Tab(label="Team plays", value="team", className="mode-tab", selected_className="mode-tab--on"),
                dcc.Tab(label="Individual movements", value="ind", className="mode-tab",
                        selected_className="mode-tab--on")]),
            dcc.Checklist([{"label": "Off-ball runs", "value": "off_ball"}, {"label": "Carries", "value": "carry"}],
                          ["off_ball", "carry"], id="kinds", className="pills"),
            dcc.Checklist([{"label": "Opponent's half", "value": "opp"}, {"label": "Pulled a defender", "value": "drag"},
                           {"label": "Build-up", "value": "bu"}], ["opp", "drag", "bu"], id="filters", className="pills"),
            dcc.Loading(type="circle", color="#1f7a6a", className="list-loading", children=[
            html.Div(id="ind-wrap", children=[
                dash_table.DataTable(id="plays", page_size=14, sort_action="native",
                                     sort_by=[{"column_id": "clock", "direction": "asc"}],
                                     columns=[{"name": "Team", "id": "team"}, {"name": "Player", "id": "player"},
                                              {"name": "", "id": "kind"}, {"name": "Clock", "id": "clock"},
                                              {"name": "Space", "id": "space"}],
                                     style_cell_conditional=[
                                         {"if": {"column_id": "team"}, "width": "27%"},
                                         {"if": {"column_id": "player"}, "width": "29%"},
                                         {"if": {"column_id": "kind"}, "width": "12%"},
                                         {"if": {"column_id": "clock"}, "width": "18%"},
                                         {"if": {"column_id": "space"}, "width": "14%", "textAlign": "right"}],
                                     tooltip_duration=None,
                                     style_as_list_view=True, style_data_conditional=[], **LIST_TABLE_STYLE)]),
            html.Div(id="team-wrap", children=[
                dash_table.DataTable(id="teamplays", page_size=14, sort_action="native",
                                     sort_by=[{"column_id": "clock", "direction": "asc"}],
                                     columns=[{"name": "Team", "id": "team"}, {"name": "Moves", "id": "moves"},
                                              {"name": "Clock", "id": "clock"}, {"name": "Space", "id": "space"}],
                                     style_cell_conditional=[
                                         {"if": {"column_id": "team"}, "width": "40%"},
                                         {"if": {"column_id": "moves"}, "width": "16%", "textAlign": "right"},
                                         {"if": {"column_id": "clock"}, "width": "24%"},
                                         {"if": {"column_id": "space"}, "width": "20%", "textAlign": "right"}],
                                     style_as_list_view=True, style_data_conditional=[], **LIST_TABLE_STYLE)]),
            html.Div(id="plays-status", className="plays-status"),
            ]),
            html.Div(className="jump", children=[
                html.Label("Go to a moment", htmlFor="clock"),
                html.Div(className="jump-row", children=[dcc.Input(id="clock", placeholder="P1 12:34"),
                                                         html.Button("Go", id="go")]),
            ]),
        ]),
        html.Main(className="stage", children=[
            html.Div(className="stage-head", children=[
                html.Div(id="play-title", className="play-title", children="Pick a play from the list"),
                html.Div(id="frame-title", className="frame-title"),
            ]),
            html.Div(className="pitch-wrap", children=[
                dcc.Graph(id="pitch", className="pitch", config=dict(displaylogo=False, scrollZoom=False,
                                                                      modeBarButtonsToRemove=["select2d", "lasso2d"])),
                # the spinner sits in the middle of the pitch while a play is computed
                html.Div(className="pitch-spinner", children=dcc.Loading(
                    html.Div(id="load-status", className="load-status"), type="circle", color="#1f7a6a")),
            ]),
            html.Div(className="transport", children=[
                html.Button("⏮", id="back", title="Back one frame"),
                html.Button("Play", id="play", className="primary"),
                html.Button("⏭", id="fwd", title="Forward one frame"),
                dcc.Dropdown([{"label": "0.5×", "value": 200}, {"label": "1×", "value": 100},
                              {"label": "2×", "value": 50}], 100, id="speed", clearable=False, className="speed"),
                html.Div(className="jumps", children=[html.Button("Start", id="j-start"), html.Button("Peak", id="j-peak"),
                                                      html.Button("Prime", id="j-prime"), html.Button("Pass", id="j-pass")]),
            ]),
            dcc.Slider(id="frame", min=0, max=1, step=1, value=0, marks=None, updatemode="drag", className="timeline"),
            html.Div(className="moment-wrap", children=[
                html.H2("Where teammates should go at this moment"),
                dcc.Markdown(id="moment", className="moment"),
            ]),
        ]),
        html.Aside(className="side", children=[
            html.H2("This play"),
            dcc.Markdown(id="summary", className="summary"),
            html.H2("Selected player"),
            dcc.Markdown(id="player-card", className="summary"),
            html.H2("On the pitch"),
            dcc.Checklist([{"label": n, "value": k} for k, n in PITCH_ITEMS], [k for k, _ in PITCH_ITEMS], id="items",
                          className="layers"),
            html.H2("Layers"),
            dcc.Checklist([{"label": n, "value": k} for k, n in LAYERS],
                          [k for k in DEFAULT_LAYERS if k in dict(LAYERS)], id="layers", className="layers"),
            html.P("Click a player on the pitch to follow him. Click him again to let go.", className="hint"),
        ]),
    ]),
    dcc.Store(id="playdata"), dcc.Store(id="selected"), dcc.Store(id="reach"),
    dcc.Interval(id="tick", interval=100, disabled=True),
])


# ---------------------------------------------------------------------------------------------------------------------
# Server callbacks (only when something new must be computed)
# ---------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def plays_for(mid, kinds, filters):
    R = core.RUNS[(core.RUNS.match_id == mid) & core.RUNS.kind.isin(kinds)]
    if "opp" in filters:
        R = R[R.attacking_half]
    if "drag" in filters:
        R = R[R.n_dragged > 0]
    if "bu" in filters:
        R = R[R.phase.isin(core.BUILDUP_PHASES)]
    R = R.sort_values("created_opp_half", ascending=False).head(300).sort_values(["period", "frame_start"])
    return [dict(id=int(k), team=core.team_names.get(int(r.team_id), ""), player=name(r.player_id),
                 kind="carry" if r.kind == "carry" else "run",
                 clock=clock_of(mid, r.frame_start, r.period), space=round(float(r.created_opp_half), 3))
            for k, r in zip(R.index, R.itertuples())]


@lru_cache(maxsize=64)
def teamplays_for(mid, kinds, filters):
    T = team_plays_for_match(mid)
    if not len(T):
        return []
    R = core.RUNS
    keep = []
    for k, t in T.iterrows():
        m = R.loc[t.members]
        m = m[m.kind.isin(kinds)]
        if not len(m):
            continue
        if "opp" in filters and not m.attacking_half.any():
            continue
        if "drag" in filters and not (m.n_dragged > 0).any():
            continue
        if "bu" in filters and not m.phase.isin(core.BUILDUP_PHASES).any():
            continue
        keep.append(dict(id=f"{mid}:{k}", team=core.team_names.get(int(t.team_id), ""), moves=int(t.n_moves),
                         clock=clock_of(mid, t.frame_start, t.period), space=round(float(t.space), 3)))
    return keep


@app.callback(Output("plays", "data"), Output("teamplays", "data"), Output("plays-status", "children"),
              Input("match", "value"), Input("kinds", "value"), Input("filters", "value"), Input("mode", "value"))
def update_plays(mid, kinds, filters, mode):
    if mid is None:
        return [], [], ""
    k, f = tuple(sorted(kinds or [])), tuple(sorted(filters or []))
    try:
        ind = plays_for(int(mid), k, f)
        team = teamplays_for(int(mid), k, f)
    except Exception as exc:                       # never fail silently
        return [], [], f"Couldn't build the list: {exc!r}"
    n = len(team) if mode == "team" else len(ind)
    what = "team play" if mode == "team" else "movement"
    return ind, team, f"{n} {what}{'s' if n != 1 else ''} in this match"


@app.callback(Output("ind-wrap", "style"), Output("team-wrap", "style"), Input("mode", "value"))
def show_list(mode):
    return ({"display": "none"}, {}) if mode == "team" else ({}, {"display": "none"})


@app.callback(Output("playdata", "data"), Output("load-status", "children"), Output("selected", "data"),
              Input("plays", "active_cell"), Input("teamplays", "active_cell"), Input("go", "n_clicks"),
              State("plays", "derived_viewport_data"), State("teamplays", "derived_viewport_data"),
              State("match", "value"), State("clock", "value"), prevent_initial_call=True)
def load_play(cell, tcell, _go, view, tview, mid, clock):
    if dash_ctx.triggered_id == "teamplays":
        if not tcell or not tview or tcell["row"] >= len(tview):
            return no_update, no_update, no_update
        m, k = tview[tcell["row"]]["id"].split(":")
        return precompute_teamplay(int(m), int(k)), "", None
    if dash_ctx.triggered_id == "go":
        try:
            trk = CTX.get(mid).trk
            p = int(clock.strip().split()[0][1])
            mm, ss = map(int, clock.strip().split()[1].split(":"))
            f = trk.frames[trk.period == p].min() + ((mm * 60 + ss) - (45 * 60 if p == 2 else 0)) * core.FPS
            i0 = int(np.clip(np.searchsorted(trk.frames, f), 0, len(trk.frames) - 1))
        except Exception:
            return no_update, "Type a clock like P1 12:34", no_update
        return precompute_free(int(mid), i0), "", None
    if not cell or not view or cell["row"] >= len(view):
        return no_update, no_update, no_update
    return precompute_play(int(view[cell["row"]]["id"])), "", None


@app.callback(Output("reach", "data"), Input("selected", "data"), State("playdata", "data"), prevent_initial_call=True)
def load_reach(pid, data):
    if pid is None or not data:
        return None
    side = next((p[1] for f in data["frames"] for p in f["players"] if p[0] == pid), None)
    if side != "a":
        return None
    return dict(pid=pid, frames=reach_for(data["key"], pid))


# ---------------------------------------------------------------------------------------------------------------------
# Browser callbacks (assets/space.js): drawing, playback, selection
# ---------------------------------------------------------------------------------------------------------------------
app.clientside_callback(ClientsideFunction("space", "render"),
                        Output("pitch", "figure"), Output("moment", "children"), Output("frame-title", "children"),
                        Output("player-card", "children"),
                        Input("playdata", "data"), Input("frame", "value"), Input("layers", "value"),
                        Input("items", "value"), Input("selected", "data"), Input("reach", "data"))
app.clientside_callback(ClientsideFunction("space", "highlightRow"), Output("plays", "style_data_conditional"),
                        Input("plays", "active_cell"), State("plays", "derived_viewport_data"))
app.clientside_callback(ClientsideFunction("space", "highlightRow"), Output("teamplays", "style_data_conditional"),
                        Input("teamplays", "active_cell"), State("teamplays", "derived_viewport_data"))
app.clientside_callback(ClientsideFunction("space", "timeline"),
                        Output("frame", "max"), Output("frame", "marks"), Output("summary", "children"),
                        Output("play-title", "children"), Input("playdata", "data"))
app.clientside_callback(ClientsideFunction("space", "move"), Output("frame", "value"),
                        Input("playdata", "data"), Input("tick", "n_intervals"), Input("back", "n_clicks"),
                        Input("fwd", "n_clicks"), Input("j-start", "n_clicks"), Input("j-peak", "n_clicks"),
                        Input("j-prime", "n_clicks"), Input("j-pass", "n_clicks"), State("frame", "value"),
                        State("frame", "max"))
app.clientside_callback(ClientsideFunction("space", "playpause"),
                        Output("tick", "disabled"), Output("play", "children"), Output("tick", "interval"),
                        Input("play", "n_clicks"), Input("speed", "value"), State("tick", "disabled"))
app.clientside_callback(ClientsideFunction("space", "select"), Output("selected", "data", allow_duplicate=True),
                        Input("pitch", "clickData"), State("selected", "data"), prevent_initial_call=True)

if __name__ == "__main__":
    app.run(debug=args.debug, port=args.port)
