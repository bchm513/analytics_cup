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
            drag_ids = [int(trk.player_ids[k]) for k in st["dfn"] if trk.player_ids[k] in dragged]
            f["drag"] = [r2(list(ex) + list(now)) + [pid] for ex, now, pid in zip(sc["drag_exp"], sc["drag_now"], drag_ids)
                         if not np.isnan(ex[0])]
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
app = Dash(__name__, title="Space Creation Explorer", assets_folder=str(APP_DIR / "assets"))
match_opts = [{"label": match_label(m), "value": m} for m in core.match_ids]

PITCH_ITEMS = [("attackers", "Attacking players"), ("defenders", "Defending players"), ("ball", "Ball"),
               ("path", "Path of the movement")]
LAYERS = [("pc", "Who controls the pitch"), ("created", "Space opened for teammates"), ("bens", "Where teammates should go"),
          ("drag", "Defenders pulled out of position"), ("prime", "Prime space (designed and open)"),
          ("designed", "Designed space (attack only)"), ("cover", "Is the vacated spot covered?"),
          ("expected", "Every defender's expected spot")]
DEFAULT_LAYERS = ["attackers", "defenders", "ball", "path", "pc", "created", "bens", "drag", "cover"]

app.layout = html.Div(className="shell", children=[
    html.Header(className="topbar", children=[
        html.H1("Space Creation Explorer"),
        dcc.Dropdown(match_opts, match_opts[0]["value"] if match_opts else None, id="match", clearable=False,
                     className="match-select"),
    ]),
    html.Div(className="body", children=[
        html.Aside(className="rail", children=[
            html.H2("Plays"),
            dcc.Checklist([{"label": "Off-ball runs", "value": "off_ball"}, {"label": "Carries", "value": "carry"}],
                          ["off_ball", "carry"], id="kinds", className="pills"),
            dcc.Checklist([{"label": "Opponent's half", "value": "opp"}, {"label": "Pulled a defender", "value": "drag"},
                           {"label": "Build-up", "value": "bu"}], ["opp", "drag", "bu"], id="filters", className="pills"),
            dash_table.DataTable(id="plays", page_size=14, sort_action="native",
                                 sort_by=[{"column_id": "clock", "direction": "asc"}],
                                 columns=[{"name": "Player", "id": "player"}, {"name": "", "id": "kind"},
                                          {"name": "Clock", "id": "clock"}, {"name": "Space", "id": "space"}],
                                 style_as_list_view=True, style_data_conditional=[]),
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
                dcc.Loading(html.Div(id="load-status", className="load-status"), type="dot", color="#2a9d8f"),
            ]),
            dcc.Graph(id="pitch", className="pitch", config=dict(displaylogo=False, scrollZoom=False,
                                                                  modeBarButtonsToRemove=["select2d", "lasso2d"])),
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
    trk = CTX.get(mid).trk
    return [dict(id=int(k), player=name(r.player_id), kind="carry" if r.kind == "carry" else "run",
                 clock=core.match_clock(trk, int(r.idx_start)), space=round(float(r.created_opp_half), 3))
            for k, r in zip(R.index, R.itertuples())]


@app.callback(Output("plays", "data"), Input("match", "value"), Input("kinds", "value"), Input("filters", "value"))
def update_plays(mid, kinds, filters):
    if mid is None:
        return []
    return plays_for(int(mid), tuple(sorted(kinds or [])), tuple(sorted(filters or [])))


@app.callback(Output("playdata", "data"), Output("load-status", "children"), Output("selected", "data"),
              Input("plays", "active_cell"), Input("go", "n_clicks"),
              State("plays", "derived_viewport_data"), State("match", "value"), State("clock", "value"),
              prevent_initial_call=True)
def load_play(cell, _go, view, mid, clock):
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
