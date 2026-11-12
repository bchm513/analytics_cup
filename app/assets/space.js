/* Space Creation Explorer — everything that happens per frame runs here, in the browser.
   The server sends a whole play at once (dcc.Store "playdata"); these functions turn one frame of it into a figure. */

const C = {
  turf: "#eef3ec", line: "#b3c2b4", ink: "#1f2a24",
  att: "#c8394e", def: "#2b6a94", created: "42,157,143", ben: "#1f7a6a",
  prime: "233,162,59", designed: "120,100,176", reach: "106,76,147"
};

function zgrid(arr, g, floor) {
  const z = [];
  for (let r = 0; r < g.ny; r++) {
    const row = new Array(g.nx);
    for (let c = 0; c < g.nx; c++) {
      const v = arr[r * g.nx + c] / 100;
      row[c] = v < floor ? null : v;
    }
    z.push(row);
  }
  return z;
}

function heat(arr, g, scale, floor) {
  return {type: "heatmap", x: g.xs, y: g.ys, z: zgrid(arr, g, floor), zmin: 0, zmax: 1, colorscale: scale,
          showscale: false, hoverinfo: "skip", zsmooth: "best"};
}

function ramp(rgb, a0, a1) {
  return [[0, `rgba(${rgb},${a0})`], [1, `rgba(${rgb},${a1})`]];
}

function pitchShapes() {
  const L = 105, W = 68, ln = {color: C.line, width: 1.4};
  const s = [
    {type: "rect", x0: -L / 2, x1: L / 2, y0: -W / 2, y1: W / 2, line: ln, fillcolor: C.turf, layer: "below"},
    {type: "line", x0: 0, x1: 0, y0: -W / 2, y1: W / 2, line: ln, layer: "below"},
    {type: "circle", x0: -9.15, x1: 9.15, y0: -9.15, y1: 9.15, line: ln, layer: "below"}
  ];
  for (const sg of [-1, 1]) {
    s.push({type: "rect", x0: sg * L / 2, x1: sg * (L / 2 - 16.5), y0: -20.16, y1: 20.16, line: ln, layer: "below"});
    s.push({type: "rect", x0: sg * L / 2, x1: sg * (L / 2 - 5.5), y0: -9.16, y1: 9.16, line: ln, layer: "below"});
    s.push({type: "rect", x0: sg * L / 2, x1: sg * (L / 2 + 1.6), y0: -3.66, y1: 3.66, line: ln, layer: "below"});
  }
  return s;
}

function arrowAnn(x0, y0, x1, y1, color, width) {
  return {x: x1, y: y1, ax: x0, ay: y0, xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true,
          arrowhead: 2, arrowsize: 1, arrowwidth: width, arrowcolor: color, standoff: 7, startstandoff: 7, text: ""};
}

function emptyFigure() {
  return {data: [], layout: baseLayout(pitchShapes(), [])};
}

function baseLayout(shapes, anns) {
  return {
    shapes: shapes, annotations: anns, showlegend: false, autosize: true, uirevision: "pitch",
    margin: {l: 4, r: 4, t: 4, b: 4}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {range: [-54.5, 54.5], visible: false, fixedrange: true},
    yaxis: {range: [-35.5, 35.5], visible: false, scaleanchor: "x", scaleratio: 1, fixedrange: true},
    hovermode: "closest", dragmode: false,
    hoverlabel: {font: {family: "Barlow, sans-serif", size: 13}, bgcolor: "white", bordercolor: C.line},
    font: {family: "Barlow, sans-serif", color: C.ink}
  };
}

const TABLE_ROWS = 5;

function benTable(bens) {
  // Always the same size, so the page never jumps: empty rows read NA.
  let t = "| Teammate | Share | To reach | Gets there first | Control there |\n|---|---|---|---|---|\n";
  for (let n = 0; n < TABLE_ROWS; n++) {
    const b = bens && bens[n];
    t += b ? `| ${b.name} | ${Math.round(b.share * 100)}% | ${b.t.toFixed(1)} s | ${Math.round(b.first * 100)}% | ` +
             `${Math.round(b.ctrl * 100)}% |\n`
           : "| NA | NA | NA | NA | NA |\n";
  }
  return t;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  space: {
    render: function (data, k, layers, items, sel, reach) {
      if (!data || !data.frames || !data.frames.length) {
        return [emptyFigure(), benTable(null), "", "Pick a play from the list, or go to a moment by its match clock."];
      }
      k = Math.max(0, Math.min(k || 0, data.frames.length - 1));
      const f = data.frames[k], g = data.grid, L = new Set((layers || []).concat(items || []));
      const traces = [], anns = [], shapes = pitchShapes();

      if (L.has("pc") && f.pc) {
        traces.push(heat(f.pc, g, [[0, "rgba(43,106,148,0.42)"], [0.5, "rgba(238,243,236,0)"],
                                   [1, "rgba(200,57,78,0.42)"]], -1));
      }
      if (L.has("designed") && f.designed) traces.push(heat(f.designed, g, ramp(C.designed, 0.05, 0.6), 0.06));
      if (L.has("created") && f.created) traces.push(heat(f.created, g, ramp(C.created, 0.08, 0.85), 0.05));
      if (L.has("prime") && f.prime) traces.push(heat(f.prime, g, ramp(C.prime, 0.15, 0.9), 0.1));
      if (sel && reach && reach.pid === sel && reach.frames && reach.frames[k]) {
        traces.push(heat(reach.frames[k], g, ramp(C.reach, 0.05, 0.55), 0.08));
      }

      // the movement's path so far
      if (L.has("path") && data.runner_path && data.runner_path.length && k >= data.a_pos) {
        const n = Math.min(k, data.e_pos) - data.a_pos + 1;
        const p = data.runner_path.slice(0, Math.max(n, 0));
        traces.push({type: "scatter", mode: "lines", x: p.map(q => q[0]), y: p.map(q => q[1]), hoverinfo: "skip",
                     line: {color: data.movement === "carry" ? "#e07a1f" : C.att, width: 3,
                            dash: data.movement === "carry" ? "solid" : "dot"}});
      }
      if (L.has("drag") && f.drag) {
        for (const d of f.drag) {
          const who = data.names[d[4]] || "This defender";
          traces.push({type: "scatter", mode: "markers", x: [d[0]], y: [d[1]], hoverinfo: "text",
                       customdata: [d[4]],
                       hovertext: `${who}, ${data.roles[d[4]] || ""}: where he should have been`,
                       marker: {size: 12, color: "rgba(0,0,0,0)", line: {color: C.def, width: 2}}});
          anns.push(arrowAnn(d[0], d[1], d[2], d[3], C.def, 2));
        }
      }
      if (L.has("cover") && f.cover) {
        const [x, y, r, v] = f.cover;
        const col = {clear: C.ben, overload: "#7a5cc4", covered: "#8a8f8b"}[v];
        shapes.push({type: "circle", x0: x - r, x1: x + r, y0: y - r, y1: y + r, line: {color: col, width: 2, dash: "dot"}});
      }
      if (L.has("expected") && f.exp) {
        traces.push({type: "scatter", mode: "markers", x: f.exp.map(e => e[1]), y: f.exp.map(e => e[2]),
                     hoverinfo: "skip", marker: {size: 8, color: "rgba(0,0,0,0)", line: {color: C.def, width: 1.2}}});
      }
      if (L.has("bens") && f.bens) {
        f.bens.forEach((b, n) => {
          if (n > 2 || (n > 0 && b.share < 0.15)) return;
          const w = n === 0 ? 3 : 1.8;
          traces.push({type: "scatter", mode: "lines", x: [b.x0, b.x1], y: [b.y0, b.y1], hoverinfo: "skip",
                       line: {color: C.ben, width: w, dash: "dash"}, opacity: n === 0 ? 1 : 0.75});
          anns.push(arrowAnn(b.x0, b.y0, b.x1, b.y1, C.ben, w));
          traces.push({type: "scatter", mode: "markers", x: [b.x1], y: [b.y1], hoverinfo: "text",
                       hovertext: `${b.name}: best spot, ${Math.round(b.share * 100)}% of the usable space`,
                       marker: {symbol: "star", size: n === 0 ? 20 : 14, color: C.ben, line: {color: "white", width: 1}}});
          anns.push({x: b.x0, y: b.y0 + 3.2, xref: "x", yref: "y", showarrow: false,
                     text: `${b.name} ${Math.round(b.share * 100)}%`,
                     font: {size: n === 0 ? 13 : 11, color: C.ben, family: "Barlow Semi Condensed, sans-serif"},
                     bgcolor: "rgba(255,255,255,0.8)", borderpad: 2});
        });
      }
      if (f.pass) {
        traces.push({type: "scatter", mode: "markers", x: [f.pass[0]], y: [f.pass[1]], hoverinfo: "text",
                     hovertext: "where the ball went", marker: {symbol: "x", size: 15, color: C.ink}});
      }

      // players last, so they sit on top and are easy to click
      for (const side of ["a", "d"]) {
        if (!L.has(side === "a" ? "attackers" : "defenders")) continue;
        const ps = f.players.filter(p => p[1] === side);
        traces.push({type: "scatter", mode: "markers+text", x: ps.map(p => p[2]), y: ps.map(p => p[3]),
                     text: ps.map(p => p[4]), textfont: {color: "white", size: 10, family: "Barlow Semi Condensed"},
                     customdata: ps.map(p => p[0]), hoverinfo: "text",
                     hovertext: ps.map(p => `${data.names[p[0]] || p[0]}, ${data.roles[p[0]] || ""}, ${p[5]} m/s`),
                     marker: {size: 19, color: side === "a" ? C.att : C.def, line: {color: "white", width: 1.5}}});
      }
      if (f.runner && L.has("attackers")) {
        traces.push({type: "scatter", mode: "markers", x: [f.runner[0]], y: [f.runner[1]], hoverinfo: "skip",
                     marker: {size: 32, color: "rgba(0,0,0,0)", line: {color: C.ink, width: 2.5}}});
      }
      const me = sel ? f.players.find(p => p[0] === sel) : null;
      if (me) {
        traces.push({type: "scatter", mode: "markers", x: [me[2]], y: [me[3]], hoverinfo: "skip",
                     marker: {size: 34, color: "rgba(0,0,0,0)", line: {color: `rgb(${C.reach})`, width: 3}}});
      }
      if (L.has("ball")) {
        traces.push({type: "scatter", mode: "markers", x: [f.ball[0]], y: [f.ball[1]], hoverinfo: "skip",
                     marker: {size: 10, color: "white", line: {color: C.ink, width: 2}}});
      }

      // who-should-go-where table (fixed size) and the selected player's card
      const table = benTable(f.bens);
      let card = "Click a player on the pitch to follow him.";
      if (me) {
        const nm = data.names[me[0]] || me[0];
        card = `**${nm}**, ${data.roles[me[0]] || ""}, moving at ${me[5]} m/s`;
        if (me[1] === "a") {
          const b = (f.bens || []).find(x => x.pid === me[0]);
          card += "\n\nPurple shows where he can arrive before the quickest defender.";
          if (b) card += ` In the opened space he has ${Math.round(b.share * 100)}% of the usable room, ` +
                         `${b.t.toFixed(1)} s from his best spot.`;
        } else {
          const e = (f.exp || []).find(x => x[0] === me[0]);
          if (e) {
            const d = Math.hypot(e[1] - me[2], e[2] - me[3]);
            card += `\n\nHe is ${d.toFixed(1)} m from where a defender in his role usually stands with the ball here.`;
          }
        }
      }
      return [{data: traces, layout: baseLayout(shapes, anns)}, table, `${f.clock}   ${f.stage}`, card];
    },

    highlightRow: function (cell, view) {
      if (!cell || !view || cell.row >= view.length) return [];
      const id = view[cell.row].id;
      return [{if: {filter_query: `{id} = ${id}`}, backgroundColor: "#d6ece8", fontWeight: 600}];
    },

    timeline: function (data) {
      if (!data || !data.frames) return [1, {}, "", "Pick a play from the list"];
      const marks = {};
      for (const [k, lab] of Object.entries(data.marks || {})) marks[k] = {label: lab};
      return [data.frames.length - 1, marks, data.summary, data.title];
    },

    move: function (data, tick, back, fwd, js, jp, jr, jc, value, max) {
      const trig = (window.dash_clientside.callback_context.triggered[0] || {}).prop_id || "";
      if (!data) return 0;
      value = value || 0;
      if (trig.startsWith("playdata")) return data.start || 0;
      if (trig.startsWith("tick")) return value >= max ? 0 : value + 1;
      if (trig.startsWith("back")) return Math.max(0, value - 1);
      if (trig.startsWith("fwd")) return Math.min(max, value + 1);
      const want = {"j-start": "start", "j-peak": "peak", "j-prime": "prime", "j-pass": "pass"}[trig.split(".")[0]];
      if (want) {
        for (const [k, lab] of Object.entries(data.marks || {})) {
          if (lab.split(" / ").includes(want)) return parseInt(k, 10);
        }
      }
      return value;
    },

    playpause: function (n, speed, disabled) {
      const trig = (window.dash_clientside.callback_context.triggered[0] || {}).prop_id || "";
      if (trig.startsWith("play.")) disabled = !disabled;
      return [disabled, disabled ? "Play" : "Pause", speed];
    },

    select: function (click, current) {
      const pt = click && click.points && click.points[0];
      if (!pt || pt.customdata === undefined || pt.customdata === null) return window.dash_clientside.no_update;
      const pid = Array.isArray(pt.customdata) ? pt.customdata[0] : pt.customdata;
      return current === pid ? null : pid;
    }
  }
});
