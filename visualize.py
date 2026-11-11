"""Live visualisation: watch a controller manage traffic, decision by decision.

    python visualize.py                      # Part A: isolated intersection, trained FNN-GA
    python visualize.py --part B             # Part B: 3x3 grid, normal balanced traffic
    python visualize.py --part B --scenario rush-hour  # demand rises to 1.9x, then falls
    python visualize.py --part B --scenario incident   # exit restriction, minutes 5-35
    python visualize.py --part B --scenario incident --incident-start 5 --incident-end 20
    python visualize.py --part C             # Part C: 3x3 grid, real METR-LA morning rush
    python visualize.py --controller fixed   # same scene with another controller (see --help)
    python visualize.py --part B --scenario incident --snapshot shot.png --at 1500

Left: the simulator. Every vehicle is drawn where it really is (grey = moving,
orange = queued at the stop line), the bar at each stop line is that approach's
signal, a red-shaded link is completely full (spillback), red +N = cars that
cannot even enter the network.

Right: the controller of the selected intersection (click another one to switch):
  * what it sees    - qg, qr, AV (and D in the coordinated grid) with their fuzzy terms
  * what it thinks  - the rules that fire hardest and their consequents
  * what it does    - the green extension dt and what the signal does with it
  * signal history and the queue on each approach over the last minutes

Keys: space = pause / resume, up / down = faster / slower, r = restart.
The scene is simulated first (a few seconds), then played back.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["LAB3_INTERACTIVE"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

if "--snapshot" in sys.argv:
    matplotlib.use("Agg")
else:
    for _b in ("TkAgg", "QtAgg", "GTK3Agg"):
        try:
            matplotlib.use(_b, force=True)
            import matplotlib.pyplot as _plt  # noqa: F401
            _plt.figure()
            _plt.close("all")
            break
        except Exception:
            continue
    else:
        sys.exit("No window backend available. On Ubuntu:  sudo apt install python3-tk   "
                 "(or use --snapshot file.png to save a frame instead).")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from semaphore import fuzzy  # noqa: E402
from semaphore import report as R  # noqa: E402
from semaphore.config import DEFAULT  # noqa: E402
from semaphore.controllers import FixedController, FNNController, webster_plan, expected_side_flows  # noqa: E402
from semaphore.fnn import FNN  # noqa: E402
from semaphore.sim import GridSim, SIDES, PHASE_SIDES  # noqa: E402

CAR_MOVING, CAR_QUEUED = "#7a7974", "#eb6834"
SIG = {"G": "#1baf7a", "Y": "#eda100", "R": "#e34948"}
ROAD = "#dcdbd6"
SIDE_DIR = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}   # node -> upstream direction
SIDE_COL = [R.SERIES[0], R.SERIES[1], R.SERIES[2], "#b81f83"]      # N blue, E orange, S green, W magenta
VAR_COL = [R.SERIES[0], R.SERIES[7], R.SERIES[2], R.SERIES[3]]      # qg, qr, AV, D


# --------------------------------------------------------------------------- recording
class LoggedFNN(FNNController):
    """FNN controller that also records what it saw, which rules fired, and what it chose."""

    def __init__(self, net, cfg, node, log):
        super().__init__(net, cfg)
        self.node, self.log = node, log

    def decide(self, obs):
        x = self.features(obs)
        lay = self.net.layers(np.array([x]))
        dt = float(lay["out"][0])
        self.log.setdefault(self.node, []).append(
            {"t": obs["t"], "x": x, "dt": dt, "phibar": lay["phibar"][0], "mu": lay["mu"][0],
             "phase": obs["phase"]})
        return dt


class Recorder:
    def __init__(self, sim, t0, t1, every, link_m, spacing, lane_w):
        self.sim, self.t0, self.t1, self.every = sim, t0, t1, every
        self.link_m, self.spacing, self.lane_w = link_m, spacing, lane_w
        self.hw = 3 * lane_w
        pitch = link_m + 2 * self.hw
        self.centre = {n: np.array([n[1] * pitch, -n[0] * pitch], float) for n in sim.nodes}
        self.frames = []

    def __call__(self, sim, t):
        if not (self.t0 <= t < self.t1) or (t - self.t0) % self.every:
            return
        hw, lw = self.hw, self.lane_w
        mov, que, full, buffers, sig, q = [], [], [], {}, {}, {}
        for n in sim.nodes:
            c = self.centre[n]
            s_ = sim.signals[n]
            for si, L in enumerate(sim.links[n]):
                d = np.array(SIDE_DIR[SIDES[si]], float)
                p = np.array([-d[1], d[0]])
                served = si in PHASE_SIDES[s_.phase]
                sig[(n, si)] = ("G" if s_.state == "G" else "Y") if served else "R"
                nq = len(L.queue)
                q[(n, si)] = nq
                for i in range(nq):
                    x = 3.0 + (i // 3) * self.spacing
                    que.append(c + d * (hw + min(x, self.link_m)) + p * ((i % 3) + 0.5) * lw)
                tail = 3.0 + (nq // 3 + 1) * self.spacing
                for v in L.transit:
                    x = max(tail, (v.t_stop - t) / L.tt * self.link_m)
                    mov.append(c + d * (hw + min(x, self.link_m)) + p * ((id(v) % 3) + 0.5) * lw)
                if L.occupancy >= L.storage:
                    full.append((n, si))
                if L.is_entry and L.buffer:
                    buffers[(n, si)] = len(L.buffer)
        green = {n: (sim.signals[n].phase, sim.signals[n].state, t - sim.signals[n].t_start,
                     sim.signals[n].ext_used) for n in sim.nodes}
        queued = sum(q.values()) + sum(len(L.buffer) for L in sim.entries)
        self.frames.append({"t": t, "mov": np.array(mov).reshape(-1, 2), "que": np.array(que).reshape(-1, 2),
                            "sig": sig, "full": full, "buffers": buffers, "q": q, "green": green,
                            "queued": queued})


# --------------------------------------------------------------------------- scenes
def scene_a(controller):
    from experiments.part_a import part_a_config, side_rates
    cfg = part_a_config(DEFAULT)
    net = FNN.load(R.MODELS / "fnn3_trained.npz", cfg.ga.offset_frac)
    rates = side_rates(650, 550, cfg.road.lanes)
    log = {}
    choices = {
        "fnn": ("FNN-GA (trained)", lambda n: LoggedFNN(net, cfg, n, log)),
        "flc": ("FLC, paper rules (untrained)", lambda n: LoggedFNN(FNN(3, cfg.signal.e_max, cfg.ga.offset_frac), cfg, n, log)),
        "fixed": ("Fixed cycle (Webster)", lambda n: FixedController(webster_plan(rates, cfg))),
    }
    title, mk = choices[controller]
    return dict(cfg=cfg, rows=1, cols=1, make=mk, rates=rates, duration=2400, t0=600, t1=2400, every=1,
                exit_caps=None, lane_w=8.0, marker=30, log=log, net=net if controller == "fnn" else FNN(3),
                title=f"Part A - isolated intersection, very high demand (650 N-S / 550 E-W veh/h/lane): {title}",
                clock=lambda t: f"t = {_hms(t - 600)}", sel=(0, 0), choices=list(choices))


def scene_b(controller, scenario=None, incident_start=5.0, incident_end=35.0):
    from experiments.part_b import scenarios, entry_rates, ROWS, COLS
    cfg = DEFAULT
    net3 = FNN.load(R.MODELS / "fnn3_trained.npz", cfg.ga.offset_frac)
    net4 = net3.expand_to_4_inputs()
    configs = scenarios(cfg.warmup_s, quick=False)
    scenario_key = {None: "balanced", "rush-hour": "rush hour", "incident": "incident"}[scenario]
    sc = configs[scenario_key]
    probe = GridSim(ROWS, COLS, lambda _: FixedController([20, 20]), cfg)
    base = entry_rates(probe, sc["side_vph"], cfg.road.lanes)
    rates = base[:, None] * sc["shape"][None]
    ef = expected_side_flows(ROWS, COLS, rates.mean(axis=1), probe.entries, cfg.road.turn_probs)
    # Keep the playback window fixed so timing changes preserve the arrival sequence.
    incident_cap, default_inc0, _ = configs["incident"]["exit_caps"]["E"]
    t0 = default_inc0 - 300
    inc0 = t0 + int(round(incident_start * 60))
    inc1 = t0 + int(round(incident_end * 60))
    exit_caps = {"E": (incident_cap, inc0, inc1)} if scenario == "incident" else {}
    log = {}
    choices = {
        "coord": ("FNN-GA coordinated (shared input D)", lambda n: LoggedFNN(net4, cfg, n, log), net4),
        "naive": ("FNN-GA naive (isolated copies)", lambda n: LoggedFNN(net3, cfg, n, log), net3),
        "fixed": ("Fixed cycle (Webster)", lambda n: FixedController(webster_plan(ef[n], cfg)), net3),
    }
    title, mk, net = choices[controller]

    def clock(t):
        if scenario == "rush-hour":
            factor = sc["shape"][min(int(t), len(sc["shape"]) - 1)]
            state = f"RUSH HOUR: input traffic {factor:.2f}x baseline"
        else:
            state = ("INCIDENT: east exits 900 veh/h"
                     if scenario == "incident" and inc0 <= t < inc1 else "east exits normal")
        return f"t = {_hms(t - t0)}   {state}"
    scenario_title = {
        None: "normal balanced traffic",
        "rush-hour": "rush hour (demand rises to 1.9x, then falls)",
        "incident": f"eastbound exit incident (minutes {incident_start:g}-{incident_end:g})",
    }[scenario]
    return dict(cfg=cfg, rows=ROWS, cols=COLS, make=mk, rates=rates,
                duration=sc["duration"], t0=t0, t1=t0 + 3600, every=2, exit_caps=exit_caps,
                lane_w=6.0, marker=5, log=log, net=net, sel=(1, 1), choices=list(choices), clock=clock,
                title=f"Part B - 3x3 grid, {scenario_title}: {title}")


def scene_c(controller, data_path):
    import pandas as pd
    from semaphore import metrla as M
    from experiments.part_b import ROWS, COLS
    cfg = DEFAULT
    net3 = FNN.load(R.MODELS / "fnn3_trained.npz", cfg.ga.offset_frac)
    net4 = net3.expand_to_4_inputs()
    print("    loading METR-LA ...")
    raw = M.load_metrla(data_path)
    clean, _, _ = M.clean(raw)
    sensors, _ = M.choose_sensors(raw, 12)
    day = M.choose_days(raw, sensors, 2)[0]
    vf = M.free_flow_speed(clean[sensors])
    lam, _, _ = M.speed_to_demand(clean.loc[day:day + pd.Timedelta("23:55:00"), sensors], vf)
    rates = M.to_per_second(lam, cfg.road.lanes)
    probe = GridSim(ROWS, COLS, lambda _: FixedController([20, 20]), cfg)
    ef = expected_side_flows(ROWS, COLS, rates.mean(axis=1), probe.entries, cfg.road.turn_probs)
    log = {}
    choices = {
        "coord": ("FNN-GA coordinated", lambda n: LoggedFNN(net4, cfg, n, log), net4),
        "naive": ("FNN-GA naive", lambda n: LoggedFNN(net3, cfg, n, log), net3),
        "fixed": ("Fixed cycle (Webster, daily mean)", lambda n: FixedController(webster_plan(ef[n], cfg)), net3),
    }
    title, mk, net = choices[controller]
    return dict(cfg=cfg, rows=ROWS, cols=COLS, make=mk, rates=rates, duration=86400, t0=6 * 3600,
                t1=11 * 3600, every=5, exit_caps=None, lane_w=6.0, marker=5, log=log, net=net, sel=(1, 1),
                choices=list(choices), clock=lambda t: f"{_hms(t)}  ({day.date()}, real METR-LA demand)",
                title=f"Part C - 3x3 grid driven by METR-LA sensors, morning rush: {title}")


def _hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def simulate(sc, seed):
    sim = GridSim(sc["rows"], sc["cols"], sc["make"], sc["cfg"], seed=seed, exit_caps=sc["exit_caps"],
                  record_timeseries=False)
    rec = Recorder(sim, sc["t0"], sc["t1"], sc["every"], sc["cfg"].road.link_length_m,
                   sc["cfg"].road.jam_spacing_m, sc["lane_w"])
    sim.run(min(sc["duration"], sc["t1"] + 1), sc["rates"], hooks=[rec])
    return sim, rec


# --------------------------------------------------------------------------- viewer
class Viewer:
    def __init__(self, sc, sim, rec):
        self.sc, self.sim, self.rec = sc, sim, rec
        self.frames = rec.frames
        self.k, self.speed, self.paused = 0, 1, False
        self.sel = sc["sel"]
        self.net = sc["net"]
        self.rule_names = (["/".join(k) for k, _ in fuzzy.rule_table_3()] if self.net.n_inputs == 3
                           else ["/".join(k) for k, _ in fuzzy.rule_table_4()])
        self.w = self.net.split(self.net.genome)[0][0]
        self.vars = fuzzy.INPUTS_3 if self.net.n_inputs == 3 else fuzzy.INPUTS_4
        self.adaptive = bool(sc["log"]) or not isinstance(next(iter(sim.signals.values())).ctrl, FixedController)
        self.times = np.array([f["t"] for f in self.frames])
        # per-node arrays for the history panels
        self.qarr = {n: np.array([[f["q"][(n, si)] for f in self.frames] for si in range(4)]) for n in sim.nodes}
        code = lambda g: 2 if g[1] == "Y" else g[0]            # 0 = N-S green, 1 = E-W green, 2 = yellow
        self.parr = {n: np.array([code(f["green"][n]) for f in self.frames]) for n in sim.nodes}
        self.dec_t = {n: np.array([d["t"] for d in sc["log"].get(n, [])]) for n in sim.nodes}
        self._build()

    # ---- layout --------------------------------------------------------
    def _build(self):
        sc = self.sc
        self.fig = fig = plt.figure(figsize=(16, 9), facecolor=R.SURFACE)
        fig.canvas.manager.set_window_title("Lab 3 - smart semaphore control") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
        fig.suptitle(sc["title"], fontsize=12, fontweight="bold", y=0.975)
        fig.text(0.29, 0.015, "grey = moving   orange = queued   stop-line bar = signal   red link = full (spillback)"
                              "   red +N = waiting to enter\nspace pause   up/down speed   r restart   "
                              "click an intersection to inspect its controller",
                 ha="center", fontsize=8.5, color=R.TEXT2, linespacing=1.5)
        self.ax = ax = fig.add_axes([0.01, 0.07, 0.56, 0.86])
        self._draw_static(ax)
        self.mov = ax.scatter([], [], s=sc["marker"], c=CAR_MOVING, marker="s", lw=0, zorder=4)
        self.que = ax.scatter([], [], s=sc["marker"], c=CAR_QUEUED, marker="s", lw=0, zorder=5)
        self.info = ax.text(0.01, 0.995, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                            bbox=dict(fc=R.SURFACE, ec="none", alpha=0.85, pad=3), zorder=10)
        # right-hand controller panel
        x0, wd = 0.61, 0.37
        self.head = fig.text(x0, 0.905, "", fontsize=11, fontweight="bold", color=R.TEXT, va="top")
        self.status = fig.text(x0, 0.872, "", fontsize=9.5, color=R.TEXT, va="top", family="monospace")
        self.ax_in = fig.add_axes([x0 + 0.04, 0.68, wd - 0.04, 0.13])
        self.rules = fig.text(x0, 0.655, "", fontsize=9, color=R.TEXT, va="top", family="monospace", linespacing=1.45)
        self.out = fig.text(x0, 0.475, "", fontsize=10.5, color=R.TEXT, va="top", fontweight="bold", linespacing=1.4)
        self.ax_tl = fig.add_axes([x0, 0.385, wd, 0.035])
        self.ax_q = fig.add_axes([x0 + 0.03, 0.075, wd - 0.03, 0.22])
        self._init_panels()
        fig.canvas.mpl_connect("key_press_event", self._key)
        fig.canvas.mpl_connect("button_press_event", self._click)

    def _draw_static(self, ax):
        rows, cols = self.sc["rows"], self.sc["cols"]
        link_m, hw = self.rec.link_m, self.rec.hw
        pitch = link_m + 2 * hw
        ext = hw + link_m
        for r in range(rows):
            y = -r * pitch
            ax.add_patch(Rectangle((-ext, y - hw), (cols - 1) * pitch + 2 * ext, 2 * hw, fc=ROAD, ec="none", zorder=0))
            ax.plot([-ext, (cols - 1) * pitch + ext], [y, y], color="#f2c14e", lw=0.6, zorder=1)
        for c in range(cols):
            x = c * pitch
            ax.add_patch(Rectangle((x - hw, -(rows - 1) * pitch - ext), 2 * hw, (rows - 1) * pitch + 2 * ext,
                                   fc=ROAD, ec="none", zorder=0))
            ax.plot([x, x], [-(rows - 1) * pitch - ext, ext], color="#f2c14e", lw=0.6, zorder=1)
        segs, self.sig_keys, self.shades = [], [], {}
        for n, cc in self.rec.centre.items():
            ax.add_patch(Rectangle(cc - hw, 2 * hw, 2 * hw, fc="#cfcec8", ec="none", zorder=1))
            for si, side in enumerate(SIDES):
                d = np.array(SIDE_DIR[side], float)
                p = np.array([-d[1], d[0]])
                a = cc + d * (hw + 1.5)
                segs.append([a, a + p * hw])
                self.sig_keys.append((n, si))
                poly = [cc + d * hw, cc + d * (hw + link_m), cc + d * (hw + link_m) + p * hw, cc + d * hw + p * hw]
                self.shades[(n, si)] = ax.add_patch(Polygon(poly, closed=True, fc="#e34948", alpha=0.0, ec="none", zorder=1))
        self.sig_lc = LineCollection(segs, linewidths=3.2, zorder=6)
        ax.add_collection(self.sig_lc)
        self.sel_box = ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=R.SERIES[0], lw=2.2, zorder=7))
        self.btxt = {}
        for L in self.sim.entries:
            d = np.array(SIDE_DIR[L.side], float)
            pos = self.rec.centre[L.node] + d * (ext + 6)
            self.btxt[(L.node, SIDES.index(L.side))] = ax.text(*pos, "", fontsize=8, color=R.SERIES[7], ha="center",
                                                                va="center", fontweight="bold", zorder=10)
        span = (rows - 1) * pitch + 2 * ext
        ax.set_xlim(-ext - 0.06 * span, (cols - 1) * pitch + ext + 0.06 * span)
        ax.set_ylim(-(rows - 1) * pitch - ext - 0.04 * span, ext + 0.14 * span)
        ax.set_aspect("equal")
        ax.axis("off")

    def _init_panels(self):
        ax = self.ax_in
        names = [v.name for v in self.vars]
        self.bars = ax.barh(range(len(names)), [0] * len(names), color=VAR_COL[:len(names)], height=0.62)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(False)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title("what it sees (bar = share of the input's range)", fontsize=9, color=R.TEXT2, loc="left")
        self.bar_txt = [ax.text(0.01, i, "", va="center", fontsize=8.5, color=R.TEXT, family="monospace")
                        for i in range(len(names))]
        ax = self.ax_tl
        ax.set_yticks([])
        ax.set_xlabel("signal, last 5 min:  blue = N-S green   aqua = E-W green   yellow = intergreen",
                      fontsize=8, color=R.TEXT2)
        ax.tick_params(labelsize=7)
        self.tl_img = ax.imshow(np.zeros((1, 2)), aspect="auto", cmap=ListedColormap([R.SERIES[0], R.SERIES[2], R.SERIES[3]]),
                                vmin=0, vmax=2, extent=(-300, 0, 0, 1), interpolation="nearest")
        ax.set_xlim(-300, 0)
        ax = self.ax_q
        self.qlines = [ax.plot([], [], color=SIDE_COL[i], lw=1.6, label=SIDES[i])[0] for i in range(4)]
        ax.set_xlim(-600, 0)
        ax.set_xlabel("seconds before now", fontsize=8, color=R.TEXT2)
        ax.set_ylabel("queue (veh)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7.5, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                  handlelength=1.2, columnspacing=1.0)
        ax.set_title("queue per approach, last 10 min", fontsize=9, color=R.TEXT2, loc="left")

    # ---- interaction ---------------------------------------------------
    def _key(self, e):
        if e.key == " ":
            self.paused = not self.paused
        elif e.key == "up":
            self.speed = min(self.speed * 2, 64)
        elif e.key == "down":
            self.speed = max(self.speed // 2, 1)
        elif e.key == "r":
            self.k = 0

    def _click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p = np.array([e.xdata, e.ydata])
        self.sel = min(self.rec.centre, key=lambda n: np.linalg.norm(self.rec.centre[n] - p))
        self.draw(self.k)
        self.fig.canvas.draw_idle()

    # ---- drawing ---------------------------------------------------------
    def draw(self, k):
        f = self.frames[k]
        t = f["t"]
        self.mov.set_offsets(f["mov"] if len(f["mov"]) else np.empty((0, 2)))
        self.que.set_offsets(f["que"] if len(f["que"]) else np.empty((0, 2)))
        self.sig_lc.set_color([SIG[f["sig"][key]] for key in self.sig_keys])
        full = set(f["full"])
        for key, sh in self.shades.items():
            sh.set_alpha(0.3 if key in full else 0.0)
        for key, tx in self.btxt.items():
            nb = f["buffers"].get(key, 0)
            tx.set_text(f"+{nb}" if nb else "")
        speed = self.speed * self.sc["every"] * 25
        self.info.set_text(f"{self.sc['clock'](t)}\nvehicles queued {f['queued']:5d}   full links {len(full):2d}\n"
                           f"playback {speed:.0f}x real time{'   PAUSED' if self.paused else ''}")
        c = self.rec.centre[self.sel]
        hw = self.rec.hw
        self.sel_box.set_xy(c - hw - 6)
        self.sel_box.set_width(2 * hw + 12)
        self.sel_box.set_height(2 * hw + 12)
        self._panel(k, f)

    def _panel(self, k, f):
        n = self.sel
        ph, st, el, ext = f["green"][n]
        name = "N-S" if ph == 0 else "E-W"
        label = f"intersection {n}" if self.sc["rows"] > 1 else "the intersection"
        self.head.set_text(f"Controller of {label}")
        sig = f"{name} green for {el:3d} s" if st == "G" else f"{name} yellow / all-red"
        self.status.set_text(f"{sig}    extension used {ext:4.1f} / 30 s\n"
                             f"queues  N {f['q'][(n, 0)]:3d}   E {f['q'][(n, 1)]:3d}   S {f['q'][(n, 2)]:3d}   "
                             f"W {f['q'][(n, 3)]:3d}")
        # signal timeline
        lo = np.searchsorted(self.times, f["t"] - 300)
        self.tl_img.set_data(self.parr[n][lo:k + 1][None])
        self.tl_img.set_extent((self.times[lo] - f["t"], 0.0, 0, 1))
        # queue history
        lo2 = np.searchsorted(self.times, f["t"] - 600)
        tt = self.times[lo2:k + 1] - f["t"]
        ymax = max(5, self.qarr[n].max() * 1.08)
        for i, ln in enumerate(self.qlines):
            ln.set_data(tt, self.qarr[n][i, lo2:k + 1])
        self.ax_q.set_ylim(0, ymax)
        # latest decision
        dts = self.dec_t.get(n)
        if dts is None or len(dts) == 0 or not self.sc["log"]:
            for b in self.bars:
                b.set_width(0)
            for tx in self.bar_txt:
                tx.set_text("")
            self.rules.set_text("Fixed-time plan: no sensing, no decisions.\n"
                                "Green times are preset from average demand (Webster).")
            self.out.set_text("")
            return
        j = np.searchsorted(dts, f["t"], side="right") - 1
        if j < 0:
            self.rules.set_text("waiting for the first decision ...")
            self.out.set_text("")
            return
        d = self.sc["log"][n][j]
        mu = d["mu"]
        off = 0
        for i, (var, val) in enumerate(zip(self.vars, d["x"])):
            nt = len(var.terms)
            m = mu[off:off + nt]
            off += nt
            hi = var.universe[1]
            self.bars[i].set_width(min(val / hi, 1.0))
            terms = "  ".join(f"{t} {v:.2f}" for t, v in sorted(zip(var.terms, m), key=lambda z: -z[1]) if v > 0.01)
            unit = "" if var.name == "D" else " veh/lane"
            self.bar_txt[i].set_text(f"{val:5.2f}{unit}   ->  {terms}")
            self.bar_txt[i].set_x(min(val / hi, 1.0) + 0.01 if val / hi < 0.45 else 0.01)
            self.bar_txt[i].set_color(R.TEXT if val / hi < 0.45 else "white")
        top = np.argsort(d["phibar"])[::-1][:4]
        lines = ["what it thinks (strongest rules at this decision)",
                 "  rule  " + ("qg/qr/AV/D" if self.net.n_inputs == 4 else "qg/qr/AV") + "        weight   then dt"]
        for r in top:
            if d["phibar"][r] < 0.005:
                break
            lines.append(f"  IF {self.rule_names[r]:<16s}  {d['phibar'][r] * 100:5.1f} %   {self.w[r]:5.1f} s")
        self.rules.set_text("\n".join(lines))
        dt = d["dt"]
        ago = f["t"] - d["t"]
        if dt >= 10:
            act = "keep green, ask again in 10 s (if extension budget is left)"
        elif dt >= 0.5:
            act = f"end this green in {round(dt)} s"
        else:
            act = "end this green now -> switch phase"
        prev = st == "G" and d["phase"] != ph
        note = ("   [previous green - next decision at g_min = 10 s]" if prev else "")
        self.out.set_text(f"what it does: dt = {dt:4.1f} s   (decided {ago:.0f} s ago, "
                          f"{'N-S' if d['phase'] == 0 else 'E-W'} green){note}\n-> {act}")

    def _tick(self, _):
        if not self.paused:
            self.k = (self.k + self.speed) % len(self.frames)
        self.draw(self.k)
        return []

    def show(self):
        self.anim = FuncAnimation(self.fig, self._tick, interval=40, cache_frame_data=False)
        plt.show()

    def snapshot(self, path, at):
        self.k = int(np.clip(np.searchsorted(self.times, self.sc["t0"] + at), 0, len(self.frames) - 1))
        self.draw(self.k)
        self.fig.savefig(path, dpi=110)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "C"], default="A")
    ap.add_argument("--controller", default=None,
                    help="A: fnn | flc | fixed    B, C: coord | naive | fixed   (default: the trained/coordinated one)")
    ap.add_argument("--data", default=str(ROOT / "data" / "kaggle" / "METR-LA.h5"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scenario", choices=["incident", "rush-hour"], default=None,
                    help="Part B: choose an event (default: normal balanced traffic)")
    ap.add_argument("--incident-start", type=float, default=None,
                    help="Part B: incident start in playback minutes (default: 5)")
    ap.add_argument("--incident-end", type=float, default=None,
                    help="Part B: incident end in playback minutes (default: 35; maximum: 60)")
    ap.add_argument("--snapshot", default=None, help="save one frame to this PNG instead of opening a window")
    ap.add_argument("--at", type=float, default=600, help="with --snapshot: seconds into the scene")
    a = ap.parse_args()
    custom_incident = a.incident_start is not None or a.incident_end is not None
    if (a.scenario is not None or custom_incident) and a.part != "B":
        ap.error("scenario and incident options are only available for Part B")
    if custom_incident and a.scenario != "incident":
        ap.error("--incident-start and --incident-end require --scenario incident")
    incident_start = 5.0 if a.incident_start is None else a.incident_start
    incident_end = 35.0 if a.incident_end is None else a.incident_end
    if not 0 <= incident_start < incident_end <= 60:
        ap.error("incident times must satisfy 0 <= start < end <= 60 playback minutes")
    if round(incident_start * 60) >= round(incident_end * 60):
        ap.error("incident must last at least one second after rounding to simulation seconds")
    if not (R.MODELS / "fnn3_trained.npz").exists():
        sys.exit("models/fnn3_trained.npz missing - run  python main.py --part A  first")
    ctrl = a.controller or {"A": "fnn", "B": "coord", "C": "coord"}[a.part]
    if a.part == "A":
        sc = scene_a(ctrl)
    elif a.part == "B":
        sc = scene_b(ctrl, scenario=a.scenario,
                     incident_start=incident_start, incident_end=incident_end)
    else:
        if not Path(a.data).exists():
            sys.exit(f"{a.data} not found - put the Kaggle METR-LA.h5 there (see README)")
        sc = scene_c(ctrl, a.data)
    print(f"Simulating Part {a.part} with controller '{ctrl}' ...")
    sim, rec = simulate(sc, a.seed)
    v = Viewer(sc, sim, rec)
    if a.snapshot:
        v.snapshot(a.snapshot, a.at)
    else:
        print("Window open: space = pause, up/down = speed, r = restart, click an intersection to inspect it.")
        v.show()


if __name__ == "__main__":
    main()
