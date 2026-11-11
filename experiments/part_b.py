"""Part B - 3x3 grid: fixed-cycle vs naive per-intersection FNN-GA vs coordinated
FNN-GA with a shared fuzzy input (neighbour's receiving-link occupancy D)."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semaphore import report as R
from semaphore.config import Config
from semaphore.controllers import FixedController, FNNController, webster_plan, expected_side_flows
from semaphore.fnn import FNN
from semaphore.sim import GridSim

ROWS = COLS = 3


def peak_profile(duration, centre, half_width, amp):
    t = np.arange(duration)
    return 1 + amp * np.clip(1 - np.abs((t - centre) / half_width), 0, 1)


def scenarios(warm, quick):
    d = warm + (2400 if quick else 5400)
    return {
        # Undersaturated: does anything go wrong when nothing is congested?
        "balanced": dict(side_vph={"N": 350, "S": 350, "E": 300, "W": 300},
                         shape=np.ones(d), exit_caps={}, duration=d),
        # Oversaturated rush hour: demand ramps to 1.9x in the middle of the run.
        "rush hour": dict(side_vph={"N": 450, "S": 450, "E": 380, "W": 420},
                          shape=peak_profile(d, warm + (d - warm) / 2, (d - warm) / 3, 0.9),
                          exit_caps={}, duration=d),
        # Downstream incident: eastbound exits drop to 900 veh/h for 30 min.
        "incident": dict(side_vph={"N": 350, "S": 350, "E": 300, "W": 550}, shape=np.ones(d),
                         exit_caps={"E": (0.25, warm + 1200, warm + 3000)}, duration=d),
    }


def entry_rates(sim, side_vph, lanes):
    return np.array([side_vph[L.side] for L in sim.entries], float) * lanes / 3600.0


def make_controllers(net3: FNN, net4: FNN, cfg, ef):
    return {
        "Fixed (Webster)": lambda n: FixedController(webster_plan(ef[n], cfg)),
        "FNN-GA naive": lambda n: FNNController(net3, cfg),
        "FNN-GA coordinated": lambda n: FNNController(net4, cfg),
    }


def run_network(controllers, rates, cfg, seeds, duration, exit_caps, rows=ROWS, cols=COLS, extra=None):
    out, keep = [], {}
    for name, mk in controllers.items():
        for s in seeds:
            sim = GridSim(rows, cols, mk, cfg, seed=2000 + s, exit_caps=exit_caps)
            m = sim.run(duration, rates)
            greens = [g[1] for sig in sim.signals.values() for g in sig.green_log if g[0] >= cfg.warmup_s]
            row = {"controller": name, "seed": s, **{k: m[k] for k in R.METRIC_COLS},
                   "mean_green_s": float(np.mean(greens)) if greens else np.nan,
                   "short_green_share": float(np.mean(np.array(greens) <= cfg.signal.g_min + 1)) if greens else np.nan}
            if extra:
                row.update(extra)
            out.append(row)
            if s == seeds[0]:
                keep[name] = {"node_wait": m["node_wait"], "ts": m["ts_queue"], "sim": sim}
    return out, keep


def fig_network(agg):
    metrics = [("avg_wait", "avg wait per stop (s)"), ("p95_wait", "95th-pct wait per stop (s)"),
               ("worst_node_wait", "worst intersection, mean wait (s)"),
               ("blocked_green_s", "green wasted on a full downstream\nlink (s per intersection-hour)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    fig.subplots_adjust(wspace=0.38)
    for ax, (m, lab) in zip(axes, metrics):
        R.grouped_bars(ax, agg, m, "scenario", "controller", R.CTRL_COLOR, fmt="{:.0f}")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Part B - 3x3 grid, network-wide metrics (mean of seeds)", fontsize=11, fontweight="bold")
    return R.save(fig, "b_network_comparison.png")


def fig_heatmap(keeps, scenario):
    names = list(keeps)
    vals = [np.array(keeps[n]["node_wait"]).reshape(ROWS, COLS) for n in names]
    vmax = max(np.nanmax(v) for v in vals)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6))
    for ax, n, v in zip(axes, names, vals):
        im = ax.imshow(v, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), x in np.ndenumerate(v):
            ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if x > 0.6 * vmax else R.TEXT)
        ax.set_title(n)
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean wait per stop (s)")
    fig.suptitle(f"Per-intersection mean wait, '{scenario}' scenario (west -> east = left -> right)",
                 fontsize=11, fontweight="bold")
    return R.save(fig, f"b_node_wait_{scenario.replace(' ', '_')}.png")


def fig_timeseries(all_keeps, cfg):
    fig, axes = plt.subplots(1, len(all_keeps), figsize=(6 * len(all_keeps), 3.6))
    for ax, (sc, keeps) in zip(np.atleast_1d(axes), all_keeps.items()):
        for n, k in keeps.items():
            ts = np.array(k["ts"])
            ax.plot((ts[:, 0] - cfg.warmup_s) / 60, ts[:, 1] + ts[:, 2], color=R.CTRL_COLOR[n], label=n)
        ax.set_title(sc)
        ax.set_xlabel("minutes after warm-up")
    np.atleast_1d(axes)[0].set_ylabel("vehicles queued in network\n(stop lines + entry buffers)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    return R.save(fig, "b_queue_timeseries.png")


def scaling_benchmark(net3, cfg, sizes=(3, 5, 10, 20), sim_s=600):
    """Measured simulator cost per grid size, and extrapolated GA training cost."""
    rows = []
    for n in sizes:
        probe = GridSim(n, n, lambda _: FixedController([20, 20]), cfg)
        lam = entry_rates(probe, {"N": 350, "S": 350, "E": 300, "W": 300}, cfg.road.lanes)
        sim = GridSim(n, n, lambda _: FNNController(net3, cfg), cfg, seed=1, record_timeseries=False)
        t0 = time.time()
        sim.run(sim_s, lam, warmup=0)
        el = time.time() - t0
        per_hour = el * 3600 / sim_s
        ga_evals = cfg.ga.population * cfg.ga.max_generations
        rows.append({"grid": f"{n}x{n}", "intersections": n * n, "sim_s_per_network_hour": per_hour,
                     "network_fitness_GA_hours": ga_evals * per_hour / 3600,
                     "genes_if_one_FNN_per_node": n * n * net3.n_genes})
    return pd.DataFrame(rows)


def run(base_cfg: Config, net3: FNN, quick=False):
    R.setup()
    cfg = base_cfg
    net4 = net3.expand_to_4_inputs()
    net4.save(R.MODELS / "fnn4_coordinated.npz")
    seeds = [0] if quick else [0, 1, 2]
    rows, keeps_all = [], {}
    for sname, sc in scenarios(cfg.warmup_s, quick).items():
        print(f"[B] scenario '{sname}' ...")
        probe = GridSim(ROWS, COLS, lambda _: FixedController([20, 20]), cfg)
        base = entry_rates(probe, sc["side_vph"], cfg.road.lanes)
        rates = base[:, None] * sc["shape"][None]
        ef = expected_side_flows(ROWS, COLS, base * sc["shape"].mean(), probe.entries, cfg.road.turn_probs)
        r, keeps = run_network(make_controllers(net3, net4, cfg, ef), rates, cfg, seeds, sc["duration"],
                               sc["exit_caps"], extra={"scenario": sname})
        rows += r
        keeps_all[sname] = keeps
        if sname != "balanced":
            fig_heatmap({k: v for k, v in keeps.items()}, sname)
    df = pd.DataFrame(rows)
    agg = R.summarise(rows, ["scenario", "controller"])
    extra = df.groupby(["scenario", "controller"], sort=False)[["mean_green_s", "short_green_share"]].mean().reset_index()
    agg = agg.merge(extra, on=["scenario", "controller"])
    R.write_table(agg, "b_network_comparison")
    fig_network(agg)
    fig_timeseries({k: v for k, v in keeps_all.items() if k != "balanced"}, cfg)
    print(agg[["scenario", "controller", "avg_wait", "p95_wait", "worst_node_wait", "avg_delay_all",
               "blocked_green_s", "short_green_share", "unfinished"]].round(2).to_string(index=False))

    print("[B5] Scaling benchmark ...")
    sc_df = scaling_benchmark(net3, cfg, sizes=(3, 5, 10) if quick else (3, 5, 10, 20))
    R.write_table(sc_df, "b_scaling_benchmark")
    print(sc_df.round(2).to_string(index=False))
    return net4, agg
