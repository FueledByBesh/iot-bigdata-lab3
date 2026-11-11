"""Part A - isolated intersection: FLC -> FNN -> GA, compared with fixed-cycle."""
from __future__ import annotations

import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semaphore import fuzzy
from semaphore import report as R
from semaphore.config import Config, RoadConfig
from semaphore.controllers import FixedController, FNNController, webster_plan
from semaphore.fnn import FNN
from semaphore.ga import run_ga
from semaphore.oracle import collect_training_set
from semaphore.sim import GridSim

# Synthetic Poisson scenarios, per-lane flow (veh/h/lane) on (N-S, E-W).
SCENARIOS = {
    "low": (200, 150),
    "medium": (400, 300),
    "high": (550, 450),
    "very high": (650, 550),
    "asymmetric": (600, 150),
}


def part_a_config(base: Config) -> Config:
    """The paper's intersection is straight-through only."""
    return replace(base, road=replace(base.road, turn_probs=(1.0, 0.0, 0.0)))


def side_rates(ns, ew, lanes):
    return np.array([ns, ew, ns, ew], float) * lanes / 3600.0


def evaluate_isolated(controllers, rates, cfg, seeds, duration):
    rows = []
    for name, make in controllers.items():
        for s in seeds:
            sim = GridSim(1, 1, {(0, 0): make()}, cfg, seed=1000 + s)
            m = sim.run(duration, rates)
            rows.append({"controller": name, "seed": s, **{k: m[k] for k in R.METRIC_COLS}})
    return rows


# ---------------------------------------------------------------------------
def fig_membership(net_trained: FNN):
    init = FNN(3)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    variables = [(fuzzy.QG, 0), (fuzzy.AV, 8), (fuzzy.D, None), (fuzzy.DT, None)]
    titles = ["qg, qr (queue, veh/lane)\nsolid = initial, dashed = GA-tuned (qg)",
              "AV (approaching, veh/lane)\nsolid = initial, dashed = GA-tuned",
              "D (downstream occupancy, Part B)", "dt output (s)"]
    for ax, (var, term_offset), title in zip(axes, variables, titles):
        lo, hi = var.universe
        x = np.linspace(lo, hi, 400)
        for i, (t, p) in enumerate(zip(var.terms, var.params)):
            col = R.SERIES[i]
            ax.plot(x, fuzzy.trapmf(x, *p), color=col, label=t)
            if term_offset is not None:   # trained shift of the qg / AV terms
                tp = net_trained.mf_params()[term_offset + i]
                ax.plot(x, fuzzy.trapmf(x, *tp), color=col, ls="--", lw=1.2)
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.08)
        ax.legend(fontsize=8, ncol=len(var.terms), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    return R.save(fig, "a_membership_functions.png")


def tables_rule_base(net: FNN):
    rows = []
    w = net.split(net.genome)[0][0]
    for (key, term), wi in zip(fuzzy.rule_table_3(), w):
        nearest = fuzzy.DT_TERMS[int(np.argmin(np.abs(fuzzy.DT_CENTRES - wi)))]
        paper = "yes" if key in fuzzy.PAPER_RULES else ""
        rows.append({"qg": key[0], "qr": key[1], "AV": key[2], "dt (initial)": term,
                     "paper rule": paper, "dt trained (s)": round(float(wi), 1),
                     "dt trained (term)": nearest})
    df = pd.DataFrame(rows)
    R.write_table(df, "a_rule_base_48")
    mf = []
    for var in fuzzy.INPUTS_4 + (fuzzy.DT,):
        for t, p in zip(var.terms, var.params):
            mf.append({"variable": var.name, "term": t, "a": p[0], "b": p[1], "c": p[2], "d": p[3]})
    R.write_table(pd.DataFrame(mf), "a_membership_params")
    return df


def ga_sweep(net_factory, Xtr, ytr, Xva, yva, cfg, seeds, max_gen):
    rows, curves = [], {}
    settings = [("population", p, cfg.ga.mutation_prob) for p in (50, 220, 500)] + \
               [("mutation", cfg.ga.population, m) for m in (0.001, 0.01, 0.05, 0.1)]
    for kind, pop, mut in settings:
        for s in seeds:
            gcfg = replace(cfg.ga, population=pop, mutation_prob=mut, max_generations=max_gen)
            t0 = time.time()
            res = run_ga(net_factory(), Xtr, ytr, gcfg, np.random.default_rng(100 + s), Xva, yva)
            rows.append({"varied": kind, "population": pop, "mutation_prob": mut, "seed": s,
                         "train_E": res.best_error, "val_E": res.history["val"][-1],
                         "generations": res.generations, "converged": res.converged,
                         "time_s": time.time() - t0})
            curves.setdefault((kind, pop, mut), []).append(res.history["best"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["varied", "population", "mutation_prob"], sort=False).agg(
        train_E=("train_E", "mean"), val_E=("val_E", "mean"), val_E_std=("val_E", "std"),
        generations=("generations", "mean"), converged_runs=("converged", "sum"),
        time_s=("time_s", "mean")).reset_index()
    return agg, curves


def fig_sweep(curves):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, kind, title in ((axes[0], "population", "Population size (mutation 0.01)"),
                            (axes[1], "mutation", "Mutation probability (population 220)")):
        i = 0
        for (k, pop, mut), cs in curves.items():
            if k != kind:
                continue
            L = min(len(c) for c in cs)
            m = np.mean([c[:L] for c in cs], axis=0)
            lab = f"pop {pop}" if kind == "population" else f"mut {mut}"
            ax.plot(np.arange(1, L + 1), m, color=R.SERIES[i], label=lab)
            i += 1
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("best training error E")
    return R.save(fig, "a_ga_sweep.png")


def fig_convergence(res, eps):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = np.arange(1, len(res.history["best"]) + 1)
    f_best = 1 - 1 / (1 + np.array(res.history["best"]))
    ax.plot(g, res.history["best"], color=R.SERIES[0], label="best E (train)")
    ax.plot(g, res.history["mean"], color=R.SERIES[1], lw=1.2, label="population mean E")
    ax.plot(g, res.history["val"], color=R.SERIES[2], lw=1.2, ls="--", label="best individual, validation E")
    ax.plot(g, f_best, color=R.TEXT2, lw=1, ls=":", label="1 - f_best")
    ax.axhline(eps, color=R.TEXT2, lw=0.8)
    ax.annotate(f"stop condition 1 - f <= {eps}", (g[-1], eps), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=R.TEXT2)
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("squared error (normalised)")
    ax.set_title("GA convergence (paper parameters: pop 220, cx 0.5, mut 0.01)")
    ax.legend(fontsize=8)
    return R.save(fig, "a_ga_convergence.png")


def fig_baseline(agg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, v, e, lab in ((axes[0], "avg_queue", "avg_queue_std", "average queue (veh / approach)"),
                          (axes[1], "avg_wait", "avg_wait_std", "average wait per vehicle (s)")):
        R.grouped_bars(ax, agg, v, "scenario", "controller", R.CTRL_COLOR, err=e)
        ax.set_ylabel(lab)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Part A - isolated intersection, synthetic Poisson arrivals (mean of seeds, log scale)",
                 fontsize=11, fontweight="bold")
    return R.save(fig, "a_baseline_comparison.png")


# ---------------------------------------------------------------------------
def run(base_cfg: Config, quick=False, fresh=False):
    R.setup()
    cfg = part_a_config(base_cfg)
    out = {}
    print("[A1] Fuzzy controller: 48-rule base (paper's 6 rules reproduced exactly)")
    flc = fuzzy.FuzzyController(3)
    net0 = FNN(3, cfg.signal.e_max, cfg.ga.offset_frac)
    X = np.random.default_rng(0).uniform([0, 0, 0], [20, 20, 6], (500, 3))
    diff = np.max(np.abs(np.array([flc.infer(*x) for x in X]) - net0.forward_population(net0.genome[None], X)[0]))
    print(f"[A2] FNN initialised from rule base: max |FNN - FLC| = {diff:.2e}")
    out["fnn_flc_max_diff"] = float(diff)

    ts_path = R.MODELS / "training_set.npz"
    if ts_path.exists() and not fresh and not quick:
        z = np.load(ts_path)
        Xall, yall = z["X"], z["y"]
        print(f"[A3] Loaded cached training set ({len(yall)} samples) - use --fresh to regenerate")
    else:
        n_sc = 8 if quick else None
        print("[A3] Generating training samples with the rollout teacher ...")
        t0 = time.time()
        Xall, yall = collect_training_set(cfg, seed=cfg.seed, n_scenarios=n_sc)
        print(f"     {len(yall)} samples in {time.time() - t0:.0f} s")
        if not quick:
            np.savez(ts_path, X=Xall, y=yall)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(yall))
    ntr = int(0.8 * len(idx))
    Xtr, ytr, Xva, yva = Xall[idx[:ntr]], yall[idx[:ntr]], Xall[idx[ntr:]], yall[idx[ntr:]]

    gcfg = replace(cfg.ga, max_generations=40 if quick else cfg.ga.max_generations)
    print(f"     GA: population {gcfg.population}, crossover {gcfg.crossover_prob}, "
          f"mutation {gcfg.mutation_prob}, stop (1-f) <= {gcfg.epsilon}, max {gcfg.max_generations} gen")
    t0 = time.time()
    res = run_ga(FNN(3, cfg.signal.e_max, cfg.ga.offset_frac), Xtr, ytr, gcfg,
                 np.random.default_rng(cfg.seed), Xva, yva, verbose=True)
    print(f"     done in {time.time() - t0:.0f} s: E_train={res.best_error:.4f} "
          f"E_val={res.history['val'][-1]:.4f}, converged={res.converged} after {res.generations} gen")
    net = FNN(3, cfg.signal.e_max, cfg.ga.offset_frac)
    net.genome = res.best_genome
    net.save(R.MODELS / "fnn3_trained.npz")
    out["ga"] = {"train_E": res.best_error, "val_E": res.history["val"][-1], "generations": res.generations,
                 "converged": res.converged, "initial_E": res.history["best"][0],
                 "rmse_s": float(np.sqrt(res.best_error) * cfg.signal.e_max),
                 "n_train": int(ntr), "n_val": int(len(idx) - ntr)}
    pd.DataFrame({"generation": np.arange(1, len(res.history["best"]) + 1), **res.history}).to_csv(
        R.TAB / "a_ga_history.csv", index=False)
    fig_convergence(res, gcfg.epsilon)
    rb = tables_rule_base(net)
    out["rules_changed_term"] = int((rb["dt (initial)"] != rb["dt trained (term)"]).sum())
    fig_membership(net)

    print("[A3] GA sensitivity: population size and mutation rate ...")
    agg, curves = ga_sweep(lambda: FNN(3, cfg.signal.e_max, cfg.ga.offset_frac), Xtr, ytr, Xva, yva, cfg,
                           seeds=[0] if quick else [0, 1, 2], max_gen=30 if quick else 150)
    R.write_table(agg, "a_ga_sweep", floatfmt=4)
    fig_sweep(curves)

    print("[A4/A5] Isolated intersection: fixed-cycle vs FLC vs FNN-GA ...")
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    duration = cfg.warmup_s + (1800 if quick else 7200)
    rows = []
    for sname, (ns, ew) in SCENARIOS.items():
        rates = side_rates(ns, ew, cfg.road.lanes)
        greens = webster_plan(rates, cfg)
        ctrls = {
            "Fixed (Webster)": lambda g=greens: FixedController(g),
            "FLC (paper rules)": lambda: FNNController(FNN(3, cfg.signal.e_max, cfg.ga.offset_frac), cfg),
            "FNN-GA": lambda: FNNController(net, cfg),
        }
        for r in evaluate_isolated(ctrls, rates, cfg, seeds, duration):
            r.update(scenario=sname, flow_ns=ns, flow_ew=ew, webster_greens=f"{greens[0]:.0f}/{greens[1]:.0f}")
            rows.append(r)
    agg = R.summarise(rows, ["scenario", "controller"])
    R.write_table(agg, "a_baseline_comparison")
    fig_baseline(agg)
    out["baseline"] = agg[["scenario", "controller", "avg_queue", "avg_wait"]].to_dict("records")
    R.write_json(out, "a_summary")
    print(agg[["scenario", "controller", "avg_queue", "avg_wait", "p95_wait"]].round(2).to_string(index=False))
    return net, out
