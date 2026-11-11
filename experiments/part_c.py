"""Part C - METR-LA real sensor data replaces the synthetic Poisson rates."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semaphore import fuzzy
from semaphore import metrla as M
from semaphore import report as R
from semaphore.config import Config
from semaphore.controllers import FixedController, FNNController, webster_plan, expected_side_flows
from semaphore.fnn import FNN
from semaphore.ga import run_ga
from semaphore.oracle import collect_on_rates
from semaphore.sim import GridSim, SIDES

from .part_a import part_a_config
from .part_b import ROWS, COLS

DAY = 86400


def day_slice(df, day):
    return df.loc[day:day + pd.Timedelta("23:55:00")]


# ---------------------------------------------------------------------------
def fig_profile(raw, sensors, day):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    d = day_slice(raw, day)[sensors[:4]].replace(0, np.nan)
    hrs = (d.index - d.index[0]).total_seconds() / 3600
    for i, s in enumerate(sensors[:4]):
        axes[0].plot(hrs, d[s], color=R.SERIES[i], lw=1.4, label=f"sensor {s}")
    axes[0].set_xlabel(f"hour of {day.date()} ({day.day_name()})")
    axes[0].set_ylabel("speed (mph)")
    axes[0].set_xticks(range(0, 25, 3))
    axes[0].set_title("Raw METR-LA speed, representative weekday (4 sensors of the single intersection)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    miss = ((raw == 0) | raw.isna()).mean(axis=1)
    daily = miss.groupby(miss.index.normalize()).mean()
    axes[1].bar(daily.index, daily.values * 100, width=0.9, color=R.SERIES[0], edgecolor=R.SURFACE, linewidth=0.5)
    axes[1].set_ylabel("missing readings (% of all sensors)")
    axes[1].set_title("Missing data per day (0 mph = missing)")
    axes[1].tick_params(axis="x", rotation=30)
    return R.save(fig, "c_raw_profile.png")


def fig_mapping(clean_day, lam, ci, Q, sensors):
    pick = [sensors[0], sensors[3]]           # one AM-peak, one PM-peak sensor (by selection order)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    hrs = (clean_day.index - clean_day.index[0]).total_seconds() / 3600
    for i, s in enumerate(pick):
        axes[0].plot(hrs, clean_day[s], color=R.SERIES[i], label=f"sensor {s}")
        axes[1].plot(hrs, lam[s], color=R.SERIES[i])
        axes[2].plot(hrs, Q[s], color=R.SERIES[i])
    axes[0].set_ylabel("cleaned speed (mph)")
    axes[1].set_ylabel("derived arrival rate\n(veh/h/lane)")
    axes[2].set_ylabel("speed-drop queue\nestimate (veh/lane)")
    axes[2].set_xlabel("hour of day")
    axes[2].set_xticks(range(0, 25, 3))
    axes[0].legend(fontsize=8)
    axes[0].set_title("Sensor -> input mapping: speed drop -> arrival rate (closed loop) and queue estimate (check)")
    return R.save(fig, "c_sensor_mapping.png")


def fig_rules(share_syn, share_real, label):
    names = ["/".join(k) for k, _ in fuzzy.rule_table_3()]
    fig, ax = plt.subplots(figsize=(15, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.2, share_syn * 100, 0.4, color=R.SERIES[0], label="synthetic (constant-rate Poisson)")
    ax.bar(x + 0.2, share_real * 100, 0.4, color=R.SERIES[1], label="real (METR-LA day)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("share of total firing strength (%)")
    ax.set_xlabel("rule antecedent  qg / qr / AV")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    ax.set_title(f"Rule usage, {label}: synthetic vs real input distribution")
    return R.save(fig, f"c_rule_firing_{label.replace(' ', '_')}.png")


def fig_inputs(Xs, Xr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, j, var in zip(axes, range(3), (fuzzy.QG, fuzzy.QR, fuzzy.AV)):
        hi = max(var.universe[1], np.percentile(np.r_[Xs[:, j], Xr[:, j]], 99.5))
        bins = np.linspace(0, hi, 40)
        ax.hist(Xs[:, j], bins, color=R.SERIES[0], alpha=0.7, density=True, label="synthetic")
        ax.hist(Xr[:, j], bins, color=R.SERIES[1], alpha=0.6, density=True, label="real")
        last = var.params[-1][1]
        ax.axvline(last, color=R.TEXT2, lw=1, ls="--")
        ax.annotate(f"{var.terms[-1]} = 1\nabove {last:g}", (last, ax.get_ylim()[1] * 0.5),
                    xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=7, color=R.TEXT2)
        ax.set_yscale("log")
        ax.set_title(f"{var.name} at decision points")
        ax.set_xlabel("veh/lane")
    axes[0].legend(fontsize=8)
    return R.save(fig, "c_input_distributions.png")


def fig_day(demand_mean, keeps, cfg, name):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    hrs = np.arange(len(demand_mean)) / 3600
    axes[0].plot(hrs[::60], demand_mean[::60], color=R.TEXT2, lw=1.5)
    axes[0].set_ylabel("mean entry demand\n(veh/h/lane)")
    for n, k in keeps.items():
        ts = np.array(k["ts"])
        axes[1].plot(ts[:, 0] / 3600, ts[:, 1] + ts[:, 2], color=R.CTRL_COLOR[n], label=n, lw=1.4)
    axes[1].set_ylabel("vehicles queued\n(stop lines + entry buffers)")
    axes[1].set_xlabel("hour of day")
    axes[1].set_xticks(range(0, 25, 3))
    axes[1].legend(fontsize=8)
    axes[0].set_title(f"{name}: real METR-LA day")
    return R.save(fig, f"c_day_{name.replace(' ', '_')}.png")


def fig_queue_check(Q_sensor_N, sim_q_N):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    h1 = np.arange(len(Q_sensor_N)) * 5 / 60
    ax.plot(h1, Q_sensor_N, color=R.SERIES[0], label="queue estimated from sensor speed drop")
    h2 = np.arange(len(sim_q_N)) * 5 / 60
    ax.plot(h2, sim_q_N, color=R.SERIES[1], label="simulated stop-line queue (FNN-GA), 5-min mean")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("veh/lane, north approach")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=8)
    ax.set_title("Sanity check: sensor-side queue estimate vs closed-loop simulated queue")
    return R.save(fig, "c_queue_check.png")


def fig_summary(single, network):
    """Deliverable plot: queue and wait, baseline vs coordinated vs real-data-driven, both scales."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for r, (df, title) in enumerate(((single, "single intersection"), (network, "3x3 network"))):
        for c, (m, lab) in enumerate((("avg_queue", "average queue (veh/approach)"),
                                      ("avg_wait", "average wait per stop (s)"))):
            ax = axes[r, c]
            R.grouped_bars(ax, df, m, "input", "controller", R.CTRL_COLOR)
            ax.set_ylabel(lab)
            ax.set_title(title)
        axes[r, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Queue length and wait time: baseline vs coordinated vs real-data-driven (24 h METR-LA day)",
                 fontsize=11, fontweight="bold")
    return R.save(fig, "summary_queue_wait.png")


# ---------------------------------------------------------------------------
def firing_share(ctrls):
    tot = sum(c.firing_sum for c in ctrls)
    tot = tot.reshape(-1, len(fuzzy.D.terms)).sum(axis=1) if len(tot) == 144 else tot
    return tot / tot.sum()


def run(base_cfg: Config, net3: FNN, net4: FNN, data_path, quick=False):
    R.setup()
    mc = M.MappingConfig()
    print(f"[C1] Loading {data_path}")
    raw = M.load_metrla(data_path)
    prof = M.profile(raw)
    clean, missing0, clean_rep = M.clean(raw, mc)
    sensors, cand = M.choose_sensors(raw, 12, mc)
    days = M.choose_days(raw, sensors, 2)
    eval_day, train_day = days[0], days[1]
    prof.update({f"cleaning_{k}": v for k, v in clean_rep.items()})
    prof.update({"selected_sensors": sensors, "eval_day": str(eval_day.date()),
                 "retune_train_day": str(train_day.date()), "sensors_passing_selection": len(cand)})
    R.write_json(prof, "c_data_profile")
    R.write_table(pd.DataFrame([{"field": k, "value": str(v)} for k, v in prof.items()]), "c_data_profile")
    for k in ("time_start", "time_end", "sampling_interval", "missing_frac_total", "timesteps_all_sensors_missing"):
        print(f"     {k}: {prof[k]}")
    fig_profile(raw, sensors, eval_day)

    vf = M.free_flow_speed(clean[sensors], mc)
    lam_e, ci_e, Q_e = M.speed_to_demand(day_slice(clean, eval_day)[sensors], vf, mc)
    lam_t, _, _ = M.speed_to_demand(day_slice(clean, train_day)[sensors], vf, mc)
    fig_mapping(day_slice(clean, eval_day)[sensors], lam_e, ci_e, Q_e, sensors)
    sel = pd.DataFrame({"sensor": sensors, "v_free_mph": vf.values,
                        "completeness": cand.loc[sensors, "complete"].values,
                        "weekday_p05_mph": cand.loc[sensors, "p05"].values,
                        "eval_day_mean_lambda_vphpl": lam_e.mean().values,
                        "eval_day_peak_lambda_vphpl": lam_e.max().values,
                        "role": [f"single: {SIDES[i]} / grid entry {i}" if i < 4 else f"grid entry {i}"
                                 for i in range(12)]})
    R.write_table(sel, "c_selected_sensors")

    duration = 6 * 3600 if quick else DAY
    seeds = [0] if quick else [0, 1, 2]
    lanes = base_cfg.road.lanes

    # ---- single intersection ---------------------------------------------
    print("[C3] Single intersection, 24 h, real vs synthetic-equivalent arrivals ...")
    cfgA = part_a_config(base_cfg)
    real4 = M.to_per_second(lam_e[sensors[:4]], lanes)[:, :duration]
    syn4 = np.repeat(real4.mean(axis=1, keepdims=True), duration, axis=1)
    greens = webster_plan(real4.mean(axis=1), cfgA)

    print("[C4] Re-tuning the GA on a different real day (same GA parameters) ...")
    real4_train = M.to_per_second(lam_t[sensors[:4]], lanes)[:, :duration]
    Xr, yr = collect_on_rates(cfgA, real4_train, duration, seed=base_cfg.seed)
    gq = replace(cfgA.ga, max_generations=40 if quick else cfgA.ga.max_generations)
    res = run_ga(FNN(3, cfgA.signal.e_max, cfgA.ga.offset_frac), Xr, yr, gq, np.random.default_rng(base_cfg.seed))
    net_rt = FNN(3, cfgA.signal.e_max, cfgA.ga.offset_frac)
    net_rt.genome = res.best_genome
    net_rt.save(R.MODELS / "fnn3_retuned_real.npz")
    retune = {"train_day": str(train_day.date()), "samples": int(len(yr)), "train_E": res.best_error,
              "generations": res.generations, "converged": res.converged,
              "ga_params": {"population": gq.population, "crossover": gq.crossover_prob,
                            "mutation": gq.mutation_prob, "epsilon": gq.epsilon}}
    w0, w1 = net3.split(net3.genome)[0][0], net_rt.split(net_rt.genome)[0][0]
    rt_rules = pd.DataFrame([{"rule qg/qr/AV": "/".join(k), "dt synthetic-trained (s)": round(float(a), 1),
                              "dt real-retuned (s)": round(float(b), 1), "change (s)": round(float(b - a), 1)}
                             for (k, _), a, b in zip(fuzzy.rule_table_3(), w0, w1)])
    R.write_table(rt_rules.reindex(rt_rules["change (s)"].abs().sort_values(ascending=False).index), "c_retuned_rules")

    single_rows, logs, q_hook = [], {}, {}
    for inp, rates in (("synthetic", syn4), ("real (METR-LA)", real4)):
        ctrls = {"Fixed (Webster)": lambda: FixedController(greens),
                 "FNN-GA": lambda: FNNController(net3, cfgA, log_firing=True),
                 "FNN-GA retuned": lambda: FNNController(net_rt, cfgA, log_firing=True)}
        for name, mk in ctrls.items():
            for s in seeds:
                ctrl = mk()
                sim = GridSim(1, 1, {(0, 0): ctrl}, cfgA, seed=3000 + s)
                buf = []
                hook = (lambda sm, t, b=buf: b.append(len(sm.links[(0, 0)][0].queue))) \
                    if (name == "FNN-GA" and s == seeds[0] and inp != "synthetic") else None
                m = sim.run(duration, rates, hooks=[hook] if hook else None)
                single_rows.append({"input": inp, "controller": name, "seed": s,
                                    **{k: m[k] for k in R.METRIC_COLS}})
                if s == seeds[0] and name != "Fixed (Webster)":
                    logs[(inp, name)] = ctrl
                if hook:
                    q_hook["N"] = np.array(buf)
    single = R.summarise(single_rows, ["input", "controller"])
    R.write_table(single, "c_single_intersection")
    print(single[["input", "controller", "avg_queue", "avg_wait", "p95_wait", "unfinished"]].round(2).to_string(index=False))

    if "N" in q_hook:
        simq = q_hook["N"][: (len(q_hook["N"]) // 300) * 300].reshape(-1, 300).mean(axis=1) / lanes
        fig_queue_check(Q_e[sensors[0]].values[: len(simq)], simq)

    # rule usage + input distributions (single intersection, FNN-GA)
    c_syn, c_real = logs[("synthetic", "FNN-GA")], logs[("real (METR-LA)", "FNN-GA")]
    sh_s, sh_r = firing_share([c_syn]), firing_share([c_real])
    fig_rules(sh_s, sh_r, "single intersection")
    Xs, Xrl = np.array(c_syn.inputs_log), np.array(c_real.inputs_log)
    np.savez(R.TAB / "c_decision_inputs.npz", synthetic=Xs, real=Xrl)
    fig_inputs(Xs, Xrl)
    names = ["/".join(k) for k, _ in fuzzy.rule_table_3()]
    rules_df = pd.DataFrame({"rule qg/qr/AV": names, "share_synthetic_%": sh_s * 100, "share_real_%": sh_r * 100,
                             "dt_trained_s": w0})
    rules_df["real/synthetic"] = rules_df["share_real_%"] / rules_df["share_synthetic_%"].clip(lower=1e-6)
    R.write_table(rules_df.sort_values("share_real_%", ascending=False), "c_rule_firing_single", floatfmt=3)
    new_rules = rules_df[(rules_df["share_synthetic_%"] < 0.1) & (rules_df["share_real_%"] >= 0.1)]
    inputs_stats = {
        "decisions_synthetic": int(len(Xs)), "decisions_real": int(len(Xrl)),
        "qg_p99_syn": float(np.percentile(Xs[:, 0], 99)), "qg_p99_real": float(np.percentile(Xrl[:, 0], 99)),
        "qr_p99_syn": float(np.percentile(Xs[:, 1], 99)), "qr_p99_real": float(np.percentile(Xrl[:, 1], 99)),
        "share_qr_in_VL_syn": float(np.mean(Xs[:, 1] >= 17)), "share_qr_in_VL_real": float(np.mean(Xrl[:, 1] >= 17)),
        "share_qg_above_universe_real": float(np.mean(Xrl[:, 0] > 20)),
        "share_qr_above_universe_real": float(np.mean(Xrl[:, 1] > 20)),
        "rules_firing_only_under_real_data": new_rules["rule qg/qr/AV"].tolist(),
    }

    # ---- network --------------------------------------------------------
    print("[C3] 3x3 network, 24 h, real vs synthetic-equivalent arrivals ...")
    cfgB = base_cfg
    probe = GridSim(ROWS, COLS, lambda _: FixedController([20, 20]), cfgB)
    real12 = M.to_per_second(lam_e[sensors], lanes)[:, :duration]
    syn12 = np.repeat(real12.mean(axis=1, keepdims=True), duration, axis=1)
    ef = expected_side_flows(ROWS, COLS, real12.mean(axis=1), probe.entries, cfgB.road.turn_probs)
    net_rows, keeps_real, net_logs = [], {}, {}
    nseeds = seeds[:2]
    for inp, rates in (("synthetic", syn12), ("real (METR-LA)", real12)):
        ctrls = {"Fixed (Webster)": lambda n: FixedController(webster_plan(ef[n], cfgB)),
                 "FNN-GA naive": lambda n: FNNController(net3, cfgB, log_firing=True),
                 "FNN-GA coordinated": lambda n: FNNController(net4, cfgB, log_firing=True)}
        for name, mk in ctrls.items():
            for s in nseeds:
                sim = GridSim(ROWS, COLS, mk, cfgB, seed=4000 + s)
                m = sim.run(duration, rates)
                net_rows.append({"input": inp, "controller": name, "seed": s, **{k: m[k] for k in R.METRIC_COLS}})
                if s == nseeds[0]:
                    if inp != "synthetic":
                        keeps_real[name] = {"ts": m["ts_queue"]}
                    if name != "Fixed (Webster)":
                        net_logs[(inp, name)] = [sig.ctrl for sig in sim.signals.values()]
    network = R.summarise(net_rows, ["input", "controller"])
    R.write_table(network, "c_network")
    print(network[["input", "controller", "avg_queue", "avg_wait", "p95_wait", "worst_node_wait",
                   "blocked_green_s", "unfinished"]].round(2).to_string(index=False))
    fig_day(real12.mean(axis=0) * 3600 / lanes, keeps_real, cfgB, "3x3 network")
    fig_rules(firing_share(net_logs[("synthetic", "FNN-GA coordinated")]),
              firing_share(net_logs[("real (METR-LA)", "FNN-GA coordinated")]), "network coordinated")
    D_real = np.array([x[3] for c in net_logs[("real (METR-LA)", "FNN-GA coordinated")] for x in c.inputs_log])
    D_syn = np.array([x[3] for c in net_logs[("synthetic", "FNN-GA coordinated")] for x in c.inputs_log])
    inputs_stats.update({"D_share_congested_or_blocked_syn": float(np.mean(D_syn >= 0.5)),
                         "D_share_congested_or_blocked_real": float(np.mean(D_real >= 0.5))})

    fig_summary(single, network)
    summary = {"profile": prof, "retune": retune, "inputs": inputs_stats,
               "single": single[["input", "controller", "avg_queue", "avg_wait", "p95_wait"]].to_dict("records"),
               "network": network[["input", "controller", "avg_queue", "avg_wait", "p95_wait"]].to_dict("records")}
    R.write_json(summary, "c_summary")
    print("     rules firing only under real data:", inputs_stats["rules_firing_only_under_real_data"])
    return summary
