"""Signal controllers: fixed-cycle (Webster) baseline and FNN-based adaptive."""
from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT
from .fnn import FNN
from .sim import PHASE_SIDES, HEAD_VEC, ARRIVE_SIDE, HEADING_FROM, RIGHT, LEFT, SIDES, SIDE_IDX


class FixedController:
    """Pre-timed two-phase plan. kind='fixed' tells the simulator to use green()."""
    kind = "fixed"

    def __init__(self, greens, offset=0):
        self.greens = [int(round(g)) for g in greens]
        self.offset = int(offset)

    def green(self, phase):
        return self.greens[phase]

    @property
    def cycle(self):
        return sum(self.greens)


def webster_plan(side_flows, cfg: Config = DEFAULT):
    """Webster's optimum cycle for a two-phase intersection.

    side_flows: veh/s arriving on each of the 4 sides (N, E, S, W).
    Lost time per phase = start-up lost time + intergreen (no discharge on
    yellow in this simulator). Cycle clamped to the paper's 40-120 s.
    """
    sc, rd = cfg.signal, cfg.road
    y = [max(side_flows[i] for i in sides) / rd.sat_flow for sides in PHASE_SIDES]
    Y = sum(y)
    L = 2 * (sc.startup_lost + sc.intergreen)
    C = sc.cycle_max if Y >= 0.95 else (1.5 * L + 5) / (1 - Y)
    C = float(np.clip(C, sc.cycle_min, sc.cycle_max))
    eff = C - L
    share = [0.5, 0.5] if Y <= 0 else [yi / Y for yi in y]
    greens = [max(sc.g_min, eff * s + sc.startup_lost) for s in share]
    return greens


class FNNController:
    """Adaptive controller: asks the FNN for the green extension dt.

    Inputs are per-lane values of the *critical* approach (the larger of the
    two approaches served by a phase):
        qg = max queue on green sides / lanes
        qr = max queue on red sides   / lanes
        AV = max approaching vehicles (<= 10 s from stop line) on green sides / lanes
        D  = max occupancy ratio of the links the green sides discharge into
             (only for the 4-input, coordinated network)
    """
    kind = "adaptive"

    def __init__(self, net: FNN, cfg: Config = DEFAULT, log_firing=False, name="fnn"):
        self.net = net
        self.g_min = cfg.signal.g_min
        self.log_firing = log_firing
        self.name = name
        self.inputs_log = []
        self.firing_sum = np.zeros(net.n_rules)
        self.firing_max_count = np.zeros(net.n_rules)
        self.n_decisions = 0

    def features(self, obs):
        lanes = obs["lanes"]
        g, r = PHASE_SIDES[obs["phase"]], PHASE_SIDES[1 - obs["phase"]]
        qg = max(obs["queue"][i] for i in g) / lanes
        qr = max(obs["queue"][i] for i in r) / lanes
        av = max(obs["av"][i] for i in g) / lanes
        x = [qg, qr, av]
        if self.net.n_inputs == 4:
            x.append(max(obs["down_occ"][i] for i in g))
        return x

    def decide(self, obs):
        x = self.features(obs)
        self.n_decisions += 1
        if self.log_firing:
            self.inputs_log.append(x)
            fb = self.net.firing(x)
            self.firing_sum += fb
            self.firing_max_count[int(np.argmax(fb))] += 1
        return self.net(*x)


def expected_side_flows(rows, cols, entry_rates, entries, turn_probs, iters=200):
    """Propagate mean boundary demand through the grid with the turn split.

    Returns {node: [flow N, E, S, W]} in veh/s; used to give every
    intersection its own Webster plan in the fixed-cycle network.
    """
    ps, pr, pl = turn_probs
    inflow = {(r, c, s): 0.0 for r in range(rows) for c in range(cols) for s in range(4)}
    base = dict(inflow)
    for L, lam in zip(entries, entry_rates):
        base[(L.node[0], L.node[1], SIDE_IDX[L.side])] += lam
    flows = dict(base)
    for _ in range(iters):
        new = dict(base)
        for (r, c, s), f in flows.items():
            if f <= 0:
                continue
            h = HEADING_FROM[SIDES[s]]
            for nh, p in ((h, ps), (RIGHT[h], pr), (LEFT[h], pl)):
                dr, dc = HEAD_VEC[nh]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and p > 0:
                    new[(nr, nc, SIDE_IDX[ARRIVE_SIDE[nh]])] += f * p
        flows = new
    return {(r, c): [flows[(r, c, s)] for s in range(4)] for r in range(rows) for c in range(cols)}
