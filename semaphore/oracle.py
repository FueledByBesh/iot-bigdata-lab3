"""Training-sample generation for the GA (Part A.3).

The paper trains the FNN by minimising a *squared error* against "sample
values", but does not say where the target extensions come from. [OURS]
We generate them with a simulation-based teacher (one-step rollout):

  At a decision point with state (queues, vehicles already on the approach
  links, current phase, extension used so far) we try every candidate
  extension e in {0, 2, ..., 30} s. For each candidate we simulate the next
  `horizon_s` seconds with a fast fluid model of the same intersection:
    * green continues for e seconds, then intergreen, then the other phase;
    * later greens follow a simple queue-clearance rule (end once the
      critical queue is cleared, within [g_min, g_min + e_max]);
    * future boundary arrivals are drawn from the known Poisson rate, and the
      cost is averaged over `n_futures` independent draws (the teacher does
      *not* see the true future);
    * cost = total vehicle-seconds queued + a terminal cost for the residual
      queue (q^2 / 2c, the fluid clearing delay).
  The target is the candidate with the lowest mean cost.

States are collected by running random-demand scenarios with a behaviour
policy that follows the teacher, with explore_eps random actions mixed in so
the training set also covers states a good controller rarely visits.
"""
from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT
from .controllers import FNNController
from .fnn import FNN
from .sim import GridSim, PHASE_SIDES

PHASE_MASK = np.zeros((2, 4), bool)
for p, sides in enumerate(PHASE_SIDES):
    PHASE_MASK[p, list(sides)] = True


def rollout_costs(state, side_rates, cfg: Config, rng):
    """Mean rollout cost for every candidate extension. Returns (cands, costs)."""
    sc, rd, oc = cfg.signal, cfg.road, cfg.oracle
    H, K = oc.horizon_s, oc.n_futures
    remaining = sc.e_max - state["ext_used"]
    cands = np.array([c for c in oc.candidates if c <= remaining + 1e-9], float)
    if len(cands) == 0:
        cands = np.array([0.0])
    C = len(cands)
    B = C * K
    # arrivals at the stop line: (K, 4, H) futures + known in-transit schedule
    fut = np.zeros((K, 4, H))
    tt = rd.travel_time_s
    if H > tt:
        fut[:, :, tt:] = rng.poisson(np.asarray(side_rates)[None, :, None], (K, 4, H - tt))
    A = fut + state["schedule"][None]
    A = np.repeat(A[None], C, axis=0).reshape(B, 4, H)
    e_c = np.repeat(cands, K)

    q = np.repeat(np.asarray(state["queue"], float)[None], B, axis=0)
    phase = np.full(B, state["phase"])
    green = np.ones(B, bool)
    first = np.ones(B, bool)
    t_state = np.full(B, -float(state["elapsed"]))     # green started this long ago
    cap = rd.sat_flow
    lanes = rd.lanes
    cost = np.zeros(B)
    for k in range(H):
        q += A[:, :, k]
        el = k - t_state
        # transitions
        mask = PHASE_MASK[phase]
        qcrit = np.where(mask, q, 0).max(axis=1) / lanes
        end_first = green & first & (k >= e_c)
        end_next = green & ~first & (((el >= sc.g_min) & (qcrit < 1.0)) | (el >= sc.g_min + sc.e_max))
        to_y = end_first | end_next
        green = green & ~to_y
        t_state = np.where(to_y, k, t_state)
        el = k - t_state
        to_g = ~green & ~to_y & (el >= sc.intergreen)
        phase = np.where(to_g, 1 - phase, phase)
        green = green | to_g
        first = first & ~to_g
        t_state = np.where(to_g, k, t_state)
        el = k - t_state
        # discharge
        can = green & (el >= sc.startup_lost)
        mask = PHASE_MASK[phase] & can[:, None]
        q -= np.where(mask, np.minimum(q, cap), 0.0)
        cost += q.sum(axis=1)
    cost += (q ** 2).sum(axis=1) / (2 * 0.45 * cap)
    return cands, cost.reshape(C, K).mean(axis=1)


def extract_state(sim: GridSim, node, t):
    sig = sim.signals[node]
    H = sim.cfg.oracle.horizon_s
    sched = np.zeros((4, H))
    for si, L in enumerate(sim.links[node]):
        for v in L.transit:
            k = v.t_stop - t
            if 0 <= k < H:
                sched[si, k] += 1
        if L.buffer and L.tt < H:          # entry buffer enters as space frees
            sched[si, L.tt] += len(L.buffer)
    return {"queue": [len(L.queue) for L in sim.links[node]], "schedule": sched,
            "phase": sig.phase, "elapsed": t - sig.t_start, "ext_used": sig.ext_used}


class TeacherController(FNNController):
    """Behaviour policy that records (features, teacher target) pairs."""

    def __init__(self, cfg, side_rates, rng, eps):
        super().__init__(FNN(3, cfg.signal.e_max, cfg.ga.offset_frac), cfg)
        self.cfg, self.rng, self.eps = cfg, rng, eps
        self.side_rates = np.asarray(side_rates, float)
        self.sim = None
        self.X, self.y = [], []

    def decide(self, obs):
        x = self.features(obs)
        state = extract_state(self.sim, (0, 0), obs["t"])
        rates = self.side_rates
        if rates.ndim == 2:            # time-varying (real data): teacher assumes the
            rates = rates[:, min(obs["t"], rates.shape[1] - 1)]   # current rate persists
        cands, costs = rollout_costs(state, rates, self.cfg, self.rng)
        target = float(cands[int(np.argmin(costs))])
        if state["ext_used"] == 0:     # only the first decision of a green has the full 0..30 range
            self.X.append(x)
            self.y.append(target)
        else:                          # later decisions: label is "remaining extension needed"
            self.X.append(x)
            self.y.append(min(self.cfg.signal.e_max, target))
        if self.rng.random() < self.eps:
            return float(self.rng.uniform(0, self.cfg.signal.e_max))
        return target


def collect_training_set(cfg: Config = DEFAULT, seed: int = 0, n_scenarios=None,
                         rate_fn=None, verbose=False):
    """Run random-demand scenarios on the isolated intersection and label states.

    rate_fn(rng) -> per-lane flows (veh/h/lane) for sides N, E, S, W; defaults
    to uniform sampling in oracle.flow_range_vphpl with N~S and E~W.
    """
    oc = cfg.oracle
    rng = np.random.default_rng(seed)
    n_sc = oc.n_scenarios if n_scenarios is None else n_scenarios
    X, y = [], []
    lo, hi = oc.flow_range_vphpl
    for s in range(n_sc):
        if rate_fn is None:
            ns, ew = rng.uniform(lo, hi, 2)
            # keep total degree of saturation feasible for a two-phase signal
            tot = ns + ew
            if tot > 1500:
                ns, ew = ns * 1500 / tot, ew * 1500 / tot
            vph = np.array([ns, ew, ns, ew]) * rng.uniform(0.85, 1.15, 4)
        else:
            vph = np.asarray(rate_fn(rng))
        side_rates = vph * cfg.road.lanes / 3600.0
        ctrl = TeacherController(cfg, side_rates, rng, oc.explore_eps)
        sim = GridSim(1, 1, {(0, 0): ctrl}, cfg, turn_probs=(1, 0, 0),
                      seed=int(rng.integers(1 << 31)), record_timeseries=False)
        ctrl.sim = sim
        sim.run(oc.scenario_duration_s, side_rates, warmup=0)
        X += ctrl.X
        y += ctrl.y
        if verbose:
            print(f"    scenario {s + 1:2d}/{n_sc}  flows={np.round(vph).astype(int)} veh/h/lane  samples={len(X)}")
    return np.array(X), np.array(y)


def collect_on_rates(cfg: Config, side_rates, duration, seed=0):
    """Label states visited while replaying given (4, T) arrival rates (Part C retune)."""
    rng = np.random.default_rng(seed)
    ctrl = TeacherController(cfg, side_rates, rng, cfg.oracle.explore_eps)
    sim = GridSim(1, 1, {(0, 0): ctrl}, cfg, turn_probs=(1, 0, 0), seed=seed, record_timeseries=False)
    ctrl.sim = sim
    sim.run(duration, side_rates, warmup=0)
    return np.array(ctrl.X), np.array(ctrl.y)
