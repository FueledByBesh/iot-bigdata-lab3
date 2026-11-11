"""All tunable parameters in one place.

Every number here is a design decision. The ones marked [PAPER] come from the
source presentation; the ones marked [OURS] are choices made for this
re-implementation and should be justified in the report.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalConfig:
    sample_time: float = 10.0      # [PAPER] sampling time t = 10 s (decision period)
    g_min: float = 10.0            # [PAPER] g(min) = 10 s
    e_max: float = 30.0            # [PAPER] e(max) = 30 s  -> max green = 40 s
    intergreen: float = 5.0        # [OURS] 3 s yellow + 2 s all-red, no discharge
    startup_lost: float = 2.0      # [OURS] start-up lost time at the beginning of green
    # Fixed-cycle (Webster) baseline limits. The deck quotes a 40-120 s cycle.
    cycle_min: float = 40.0        # [PAPER] "minimum of cycle is about 40 s"
    cycle_max: float = 120.0       # [PAPER] "maximum of cycle is about 120 s"


@dataclass(frozen=True)
class RoadConfig:
    lanes: int = 3                 # [PAPER] 4 legs x 3 lanes
    sat_flow_per_lane: float = 0.5 # [OURS] 1800 veh/h/lane (HCM base saturation flow)
    link_length_m: float = 200.0   # [OURS] block length between intersections
    free_flow_kmh: float = 50.0    # [OURS] urban speed limit
    jam_spacing_m: float = 7.5     # [OURS] storage per vehicle in a queue
    detector_horizon_s: float = 10.0  # [OURS] AV = vehicles <= 10 s from the stop line
    # Turning probabilities at every intersection (straight, right, left).
    # The paper uses straight-only; (1, 0, 0) reproduces that exactly.
    turn_probs: tuple = (0.8, 0.1, 0.1)  # [OURS] for the grid; Part A uses (1,0,0)

    @property
    def travel_time_s(self) -> int:
        return int(round(self.link_length_m / (self.free_flow_kmh / 3.6)))

    @property
    def storage_veh(self) -> int:
        return int(self.link_length_m * self.lanes / self.jam_spacing_m)

    @property
    def sat_flow(self) -> float:
        return self.sat_flow_per_lane * self.lanes


@dataclass(frozen=True)
class GAConfig:
    population: int = 220          # [PAPER]
    crossover_prob: float = 0.5    # [PAPER]
    mutation_prob: float = 0.01    # [PAPER] per gene
    epsilon: float = 0.01          # [PAPER] stop when (1 - f_best) <= epsilon
    max_generations: int = 200     # [OURS] safety cap if epsilon is never reached
    tournament_k: int = 3          # [OURS]
    elitism: int = 2               # [OURS]
    blx_alpha: float = 0.25        # [OURS] BLX-alpha blend crossover
    mutation_sigma: float = 0.10   # [OURS] Gaussian mutation, fraction of gene range
    init_sigma: float = 0.10       # [OURS] spread of initial population around the rule base
    offset_frac: float = 0.15      # [OURS] MF centres may shift +-15 % of their universe


@dataclass(frozen=True)
class OracleConfig:
    """Teacher used to label training samples for the GA (see oracle.py)."""
    horizon_s: int = 150           # [OURS] > one max cycle (2*(40+5)=90 s)
    candidates: tuple = tuple(range(0, 31, 2))  # [OURS] candidate extensions, s
    n_futures: int = 4             # [OURS] Poisson futures averaged per candidate
    n_scenarios: int = 40          # [OURS] random demand scenarios for data collection
    scenario_duration_s: int = 1800
    explore_eps: float = 0.3       # [OURS] random actions in the behaviour policy
    flow_range_vphpl: tuple = (100.0, 850.0)  # [OURS] per-lane demand range sampled


@dataclass(frozen=True)
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    warmup_s: int = 600            # [OURS] metrics ignore the first 10 minutes
    seed: int = 7


DEFAULT = Config()
