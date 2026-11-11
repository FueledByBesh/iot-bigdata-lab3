"""Quick correctness checks:  python -m pytest -q   (or  python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semaphore import fuzzy  # noqa: E402
from semaphore.config import DEFAULT  # noqa: E402
from semaphore.controllers import FixedController  # noqa: E402
from semaphore.fnn import FNN  # noqa: E402
from semaphore.ga import run_ga  # noqa: E402
from semaphore.sim import GridSim  # noqa: E402


def test_paper_rules_reproduced():
    table = dict(fuzzy.rule_table_3())
    assert len(table) == 48
    for k, v in fuzzy.PAPER_RULES.items():
        assert table[k] == v


def test_untrained_fnn_equals_flc():
    flc, net = fuzzy.FuzzyController(3), FNN(3)
    X = np.random.default_rng(0).uniform([0, 0, 0], [25, 25, 8], (300, 3))
    a = np.array([flc.infer(*x) for x in X])
    assert np.allclose(a, net.forward_population(net.genome[None], X)[0])
    flc4, net4 = fuzzy.FuzzyController(4), FNN(3).expand_to_4_inputs()
    X4 = np.c_[X, np.random.default_rng(1).uniform(0, 1, 300)]
    assert np.allclose([flc4.infer(*x) for x in X4], net4.forward_population(net4.genome[None], X4)[0])


def test_blocked_downstream_gives_zero():
    net4 = FNN(3).expand_to_4_inputs()
    assert net4(20, 0, 6, 1.0) < 1e-9          # long green queue, but nowhere to go
    assert net4(20, 0, 6, 0.0) > 20            # same state, free downstream -> long extension


def test_ga_improves_on_known_target():
    net = FNN(3)
    X = np.random.default_rng(0).uniform([0, 0, 0], [20, 20, 6], (400, 3))
    y = np.clip(1.2 * X[:, 0] - 0.8 * X[:, 1] + 2 * X[:, 2], 0, 30)
    from dataclasses import replace
    res = run_ga(net, X, y, replace(DEFAULT.ga, max_generations=30), np.random.default_rng(0))
    assert res.history["best"][-1] < res.history["best"][0]


def test_vehicle_conservation():
    sim = GridSim(3, 3, lambda n: FixedController([20, 20]), DEFAULT, seed=0)
    rates = np.full(len(sim.entries), 0.1)
    m = sim.run(1200, rates, warmup=0)
    assert m["throughput_vph"] > 0 and m["avg_wait"] >= 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
