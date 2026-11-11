"""Fuzzy logic controller (Part A.1).

Inputs (all normalised *per lane* so the same controller works on any
approach width, and on every intersection of the grid):

    AV  approaching vehicles on the current green phase, i.e. vehicles that
        are <= 10 s of free-flow travel from the stop line (veh/lane)
    qg  queue on the current green phase (veh/lane, critical approach)
    qr  queue on the current red phase   (veh/lane, critical approach)

Output:

    dt  extension of the current green phase (s), 0..e_max

Optional 4th input for Part B (shared fuzzy input):

    D   occupancy ratio (0..1) of the downstream link that the green phase
        discharges into, i.e. how full the neighbour's receiving approach is

Membership functions are trapezoids (a, b, c, d); a triangle is b == c and a
shoulder uses +-inf. The rule AND is the algebraic product (the same operator
as the FNN's rule layer) and defuzzification is the height / centre-of-sets
method, sum(phi_r * c_r) / sum(phi_r), which is exactly what the FNN's
normalisation + defuzzification layers compute. This makes the untrained FNN
identical to this controller (checked in tests/test_core.py).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

INF = math.inf


def trapmf(x, a, b, c, d):
    """Trapezoidal membership, vectorised. Shoulders use a=b=-inf or c=d=inf."""
    x = np.asarray(x, dtype=float)
    if a == -INF:
        left = np.ones_like(x)
    else:
        left = np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)
    if d == INF:
        right = np.ones_like(x)
    else:
        right = np.clip((d - x) / max(d - c, 1e-12), 0.0, 1.0)
    return np.minimum(left, right)


@dataclass(frozen=True)
class Variable:
    name: str
    universe: tuple            # (lo, hi) used for plots and GA offset bounds
    terms: tuple               # term names, in increasing order
    params: tuple              # one (a, b, c, d) per term

    def fuzzify(self, x):
        return np.stack([trapmf(x, *p) for p in self.params], axis=-1)


# --- Linguistic variables -------------------------------------------------
# [OURS] Shapes and breakpoints: the paper gives only the term names. The
# queue universe 0..20 veh/lane covers a 200 m, 3-lane link (26 veh/lane
# jam storage); AV 0..6 covers 10 s of arrivals at saturation (5 veh/lane).
QUEUE_TERMS = ("S", "M", "L", "VL")
QG = Variable("qg", (0.0, 20.0), QUEUE_TERMS,
              ((-INF, -INF, 2, 6), (3, 7, 7, 11), (8, 12, 12, 16), (13, 17, INF, INF)))
QR = Variable("qr", (0.0, 20.0), QUEUE_TERMS, QG.params)
AV = Variable("AV", (0.0, 6.0), ("L", "M", "H"),
              ((-INF, -INF, 0.5, 2), (1, 2.5, 2.5, 4), (3, 4.5, INF, INF)))
D = Variable("D", (0.0, 1.0), ("Free", "Cong", "Block"),
             ((-INF, -INF, 0.5, 0.7), (0.5, 0.75, 0.75, 0.95), (0.8, 0.95, INF, INF)))

# Output terms and their centres (height defuzzification uses the centres).
DT_TERMS = ("Z", "S", "M", "L")
DT_CENTRES = np.array([0.0, 8.0, 16.0, 26.0])
DT = Variable("dt", (0.0, 30.0), DT_TERMS,
              ((-INF, -INF, 0, 8), (0, 8, 8, 16), (8, 16, 16, 26), (16, 26, INF, INF)))

INPUTS_3 = (QG, QR, AV)
INPUTS_4 = (QG, QR, AV, D)

# The six example rules on slide 8 of the source deck: (qg, qr, AV) -> dt
PAPER_RULES = {
    ("S", "S", "L"): "S",
    ("L", "M", "M"): "S",
    ("M", "L", "L"): "Z",
    ("M", "VL", "H"): "Z",
    ("VL", "S", "L"): "L",
    ("S", "VL", "H"): "Z",
}


def base_rule_level(iqg: int, iqr: int, iav: int) -> int:
    """[OURS] Systematic completion of the paper's rule base.

    The output level grows with the green queue and with approaching
    vehicles, and shrinks with the red queue. The weights (0.6, 0.3, 1.5) and
    bias 1 are the simplest values for which all six of the paper's example
    rules are reproduced exactly (asserted below); this makes the full 48-rule
    table a consistent extension of the paper rather than a new design.
    """
    score = 0.6 * iqg + 0.3 * iav - 1.5 * iqr + 1.0
    return int(min(3, max(0, math.floor(score + 0.5))))


def rule_table_3():
    """List of ((qg, qr, AV), dt_term) for the full 4x4x3 = 48 rule grid."""
    rules = []
    for iqg, iqr, iav in itertools.product(range(4), range(4), range(3)):
        key = (QUEUE_TERMS[iqg], QUEUE_TERMS[iqr], AV.terms[iav])
        rules.append((key, DT_TERMS[base_rule_level(iqg, iqr, iav)]))
    table = dict(rules)
    for k, v in PAPER_RULES.items():
        assert table[k] == v, f"rule base contradicts paper rule {k}->{v}"
    return rules


# [OURS] Part B coordination: how the neighbour's occupancy D modifies a rule.
D_FACTOR = {"Free": 1.0, "Cong": 0.5, "Block": 0.0}


def rule_table_4():
    """3x the 48 rules: Free keeps the consequent, Congested halves it,
    Blocked forces zero extension (stop feeding a full downstream link)."""
    out = []
    for (key, term) in rule_table_3():
        c = DT_CENTRES[DT_TERMS.index(term)]
        for d in D.terms:
            out.append((key + (d,), c * D_FACTOR[d]))
    return out


def initial_consequents(n_inputs: int = 3) -> np.ndarray:
    if n_inputs == 3:
        return np.array([DT_CENTRES[DT_TERMS.index(t)] for _, t in rule_table_3()])
    return np.array([c for _, c in rule_table_4()])


class FuzzyController:
    """Plain, readable Mamdani-product / height-defuzzification controller.

    Deliberately written with explicit loops so it can be read next to the
    rule table; the vectorised, trainable version is fnn.FNN.
    """

    def __init__(self, n_inputs: int = 3):
        self.vars = INPUTS_3 if n_inputs == 3 else INPUTS_4
        if n_inputs == 3:
            self.rules = [(k, DT_CENTRES[DT_TERMS.index(t)]) for k, t in rule_table_3()]
        else:
            self.rules = rule_table_4()

    def infer(self, *x) -> float:
        degrees = [dict(zip(v.terms, v.fuzzify(xi))) for v, xi in zip(self.vars, x)]
        num = den = 0.0
        for antecedent, centre in self.rules:
            phi = 1.0
            for deg, term in zip(degrees, antecedent):
                phi *= deg[term]
            num += phi * centre
            den += phi
        return num / den if den > 1e-12 else 0.0
