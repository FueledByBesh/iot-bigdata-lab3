"""Real-coded genetic algorithm that tunes the FNN weights (Part A.3).

Follows the training algorithm on slide 13 of the source deck:
    1. input parameters and training samples
    2. fitness from the squared error, FNN weights as the variables
    3. initial population
    4. evaluate -> select -> recombine -> mutate -> new population
    5. stop when (1 - f_best) <= epsilon, else go to 4

Fitness: f = 1 / (1 + E), E = mean squared error of the FNN output against
the target extensions, both divided by e_max (so E is dimensionless and
epsilon = 0.01 means an RMS error of about 0.1 * e_max = 3 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GAConfig
from .fnn import FNN


@dataclass
class GAResult:
    best_genome: np.ndarray
    best_error: float
    generations: int
    converged: bool
    history: dict = field(default_factory=dict)   # per-generation best/mean E, val E


def mse(net: FNN, genomes, X, y):
    pred = net.forward_population(genomes, X)
    return np.mean(((pred - y[None]) / net.e_max) ** 2, axis=1)


def run_ga(net: FNN, X, y, cfg: GAConfig, rng: np.random.Generator,
           X_val=None, y_val=None, verbose: bool = False) -> GAResult:
    lo, hi = net.lo, net.hi
    span = hi - lo
    P, G = cfg.population, net.n_genes

    # Initial population: the rule base itself + Gaussian perturbations of it,
    # so the GA starts from (and can always fall back to) the expert rules.
    g0 = net.initial_genome()
    pop = g0[None] + rng.normal(0, cfg.init_sigma, (P, G)) * span[None]
    pop[0] = g0
    pop = np.clip(pop, lo, hi)
    err = mse(net, pop, X, y)

    hist = {"best": [], "mean": [], "val": []}
    converged = False
    gen = 0
    for gen in range(1, cfg.max_generations + 1):
        order = np.argsort(err)
        best_e = err[order[0]]
        hist["best"].append(float(best_e))
        hist["mean"].append(float(np.mean(err)))
        if X_val is not None:
            hist["val"].append(float(mse(net, pop[order[0]][None], X_val, y_val)[0]))
        f_best = 1.0 / (1.0 + best_e)
        if verbose and (gen % 20 == 0 or gen == 1):
            print(f"    gen {gen:4d}  E_best={best_e:.5f}  1-f={1 - f_best:.5f}")
        if (1.0 - f_best) <= cfg.epsilon:
            converged = True
            break

        # Selection: tournament of size k, lower error wins
        cand = rng.integers(0, P, (P, cfg.tournament_k))
        winners = cand[np.arange(P), np.argmin(err[cand], axis=1)]
        parents = pop[winners]

        # Crossover (prob 0.5 per pair): BLX-alpha blend, gene-wise
        children = parents.copy()
        a, b = parents[0::2], parents[1::2]
        n = min(len(a), len(b))
        do = rng.random(n) < cfg.crossover_prob
        alpha = rng.uniform(-cfg.blx_alpha, 1 + cfg.blx_alpha, (n, G))
        c1 = alpha * a[:n] + (1 - alpha) * b[:n]
        c2 = (1 - alpha) * a[:n] + alpha * b[:n]
        children[0:2 * n:2][do] = c1[do]
        children[1:2 * n:2][do] = c2[do]

        # Mutation (prob 0.01 per gene): Gaussian, sigma = fraction of range
        mask = rng.random((P, G)) < cfg.mutation_prob
        children = children + mask * rng.normal(0, cfg.mutation_sigma, (P, G)) * span[None]
        children = np.clip(children, lo, hi)

        # Elitism: carry the best individuals over unchanged
        children[: cfg.elitism] = pop[order[: cfg.elitism]]
        pop = children
        err = mse(net, pop, X, y)

    i = int(np.argmin(err))
    return GAResult(pop[i].copy(), float(err[i]), gen, converged, hist)
