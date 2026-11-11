"""Five-layer fuzzy neural network (Part A.2).

    Layer 1  fuzzification     mu_{v,t}(x_v)       trapezoid MFs, centres trainable
    Layer 2  rule (product)    phi_r = prod_v mu_{v,t_r(v)}
    Layer 3  normalisation     phibar_r = phi_r / sum_r phi_r
    Layer 4  defuzzification   phibar_r * w_r      w_r = consequent centre of rule r
    Layer 5  output            dt = sum_r phibar_r * w_r

Trainable weights ("genome" for the GA):
    w      one consequent per rule, bounded to [0, e_max]
    off    one shift per input term (moves the whole trapezoid), bounded to
           +- offset_frac * width of that variable's universe

Initialised from the fuzzy rule base (w = term centres, off = 0), so the
untrained network *is* the fuzzy controller in fuzzy.py.
"""
from __future__ import annotations

import numpy as np

from . import fuzzy


def _trap(x, a, b, c, d):
    """Broadcasting trapezoid; a/b/c/d may contain +-inf (shoulders)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        left = np.where(np.isneginf(a), 1.0, np.clip((x - a) / np.maximum(b - a, 1e-12), 0, 1))
        right = np.where(np.isposinf(d), 1.0, np.clip((d - x) / np.maximum(d - c, 1e-12), 0, 1))
    return np.minimum(left, right)


class FNN:
    def __init__(self, n_inputs: int = 3, e_max: float = 30.0, offset_frac: float = 0.15):
        self.vars = fuzzy.INPUTS_3 if n_inputs == 3 else fuzzy.INPUTS_4
        self.n_inputs = n_inputs
        self.e_max = e_max
        self.offset_frac = offset_frac
        self.term_counts = [len(v.terms) for v in self.vars]
        self.n_rules = int(np.prod(self.term_counts))
        self.n_terms = int(sum(self.term_counts))
        self.n_genes = self.n_rules + self.n_terms
        # Static MF breakpoints, shape (n_terms, 4)
        self.base_params = np.array([p for v in self.vars for p in v.params], dtype=float)
        self.term_var = np.repeat(np.arange(n_inputs), self.term_counts)
        widths = np.array([v.universe[1] - v.universe[0] for v in self.vars])
        self.off_bound = offset_frac * widths[self.term_var]
        self.lo = np.concatenate([np.zeros(self.n_rules), -self.off_bound])
        self.hi = np.concatenate([np.full(self.n_rules, e_max), self.off_bound])
        self.genome = self.initial_genome()

    # -- genome helpers ---------------------------------------------------
    def initial_genome(self) -> np.ndarray:
        w = fuzzy.initial_consequents(self.n_inputs)
        return np.concatenate([w, np.zeros(self.n_terms)])

    def split(self, g):
        g = np.atleast_2d(g)
        return g[:, : self.n_rules], g[:, self.n_rules:]

    def mf_params(self, genome=None):
        """Effective (n_terms, 4) MF breakpoints for one genome (for plots)."""
        g = self.genome if genome is None else genome
        off = g[self.n_rules:]
        return self.base_params + off[:, None]

    # -- forward pass -----------------------------------------------------
    def layers(self, X, genome=None):
        """Single genome, returns every layer's activations (for inspection)."""
        g = self.genome if genome is None else genome
        X = np.atleast_2d(np.asarray(X, float))
        w, off = self.split(g)
        P = self.base_params + off[0][:, None]
        mu = _trap(X[:, self.term_var], P[:, 0], P[:, 1], P[:, 2], P[:, 3])  # L1
        phi = self._rule_layer(mu[None])[0]                                  # L2
        s = phi.sum(axis=1, keepdims=True)
        phibar = np.where(s > 1e-12, phi / np.maximum(s, 1e-12), 0.0)        # L3
        contrib = phibar * w[0]                                              # L4
        out = contrib.sum(axis=1)                                            # L5
        return {"mu": mu, "phi": phi, "phibar": phibar, "contrib": contrib, "out": out}

    def _rule_layer(self, mu):
        """mu: (P, N, n_terms) -> phi: (P, N, n_rules), C-order rule grid."""
        splits = np.cumsum(self.term_counts)[:-1]
        parts = np.split(mu, splits, axis=2)
        phi = parts[0]
        for p in parts[1:]:
            phi = (phi[..., :, None] * p[..., None, :]).reshape(phi.shape[0], phi.shape[1], -1)
        return phi

    def forward_population(self, genomes, X, chunk: int = 32):
        """genomes: (P, n_genes), X: (N, n_inputs) -> (P, N) outputs."""
        genomes = np.atleast_2d(genomes)
        X = np.asarray(X, float)
        out = np.empty((genomes.shape[0], X.shape[0]))
        xt = X[:, self.term_var][None]                       # (1, N, T)
        for s in range(0, genomes.shape[0], chunk):
            g = genomes[s:s + chunk]
            w, off = self.split(g)
            P = self.base_params[None] + off[:, :, None]      # (p, T, 4)
            mu = _trap(xt, P[:, None, :, 0], P[:, None, :, 1], P[:, None, :, 2], P[:, None, :, 3])
            phi = self._rule_layer(mu)
            den = phi.sum(axis=2)
            num = np.einsum("pnr,pr->pn", phi, w)
            out[s:s + chunk] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        return out

    def __call__(self, *x) -> float:
        return float(self.forward_population(self.genome[None], np.array([x]))[0, 0])

    def firing(self, x):
        """Normalised firing strengths for one input vector (rule-usage stats)."""
        return self.layers(np.array([x]))["phibar"][0]

    # -- persistence ------------------------------------------------------
    def save(self, path):
        np.savez(path, genome=self.genome, n_inputs=self.n_inputs, e_max=self.e_max)

    @classmethod
    def load(cls, path, offset_frac: float = 0.15):
        z = np.load(path)
        net = cls(int(z["n_inputs"]), float(z["e_max"]), offset_frac)
        net.genome = z["genome"]
        return net

    def expand_to_4_inputs(self) -> "FNN":
        """Part B: build the 4-input network from a trained 3-input one.

        Trained consequents are kept for D=Free, halved for D=Congested and
        zeroed for D=Blocked (fuzzy.D_FACTOR); trained MF offsets are kept and
        the new D terms start un-shifted.
        """
        assert self.n_inputs == 3
        net = FNN(4, self.e_max, self.offset_frac)
        w3, off3 = self.split(self.genome)
        factors = np.array([fuzzy.D_FACTOR[d] for d in fuzzy.D.terms])
        w4 = (w3[0][:, None] * factors[None, :]).reshape(-1)
        net.genome = np.concatenate([w4, off3[0], np.zeros(len(fuzzy.D.terms))])
        return net
