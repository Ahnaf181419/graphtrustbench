"""N5: an analytic model of how much corruption reaches the one-hop retriever.

Corruption picks n = floor(f*|E|) of the task graph's |E| edges uniformly without replacement.
The retriever reads |R| of them (its five clean seeds' collapsed out-edges). The delivered dose k
is then hypergeometric: k ~ Hypergeom(population |E|, successes |R|, draws n).
"""
import math
from scipy.stats import hypergeom


def dose_distribution(E, R, f):
    n = math.floor(f * E)
    rv = hypergeom(M=E, n=R, N=n)
    return dict(n=n, mean=float(rv.mean()), p_zero=float(rv.pmf(0)))


def predicted_unchanged_share(rows):
    """rows: iterable of (E, R, f). Mean predicted P(k = 0), i.e. the expected share of cells
    whose clean ranking cannot change. Compare with the observed share of k = 0 cells."""
    rows = list(rows)
    return sum(dose_distribution(E, R, f)['p_zero'] for E, R, f in rows) / len(rows)
