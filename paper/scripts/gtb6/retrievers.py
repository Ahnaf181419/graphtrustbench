"""PPR and the dropout-ensemble mitigation on the registered data model (second registration, §3).

Graphs are the registered edge lists ({'source', 'target', ...} dicts), candidates and lexical scores
come from localization_data / localization_retrieval, and every retriever ranks files with the
registered rank_files, so ties break the same way as the registered one-hop.
"""
import hashlib
from collections import defaultdict

from localization_retrieval import rank_files, seed_nodes


def _mapped(lexical, edges):
    """Unique mapped out-neighbours: both ends are candidates, no self-loops, parallel edges collapse."""
    out = defaultdict(set)
    for e in edges:
        u, v = e['source'], e['target']
        if u in lexical and v in lexical and u != v:
            out[u].add(v)
    return {u: sorted(vs) for u, vs in out.items()}


def _restart(lexical, seed_count):
    """The seeds' BM25 mass, normalised exactly as one_hop weights its seeds."""
    seeds = seed_nodes(lexical, seed_count)
    w = [max(0, lexical[n]) for n in seeds]
    total = sum(w)
    return {s: (x / total if total else 1 / len(seeds)) for s, x in zip(seeds, w)}


def ppr(lexical, edges, seed_count=5, restart=0.15, tol=1e-8, max_iter=100):
    """Personalized PageRank on the mapped subgraph; dangling mass returns to the restart vector."""
    out, r = _mapped(lexical, edges), _restart(lexical, seed_count)
    x = dict(r)
    for _ in range(max_iter):
        nxt, dangling = defaultdict(float), 0.0
        for u, xu in x.items():
            nb = out.get(u)
            if nb:
                for v in nb:
                    nxt[v] += (1 - restart) * xu / len(nb)
            else:
                dangling += (1 - restart) * xu
        for s, rs in r.items():
            nxt[s] += (restart + dangling) * rs
        diff = sum(abs(nxt.get(n, 0.0) - x.get(n, 0.0)) for n in set(nxt) | set(x))
        x = dict(nxt)
        if diff < tol:
            break
    return x


def _keep(task, j, u, v, p):
    """Edge-hash dropout: an edge is kept or dropped identically in every graph condition of a task."""
    h = hashlib.sha256(f'{task}|{j}|{u}|{v}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2 ** 64 >= p


def dropout_ensemble(lexical, edges, task_id, scorer, m=10, p=0.2):
    """Mitigation: mean node score over m graphs with edge-hash dropout at rate p."""
    total = defaultdict(float)
    for j in range(m):
        kept = [e for e in edges if _keep(task_id, j, e['source'], e['target'], p)]
        for n, s in scorer(lexical, kept).items():
            total[n] += s / m
    return dict(total)


def bm25_order(lexical):
    """Nodes in BM25 order, ties by node id (the registered tie order; seed_nodes' order)."""
    return sorted(lexical, key=lambda n: (-lexical[n], n))


def rank(scores, lexical, candidates):
    """Registered file ranking; candidates a scorer never reached score 0 and follow in BM25 order."""
    full = dict.fromkeys(lexical, 0.0)
    full.update({n: s for n, s in scores.items() if n in lexical})
    return rank_files(full, lexical, candidates)[0]
