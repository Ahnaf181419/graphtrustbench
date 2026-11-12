"""G8.5, Table 3 (exploratory): each setting coded against its own C0; forced cells (under that setting) and
INVALID replies dropped; H2 cells paired with their matched placebos (guide G8.5)."""
from collections import defaultdict

from gemma_readout import SPECIAL
from gtb6 import stats as S
from v7_g6_readouts import modal_choice


def code_setting(rows):
    c0 = {r['task']: r['choice'] for r in rows if r['cond'] == 'C0'}
    key = lambda r: (r['task'], r['intervention'], r['seed'])
    ok = lambda r: r['choice'] != 'INVALID' and c0.get(r['task'], 'INVALID') != 'INVALID'
    P = {key(r): float(r['choice'] != c0[r['task']]) for r in rows if r['cond'] == 'P' and ok(r)}
    out = []
    for r in rows:
        if r['cond'] != 'X' or not ok(r) or key(r) not in P:
            continue
        c = c0[r['task']]
        if c not in SPECIAL and c not in r['shown']:          # forced under this setting
            continue
        out.append(dict(task=r['task'], flip=float(r['choice'] != c), pflip=P[key(r)]))
    return out


def _task_mean(pairs, field):
    acc = defaultdict(list)
    for p in pairs:
        acc[p['task']].append(p[field])
    return [sum(v) / len(v) for _, v in sorted(acc.items())]


def setting_row(runs):
    """(H2 flip rate, placebo flip rate, (mean, lo, hi) of the per-task paired difference); runs pooled."""
    pairs = [p for r in runs for p in code_setting(r)]
    for p in pairs:
        p['d'] = p['flip'] - p['pflip']
    h2, pl = _task_mean(pairs, 'flip'), _task_mean(pairs, 'pflip')
    return sum(h2) / len(h2), sum(pl) / len(pl), S.ci(_task_mean(pairs, 'd'))


def modal_rows(draws):
    """Sampled setting: per item, the modal choice over the draws (ties to the earliest draw)."""
    by_id = [{r['id']: r['choice'] for r in d} for d in draws]
    return [dict(r, choice=modal_choice([b[r['id']] for b in by_id])) for r in draws[0]]
