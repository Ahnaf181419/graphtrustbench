"""G8, patch readout: H5 (equivalence of H2-type cells to clean), H6 (Recall@5-drop cells resolve less),
and the Holm family, which is {H6, H7} since H4a/H4b/H8 are not testable as registered (deviations.md).
Cells are classified exactly as registration 1's decision readout (gemma_readout.analyze_core)."""
from collections import defaultdict

from gemma_readout import is_forced, recall5
from gtb6 import stats as S


def holm_family(pvals, alpha=0.05):
    names = sorted(pvals)
    return dict(zip(names, S.holm([pvals[n] for n in names], alpha)))


def _task_means(deltas):
    acc = defaultdict(list)
    for t, d in deltas:
        acc[t].append(d)
    return [sum(v) / len(v) for _, v in sorted(acc.items())]


def patch_hypotheses(evals, r1, r2, clean, gold):
    """evals: G7 rows of the main patch run; r1/r2: registration-1 decision stages (C0 choices, X cells)."""
    c0_choice = {r['task']: r['choice'] for r in r1 if r['cond'] == 'C0'}
    ok = [e for e in evals if e.get('tag', 'patch_comp') == 'patch_comp' and not e.get('excluded')]
    c0_res = {e['task']: float(e['resolved']) for e in ok if e['cond'] == 'C0'}
    x_res = {(e['task'], e['intervention'], e['seed']): float(e['resolved']) for e in ok if e['cond'] == 'X'}
    h2, drop = [], []
    for r in r2:
        key, t = (r['task'], r['intervention'], r['seed']), r['task']
        if key not in x_res or t not in c0_res:
            continue
        top_c = clean[t][:5]
        d = x_res[key] - c0_res[t]
        r5_c, r5_x = recall5(top_c, gold[t]), recall5(r['shown'], gold[t])
        if r5_x < r5_c:
            drop.append((t, d))                                         # forced cells included (registered)
        valid = r['choice'] != 'INVALID' and c0_choice[t] != 'INVALID'
        if valid and set(top_c) != set(r['shown']) and not is_forced(c0_choice[t], r['shown']) and r5_x == r5_c:
            h2.append((t, d))
    v5, v6 = _task_means(h2), _task_means(drop)
    m5, lo5, hi5 = S.ci(v5, 0.90) if v5 else (None, None, None)
    if len(v5) < 20:
        verdict = 'descriptive (fewer than 20 tasks)'
    elif -0.05 <= lo5 and hi5 <= 0.05:
        verdict = 'equivalent'
    elif hi5 - lo5 > 0.10:
        verdict = 'not established (underpowered)'
    else:
        verdict = 'not equivalent'
    m6, lo6, hi6 = S.ci(v6) if v6 else (None, None, None)
    p6 = S.one_sided_p(v6, direction='less') if len(v6) >= 5 else None
    informative = len({t for t, d in h2 if d or c0_res[t]})          # tasks with any resolve, C0 or H2 cell
    return dict(H5=dict(tasks=len(v5), cells=len(h2), mean=m5, lo90=lo5, hi90=hi5, verdict=verdict,
                        informative=informative),
                H6=dict(tasks=len(v6), cells=len(drop), mean=m6, lo=lo6, hi=hi6, p_one_sided=p6,
                        descriptive=len(v6) < 5),
                C0=dict(tasks=len(c0_res), resolve_rate=sum(c0_res.values()) / len(c0_res) if c0_res else None))
