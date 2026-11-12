#!/usr/bin/env python3
"""G3 (second registration, exploratory): PPR, the dropout-ensemble mitigation and low corruption rates.

Every graph condition is rebuilt with the registered corruption (context_stability.corrupted_pairs) and
checked against the committed run: each recorded intervention must match, and the rebuilt one-hop
rankings must equal all 2,185 committed one-hop rankings, before any number is written.
Resumable: one work file per task in <out>/work/ (not committed).
"""
import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import gemma_readout as GR
from context_stability import corrupted_pairs, rbo_ext
from gtb6 import retrievers as R, stats as S
from localization_retrieval import delete_edges, one_hop

FRACS, SEEDS, IVS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4), (13, 37, 101), ('missing', 'spurious')
CLEAN = ('clean', 0.0, 0)
SCORERS = ('one_hop', 'ppr', 'mit')
TAGS = {'missing': 'DEL', 'spurious': 'REW'}


def build_conditions(edges, recorded=None, fracs=FRACS):
    """{CLEAN or (iv, frac, seed): edges}; a recorded intervention that differs from the rebuild raises."""
    conds = {CLEAN: delete_edges(edges, 0.0, 0)[0]}
    for iv in IVS:
        for f in fracs:
            for sd in SEEDS:
                corrupted, info, _ = corrupted_pairs(iv, edges, f, sd)
                want = (recorded or {}).get((iv, f, sd))
                if want is not None and want != info:
                    raise ValueError(f'rebuilt intervention differs from the run: {iv} {f} {sd}')
                conds[(iv, f, sd)] = corrupted
    return conds


def rank_all(conditions, lexical, candidates, scorer):
    """Registered file ranking (gtb6.retrievers.rank) of scorer(lexical, edges) for every condition."""
    return {key: R.rank(scorer(lexical, edges), lexical, candidates) for key, edges in conditions.items()}


def cell_table(task, rankings, gold):
    """One row per corrupted condition, against the same scorer's clean ranking."""
    c = rankings[CLEAN]
    return [dict(task=task, intervention=iv, fraction=f, seed=sd,
                 d_recall=GR.recall5(r, gold) - GR.recall5(c, gold),
                 set_changed=set(r[:5]) != set(c[:5]), one_minus_rbo=1 - rbo_ext(c, r))
            for (iv, f, sd), r in rankings.items() if (iv, f, sd) != CLEAN]


def committed(pred_path):
    """Committed one-hop rankings and recorded interventions, keyed by task then condition."""
    want, info = defaultdict(dict), defaultdict(dict)
    for r in GR.load_rows(pred_path):
        if r['method'] == 'one_hop':
            key = (r['corruption'], r['severity'], r['seed'])
            want[r['instance_id']][key] = r['ranked_files']
            if key != CLEAN:
                info[r['instance_id']][key] = r['intervention']
    return want, info


def run_task(item, config, gold, want, info):
    from gtb6.corpus import load_task
    t = item['instance_id']
    candidates, lexical, edges = load_task(ROOT, item, config)
    conds = build_conditions(edges, recorded=info[t])
    hop = config['one_hop']
    oh = lambda l, e: one_hop(l, e, **hop)[0]
    scorers = dict(one_hop=oh, ppr=lambda l, e: R.ppr(l, e, seed_count=hop['seed_count'], restart=0.15),
                   mit=lambda l, e: R.dropout_ensemble(l, e, t, oh, m=10, p=0.2))
    out = dict(task=t, seconds={}, cells={}, rankings_040={})
    for name in SCORERS:
        t0 = time.perf_counter()
        rk = rank_all(conds, lexical, candidates, scorers[name])
        out['seconds'][name] = time.perf_counter() - t0
        if name == 'one_hop':
            bad = [k for k, v in want[t].items() if rk[k] != v]
            if bad or len(want[t]) != 19:
                raise AssertionError(f'{t}: rebuilt one-hop rankings differ from the committed run at {bad}')
        out['cells'][name] = cell_table(t, rk, gold[t])
        out['rankings_040'][name] = [[list(k), v] for k, v in rk.items() if k[1] in (0.0, 0.4)]
    out['clean_recall'] = {n: GR.recall5(dict((tuple(k), v) for k, v in out['rankings_040'][n])[CLEAN], gold[t])
                           for n in SCORERS}
    return out


def task_ci(cells, field):
    per = defaultdict(list)
    for c in cells:
        per[c['task']].append(float(c[field]))
    return S.ci([sum(v) / len(v) for _, v in sorted(per.items())])


def fill(results):
    cells = {n: [c for r in results for c in r['cells'][n]] for n in SCORERS}
    at = lambda n, iv, f: [c for c in cells[n] if c['intervention'] == iv and c['fraction'] == f]
    v = {}
    for iv, tag in TAGS.items():
        g = at('ppr', iv, 0.4)
        v[f'PPR_R_{tag}'] = S.fmt_ci(*task_ci(g, 'd_recall'))
        v[f'PPR_M_{tag}' + ('_N' if tag == 'REW' else '')] = str(sum(c['set_changed'] for c in g))
        m, lo, hi = task_ci(g, 'one_minus_rbo')
        v[f'PPR_RBO_{tag}'] = f'{m:.3f} [{lo:.3f}, {hi:.3f}]'
        if tag == 'REW':
            v['PPR_M_REW'] = str(round(100 * sum(c['set_changed'] for c in g) / len(g)))
    g = at('mit', 'spurious', 0.4)
    v['MIT_M_REW_N'] = str(sum(c['set_changed'] for c in g))
    v['MIT_M_REW'] = str(round(100 * sum(c['set_changed'] for c in g) / len(g)))
    secs = {n: sum(r['seconds'][n] for r in results) for n in SCORERS}
    v['MIT_COST'] = f"{secs['mit'] / secs['one_hop']:.1f}"
    d = sum(r['clean_recall']['mit'] - r['clean_recall']['one_hop'] for r in results) / len(results)
    v['MIT_RECALL'] = 'unchanged' if abs(d) < 0.0005 else f"{'down' if d < 0 else 'up'} {abs(d):.3f}"
    v['OH_M_DEL_N'] = sum(c['set_changed'] for c in at('one_hop', 'missing', 0.4))
    v['OH_M_REW_N'] = sum(c['set_changed'] for c in at('one_hop', 'spurious', 0.4))
    v['seconds'] = {n: round(s, 1) for n, s in secs.items()}
    v['low_rate'] = {f'{iv}_{f}': dict(membership=sum(c['set_changed'] for c in at('one_hop', iv, f)),
                                       recall_cells=sum(abs(c['d_recall']) > 1e-12 for c in at('one_hop', iv, f)),
                                       cells=len(at('one_hop', iv, f)))
                     for iv in IVS for f in FRACS}
    v['tasks'] = len(results)
    return v, cells


def write_csv(path, rows):
    keys = list(rows[0])
    with open(path, 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(str(r[k]) for k in keys) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='paper/results/<RUNDATE>_g3_retrievers')
    ap.add_argument('--tasks', type=int, default=None, help='smoke subset: first N selected tasks')
    ap.add_argument('--selection', type=Path, default=ROOT / 'paper/data/heldout_full_2026-11-07.json')
    ap.add_argument('--config', type=Path, default=ROOT / 'paper/config/pilot_v3.json')
    ap.add_argument('--per-task', type=Path, default=ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz')
    ap.add_argument('--pred', type=Path, default=ROOT / 'paper/results/2026-11-08_extension_v3/predictions.jsonl.gz')
    a = ap.parse_args()
    selection = json.loads(a.selection.read_text())['tasks'][:a.tasks]
    config = json.loads(a.config.read_text())
    gold = GR.rankings(GR.load_rows(a.per_task))[2]
    want, info = committed(a.pred)
    work = a.out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    (a.out / '.gitignore').write_text('work/\n')
    results = []
    for i, item in enumerate(selection, 1):
        p = work / f"{item['instance_id']}.json.gz"
        if not p.exists():
            r = run_task(item, config, gold, want, info)
            with gzip.open(p, 'wt') as fh:
                json.dump(r, fh)
            print(f"[{i}/{len(selection)}] {item['instance_id']} "
                  + ' '.join(f'{n}={s:.1f}s' for n, s in r['seconds'].items()), flush=True)
        with gzip.open(p, 'rt') as fh:
            results.append(json.load(fh))
    print('one-hop rankings reproduced for', len(results), 'tasks (19 committed conditions each)')
    v, cells = fill(results)
    if a.tasks is None:
        assert (v['OH_M_REW_N'], v['OH_M_DEL_N']) == (312, 219), (v['OH_M_REW_N'], v['OH_M_DEL_N'])
    names = dict(one_hop='onehop_all', ppr='ppr', mit='mit')
    for n in SCORERS:
        write_csv(a.out / f'cells_{names[n]}.csv', cells[n])
        with gzip.open(a.out / f'rankings_{n}_040.json.gz', 'wt') as fh:
            json.dump([dict(task=r['task'], key=k, ranking=rk) for r in results for k, rk in r['rankings_040'][n]], fh)
    (a.out / 'fill_g3.json').write_text(json.dumps(v, indent=2) + '\n')
    print(json.dumps(v, indent=2))


if __name__ == '__main__':
    main()
