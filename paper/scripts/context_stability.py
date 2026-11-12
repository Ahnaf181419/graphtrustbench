#!/usr/bin/env python3
"""Post-hoc context-stability analysis of the committed 115-task extension run (paper v5, RQ1/RQ2).

Reads only the committed per_task.jsonl.gz; the local corpus is needed only to rebuild the seeded
corrupted graphs for the delivered dose k and k_eff. No retriever is rerun or tuned.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from audit_competition_data import read_tasks
from localization_data import extract_candidates, read_snapshot
from localization_retrieval import bm25, delete_edges, edge_key, rewire_edges, seed_nodes
from render_pilot_report import bootstrap_deltas, committed_json

NAMES = {'missing': 'deletion', 'spurious': 'rewiring'}


def rbo_ext(S, T, p=0.9):
    """Extrapolated RBO (Webber, Moffat & Zobel 2010) for two rankings of equal length."""
    if len(S) != len(T):
        raise ValueError('Rankings differ in length')
    seen_s, seen_t, x, acc = set(), set(), 0, 0.0
    for d, (s, t) in enumerate(zip(S, T), start=1):
        x += 1 if s == t else (s in seen_t) + (t in seen_s)
        seen_s.add(s); seen_t.add(t)
        acc += x / d * p ** d
    k = len(S)
    return x / k * p ** k + (1 - p) / p * acc


def boot_ci(values, key, resamples=10000, seed=2026):
    """Same task-bootstrap convention as render_pilot_report.bootstrap_deltas."""
    rng = random.Random(f'{seed}:{key}')
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(values) / n, draws[int(0.025 * resamples)], draws[min(int(0.975 * resamples), resamples - 1)]


def admitted(lexical, edges, seeds, budget):
    """Per-seed visited neighbour sets, exactly as one_hop admits them (shared visit cap)."""
    adj = {s: set() for s in seeds}
    for e in edges:
        if e['source'] in adj and e['target'] in lexical:
            adj[e['source']].add(e['target'])
    visited, out = 0, []
    for s in seeds:
        take = sorted(adj[s])[:max(0, budget - visited)]
        out.append(set(take)); visited += len(take)
    return out


def corrupted_pairs(kind, edges, fraction, seed):
    """Corrupted graph, its intervention record, and the (source, original target) pairs it touched."""
    ordered = sorted(edges, key=edge_key)
    if kind == 'missing':
        kept, info = delete_edges(edges, fraction, seed)
        gone = {edge_key(e) for e in ordered} - {edge_key(e) for e in kept}
        return kept, info, {(k[0], k[1]) for k in gone}
    result, info = rewire_edges(edges, fraction, seed)
    return result, info, {(a['source'], a['target']) for a, b in zip(ordered, result) if a != b}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('.'))
    ap.add_argument('--run-dir', type=Path, default=Path('paper/results/2026-11-08_extension_v3'))
    ap.add_argument('--selection', type=Path, default=Path('paper/data/heldout_full_2026-11-07.json'))
    ap.add_argument('--heldout', type=Path, default=Path('paper/data/heldout_2026-11-08.json'))
    ap.add_argument('--out-dir', type=Path, default=Path('paper/reports/context_stability'))
    ap.add_argument('--figure', type=Path, default=Path('paper/figures/f1_dose_response.png'))
    args = ap.parse_args()
    rows = committed_json(args.run_dir / 'per_task.jsonl', lines=True)
    config = json.loads((args.run_dir / 'manifest.json').read_text())['configuration']
    selection = json.loads(args.selection.read_text())['tasks']
    heldout = {t['instance_id'] for t in json.loads(args.heldout.read_text())['tasks']}

    # --- anchors already printed in the paper; any mismatch means a wrong input ---
    gold = {r['instance_id']: set(r['gold_files']) for r in rows}
    unret = {r['instance_id']: set(r['unretrievable_gold_files']) for r in rows}
    tasks = sorted(gold)
    assert len(rows) == 4370 and len(tasks) == 115 and len(heldout) == 24 and heldout <= set(tasks)
    assert sum(map(len, gold.values())) == 248 and sum(map(len, unret.values())) == 58
    assert sum(1 for t in tasks if gold[t] <= unret[t]) == 5
    clean = {(r['instance_id'], r['method']): r for r in rows if r['severity'] == 0}
    cells = [r for r in rows if r['severity'] and r['method'] == 'one_hop']
    bm_corr = [r for r in rows if r['severity'] and r['method'] == 'bm25']
    assert all(r['ranked_files'] == clean[(r['instance_id'], 'bm25')]['ranked_files'] for r in bm_corr) and len(bm_corr) == 2070
    for r in cells:
        c = clean[(r['instance_id'], 'one_hop')]
        r['d_recall'] = r['metrics']['recall@5'] - c['metrics']['recall@5']
        r['d_success'] = r['metrics']['success@5'] - c['metrics']['success@5']
        r['full'] = r['ranked_files'] != c['ranked_files']
        r['member'] = set(r['ranked_files'][:5]) != set(c['ranked_files'][:5])
        r['rbo'] = 1 - rbo_ext(c['ranked_files'], r['ranked_files'])
        shared = [g for g in gold[r['instance_id']] if g in c['ranked_files'][:5] and g in r['ranked_files'][:5]]
        r['gm_eligible'] = bool(shared)
        r['gm_moved'] = any(c['ranked_files'][:5].index(g) != r['ranked_files'][:5].index(g) for g in shared)
    by = lambda k, s: [r for r in cells if r['corruption'] == k and r['severity'] == s]
    assert {k: sum(abs(r['d_recall']) > 1e-12 for r in cells if r['corruption'] == k) for k in NAMES} == {'missing': 40, 'spurious': 82}
    assert {k: sum(r['full'] for r in cells if r['corruption'] == k) for k in NAMES} == {'missing': 786, 'spurious': 916}
    intervals = bootstrap_deltas(rows)
    expected = {('missing', .1): -.0043, ('missing', .2): -.0058, ('missing', .4): -.0029,
                ('spurious', .1): .0041, ('spurious', .2): -.0117, ('spurious', .4): -.0309}
    for (k, s), v in expected.items():
        assert round(intervals[('one_hop', k, s)]['delta'], 4) == v, (k, s)

    def task_mean(group, field):
        per = defaultdict(list)
        for r in group:
            per[r['instance_id']].append(float(r[field]))
        return [sum(v) / len(v) for _, v in sorted(per.items())]

    vals = {'R5': f"{sum(clean[(t, 'bm25')]['metrics']['recall@5'] for t in tasks) / 115:.4f}",
            'R6': f"{sum(clean[(t, 'one_hop')]['metrics']['recall@5'] for t in tasks) / 115:.4f}",
            'R22': sum(clean[(t, 'one_hop')]['metrics']['success@5'] == 1 for t in tasks)}
    new91 = [r for r in rows if r['instance_id'] not in heldout]
    i91 = bootstrap_deltas(new91)[('one_hop', 'spurious', 0.4)]
    assert i91['tasks'] == 91
    vals['R8'] = f"{i91['delta']:+.4f} [{i91['low']:+.4f}, {i91['high']:+.4f}]"
    table1, table2 = [], []
    for k in NAMES:
        for s in (0.1, 0.2, 0.4):
            g = by(k, s)
            assert len(g) == 345
            changed = [r for r in g if abs(r['d_recall']) > 1e-12]
            table1.append(dict(intervention=NAMES[k], fraction=s, d_success=round(sum(task_mean(g, 'd_success')) / 115, 4),
                               tasks_changed=len({r['instance_id'] for r in changed}), cells_changed=len(changed)))
            m, lo, hi = boot_ci(task_mean(g, 'rbo'), ('rbo', k, s))
            table2.append(dict(intervention=NAMES[k], fraction=s, full=sum(r['full'] for r in g),
                               membership=sum(r['member'] for r in g), rbo=f"{m:.3f} [{lo:.3f}, {hi:.3f}]",
                               gold_moved=sum(r['gm_moved'] for r in g), gold_eligible=sum(r['gm_eligible'] for r in g)))
    assert [t['cells_changed'] for t in table1] == [5, 10, 25, 12, 25, 45]
    for k in NAMES:
        vals[f'T2_membership_{NAMES[k]}'] = f"{sum(r['member'] for r in cells if r['corruption'] == k):,}"

    # --- delivered dose k and k_eff, from the seeded corruptions rebuilt on the local corpus ---
    problems = {t['instance_id']: t['problem_statement'] for t in read_tasks(args.data_root / 'tasks.jsonl')}
    hop = config['one_hop']
    index = {(r['instance_id'], r['corruption'], r['severity'], r['seed']): r for r in cells}
    for item in selection:
        t = item['instance_id']
        graph = json.loads((args.data_root / item['graph_path']).read_text())
        cand, _ = extract_candidates(read_snapshot(args.data_root / item['snapshot_path']), graph['nodes'])
        lexical = bm25(problems[t], cand, **config['bm25'])
        seeds = seed_nodes(lexical, hop['seed_count'])
        reach = {(e['source'], e['target']) for e in graph['edges'] if e['source'] in seeds}
        base = admitted(lexical, graph['edges'], seeds, hop['edge_budget'])
        for k in NAMES:
            for s in (0.1, 0.2, 0.4):
                for sd in (13, 37, 101):
                    edges, info, touched = corrupted_pairs(k, graph['edges'], s, sd)
                    r = index[(t, k, s, sd)]
                    if info != r['intervention']:
                        raise ValueError(f'Recomputed intervention differs from the run: {t} {k} {s} {sd}')
                    r['k'] = len(touched & reach)
                    r['k_eff'] = sum(len(a ^ b) for a, b in zip(base, admitted(lexical, edges, seeds, hop['edge_budget'])))
        print('k/k_eff', t, flush=True)
    k0 = [r for r in cells if r['k'] == 0]
    if any(r['full'] for r in k0):
        raise AssertionError('A k = 0 cell changed its ranking')
    kpos = [r for r in cells if r['k'] > 0]
    still = [r for r in kpos if not r['full']]
    vals.update(C1=f"{len(k0):,}", R12a=f"{len(kpos):,}", R12b=f"{len(still):,}",
                R12_keff0=sum(r['k_eff'] == 0 for r in still))
    bins = [('0', 0, 0), ('1', 1, 1), ('2–3', 2, 3), ('4–7', 4, 7), ('8+', 8, 10 ** 9)]
    fig1 = {NAMES[k]: {b: sorted(r['rbo'] for r in cells if r['corruption'] == k and lo <= r['k_eff'] <= hi)
                       for b, lo, hi in bins} for k in NAMES}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'values.json').write_text(json.dumps(vals, indent=2) + '\n')
    (args.out_dir / 'table1_extra.json').write_text(json.dumps(table1, indent=2) + '\n')
    (args.out_dir / 'table2.json').write_text(json.dumps(table2, indent=2) + '\n')
    with (args.out_dir / 'cells_onehop.jsonl').open('w') as fh:
        for r in cells:
            fh.write(json.dumps({x: r[x] for x in ('instance_id', 'corruption', 'severity', 'seed', 'd_recall', 'd_success',
                                                    'full', 'member', 'rbo', 'gm_eligible', 'gm_moved', 'k', 'k_eff')}) + '\n')

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (iv, data) in zip(axes, fig1.items()):
        ax.boxplot([v or [float('nan')] for v in data.values()], showfliers=False)
        ax.set_xticks(range(1, len(bins) + 1), [f'{b}\n(n={len(v)})' for b, v in data.items()])
        ax.set_title(iv); ax.set_xlabel('k_eff (visited neighbours changed)')
    axes[0].set_ylabel('1 − RBO vs clean')
    plt.tight_layout(); plt.savefig(args.figure, dpi=200)
    summary = {iv: {b: (len(v), v[len(v) // 2] if v else None) for b, v in d.items()} for iv, d in fig1.items()}
    (args.out_dir / 'figure1_bins.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(vals, indent=2)); print(json.dumps(table1, indent=2)); print(json.dumps(table2, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
