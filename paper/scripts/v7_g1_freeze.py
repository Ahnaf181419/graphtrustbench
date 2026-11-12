#!/usr/bin/env python3
"""G1 (before the second registration): freeze the flip-risk score on the registration-1 readout,
and record the exploratory exposure and fragility checks. Uses existing committed data only.

Outputs (refuses to overwrite): paper/prereg/fliprisk_frozen.json, paper/reports/v7/g1_values.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import gemma_readout as GR
from gtb6 import exposure as EX, fliprisk as FR, retrievers as R, stats as S
from gtb6.corpus import load_task
from localization_retrieval import seed_nodes


def _cells(path):
    rows = [json.loads(line) for line in open(path)]
    return {(r['instance_id'], r['corruption'], r['severity'], r['seed']): r for r in rows}


def fit_cells(readout_dir, per_task, cells_path):
    """Every valid, non-forced 40% cell of the registration-1 readout, with its features and code."""
    clean, _, gold = GR.rankings(GR.load_rows(per_task))
    r1, r2, _, _ = GR.load_stages(readout_dir)
    c0 = {r['task']: r['choice'] for r in r1 if r['cond'] == 'C0'}
    cells = _cells(cells_path)
    out = []
    for r in r2:
        t = r['task']
        if r['choice'] == 'INVALID' or c0[t] == 'INVALID' or GR.is_forced(c0[t], r['shown']):
            continue
        top_c = clean[t][:5]
        if set(r['shown']) == set(top_c):
            kind = 'order-only'
        elif GR.recall5(r['shown'], gold[t]) == GR.recall5(top_c, gold[t]):
            kind = 'H2'
        else:
            kind = 'recall-changed'
        c = cells[(t, r['intervention'], 0.4, r['seed'])]
        out.append(dict(task=t, intervention=r['intervention'], seed=r['seed'], kind=kind, forced=False,
                        flip=int(r['choice'] != c0[t]),
                        **FR.features(clean[t], list(r['shown']), c0[t], c['rbo'], c['k_eff'])))
    return out


def exposure_and_fragility(selection, config, cells_path, per_task):
    """Predicted vs observed share of k = 0 cells, and Spearman rho of per-task churn (exploratory)."""
    from scipy.stats import spearmanr
    _, _, gold = GR.rankings(GR.load_rows(per_task))
    cells = list(_cells(cells_path).values())
    churn = defaultdict(list)
    for c in cells:
        if c['corruption'] == 'spurious' and c['severity'] == 0.4:
            churn[c['instance_id']].append(float(c['member']))
    rows, feat = [], {}
    for item in selection:
        t = item['instance_id']
        candidates, lexical, edges = load_task(ROOT, item, config)
        seeds = set(seed_nodes(lexical, config['one_hop']['seed_count']))
        read = sum(1 for e in edges if e['source'] in seeds and e['target'] in lexical)
        rows += [(len(edges), read, f) for f in (0.1, 0.2, 0.4)]
        best = {}
        for n in sorted(lexical, key=lambda n: (-lexical[n], n)):
            best.setdefault(candidates[n]['path'], lexical[n])
        files = R.rank({}, lexical, candidates)
        # seed_out counts the distinct mapped seed targets (guide G1.5 sums per-seed successors, unfiltered)
        feat[t] = dict(seed_out=len({e['target'] for e in edges if e['source'] in seeds and e['target'] in lexical}),
                       bm25_margin=best[files[4]] - best[files[5]] if len(files) > 5 else 0.0,
                       n_gold=len(gold[t]), churn=sum(churn[t]) / len(churn[t]))
    tasks = sorted(feat)
    rho = {k: spearmanr([feat[t][k] for t in tasks], [feat[t]['churn'] for t in tasks]) for k in
           ('seed_out', 'bm25_margin', 'n_gold')}
    return dict(pred_k0=EX.predicted_unchanged_share(rows),
                obs_k0=sum(c['k'] == 0 for c in cells) / len(cells), cells=len(cells),
                fragility_spearman={k: dict(rho=float(v.statistic), p=float(v.pvalue)) for k, v in rho.items()},
                tasks=len(tasks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--readout', type=Path, default=ROOT / 'paper/results/2026-11-10_gemma_readout')
    ap.add_argument('--per-task', type=Path, default=ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz')
    ap.add_argument('--cells', type=Path, default=ROOT / 'paper/reports/context_stability/cells_onehop.jsonl')
    ap.add_argument('--selection', type=Path, default=ROOT / 'paper/data/heldout_full_2026-11-07.json')
    ap.add_argument('--config', type=Path, default=ROOT / 'paper/config/pilot_v3.json')
    ap.add_argument('--frozen', type=Path, default=ROOT / 'paper/prereg/fliprisk_frozen.json')
    ap.add_argument('--values', type=Path, default=ROOT / 'paper/reports/v7/g1_values.json')
    a = ap.parse_args()
    for p in (a.frozen, a.values):
        if p.exists():
            sys.exit(f'refusing to overwrite {p}')
    cells = fit_cells(a.readout, a.per_task, a.cells)
    assert len(cells) == 530, len(cells)
    frozen = FR.fit(cells)
    values = dict(fit_cells=len(cells), flip_rate=sum(c['flip'] for c in cells) / len(cells),
                  in_sample_auc=S.auc(FR.score(frozen, cells), [c['flip'] for c in cells]),
                  coef_ci=FR.coef_intervals(cells),
                  **exposure_and_fragility(json.loads(a.selection.read_text())['tasks'],
                                           json.loads(a.config.read_text()), a.cells, a.per_task))
    for p, obj in ((a.frozen, frozen), (a.values, values)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2) + '\n')
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
