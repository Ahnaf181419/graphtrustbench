"""G8: every v7 number from committed outputs -> paper/reports/v7/fill_values_v7.json.

    python3 paper/scripts/v7_g8_fill.py --g7 paper/results/<d>_g7_eval --settings paper/results/<d>_g6_settings ...
Registered rules: H5 by its 90% interval; Holm over {H6, H7} (H4a/H4b/H8 not testable, deviations.md);
settings coded against their own C0 (G8.5). A setting whose raw files are absent is 'not run'.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import gemma_readout as g
import v7_g8_patch as P
import v7_g8_settings as ST
from gtb6 import stats as S

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / 'paper/results/2026-11-10_gemma_readout'
EXT = ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'
SET_ROWS = {'T_ON': ['set_think'], 'M26': ['set_m26'], 'E4': ['set_e4'], 'B8': ['set_b8000'],
            'P': ['set_para1', 'set_para2', 'set_para3'], 'S': [f'set_sampled_{s}' for s in range(5)]}


def _paired(evals, cond, c0_res, keys=None):
    """Per-task mean of resolved(cond cell) - resolved(C0)."""
    acc = defaultdict(list)
    for e in evals:
        if e['cond'] == cond and not e.get('excluded') and e['task'] in c0_res and e.get('tag', 'patch_comp') == 'patch_comp':
            if keys is None or (e['task'], e.get('intervention'), e.get('seed')) in keys:
                acc[e['task']].append(float(e['resolved']) - c0_res[e['task']])
    return [sum(v) / len(v) for _, v in sorted(acc.items())]


def patch_values(evals, r1, r2, clean, gold, h7_p, rep_tasks=()):
    out = P.patch_hypotheses(evals, r1, r2, clean, gold)
    h5, h6 = out['H5'], out['H6']
    ok = [e for e in evals if not e.get('excluded') and e.get('tag', 'patch_comp') == 'patch_comp']
    c0 = {e['task']: float(e['resolved']) for e in ok if e['cond'] == 'C0'}
    c0r = defaultdict(list)
    for e in evals:
        if e['cond'] == 'C0r' and not e.get('excluded') and e['task'] in c0:
            c0r[e['task']].append(float(e['resolved']))
    pl = _paired(evals, 'P', c0)
    rep = _paired(evals, 'REP', c0)
    rep_all = rep + [0.0] * len([t for t in rep_tasks if t in c0 and not any(
        e['cond'] == 'REP' and e['task'] == t for e in ok)])          # same prompt as C0 when the top five is unchanged
    holm = P.holm_family({'H6': h6['p_one_sided'] if h6['p_one_sided'] is not None else 1.0, 'H7': h7_p})
    pts = lambda v: S.fmt_pts(*S.ci(v)) if v else 'n/a'
    v = dict(RES_CLEAN=f"{100 * out['C0']['resolve_rate']:.1f}", RES_TASKS=out['C0']['tasks'],
             H5_DIFF=S.fmt_pts(h5['mean']), H5_LO90=S.fmt_pts(h5['lo90']), H5_HI90=S.fmt_pts(h5['hi90']),
             H5_LO=S.fmt_pts(h5['lo90']), H5_HI=S.fmt_pts(h5['hi90']), H5_VERDICT=h5['verdict'],
             H5_TASKS=h5['tasks'], H5_CELLS=h5['cells'], H5_INFORMATIVE=h5['informative'],
             APPLY_CLEAN=f"{100 * sum(bool(e.get('applied')) for e in ok if e['cond'] == 'C0') / len(c0):.1f}",
             H6_DIFF=S.fmt_pts(h6['mean'], h6['lo'], h6['hi']), H6_p=h6['p_one_sided'], H6_TASKS=h6['tasks'],
             H6_CELLS=h6['cells'], H6_VERDICT='holds' if holm['H6'] else 'not established',
             H7_VERDICT='holds' if holm['H7'] else 'not established', HOLM=holm,
             PL_RES_DIFF=pts(pl), REP_RES_DIFF=pts(rep_all), REP_RES_DIFF_CHANGED=pts(rep), REP_TASKS_CHANGED=len(rep),
             PATCH_FLOOR_AGREE=(f"{100 * sum(x == c0[t] for t, xs in c0r.items() for x in xs) / sum(map(len, c0r.values())):.1f}"
                                if c0r else 'not run'))
    return v


def settings_values(dirs):
    raw = {}
    for d in dirs:
        for f in Path(d).glob('raw_set_*.jsonl'):
            raw[f.stem.removeprefix('raw_')] = g.load_rows(f)
    v = {}
    for k, tags in SET_ROWS.items():
        if not all(t in raw for t in tags):
            v[f'{k}_H2'] = v[f'{k}_PL'] = v[f'{k}_D'] = 'not run'
            continue
        runs = [ST.modal_rows([raw[t] for t in tags])] if k == 'S' else [raw[t] for t in tags]
        h2, pl, d = ST.setting_row(runs)
        v.update({f'{k}_H2': f'{100 * h2:.1f}%', f'{k}_PL': f'{100 * pl:.1f}%', f'{k}_D': S.fmt_pts(*d)})
    return v


def cutoff_split(rows, stages, tasks_meta, cutoff='2025-03-01'):
    """Registered exploratory split (prereg v2 §1): H2 and H3 of registration 1, by task created_at."""
    clean, corr, gold = g.rankings(rows)
    ret = {r['instance_id']: set(r['gold_files']) - set(r['unretrievable_gold_files'])
           for r in rows if r['method'] == 'one_hop' and r['severity'] == 0}
    v = {}
    for name, keep in (('PRE', [t for t in clean if tasks_meta[t]['created_at'][:10] < cutoff]),
                       ('POST', [t for t in clean if tasks_meta[t]['created_at'][:10] >= cutoff])):
        sub = [[r for r in st if r['task'] in keep] for st in stages]
        o = g.analyze_core(*sub, {t: clean[t] for t in keep}, {k: x for k, x in corr.items() if k[0] in keep}, gold, ret)
        v.update({f'SPLIT_{name}_N': len(keep), f'SPLIT_{name}_H2': S.fmt_pts(*o['H2']['ci']),
                  f'SPLIT_{name}_H3': S.fmt_pts(*o['H3']['ci']), f'SPLIT_{name}_H3_TASKS': o['H3']['tasks'],
                  f'SPLIT_{name}_H3_READING': o['H3']['verdict']})
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--g7', type=Path, required=True)
    ap.add_argument('--settings', type=Path, nargs='*', default=[])
    ap.add_argument('--out', type=Path, default=ROOT / 'paper/reports/v7/fill_values_v7.json')
    a = ap.parse_args()
    clean, _, gold = g.rankings(g.load_rows(EXT))
    r1, r2, _, _ = g.load_stages(V5)
    evals = g.load_rows(a.g7 / 'evaluations.jsonl')
    g3 = json.loads((ROOT / 'paper/results/2026-11-11_g3_retrievers/fill_g3.json').read_text())
    g4 = json.loads((ROOT / 'paper/results/2026-11-11_g4_trace/fill_g4.json').read_text())
    import gzip
    rep_tasks = json.load(gzip.open(ROOT / 'paper/results/2026-11-11_g4_trace/repaired_rankings.json.gz', 'rt'))
    v = {k: x for k, x in {**g3, **g4}.items() if isinstance(x, (str, int, float))}
    v.update(patch_values(evals, r1, r2, clean, gold, g4['H7_p'], rep_tasks))
    v.update(settings_values(a.settings))
    meta = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'tasks.jsonl')}
    v.update(cutoff_split(g.load_rows(EXT), g.load_stages(V5), meta))
    v.update(H4_H2_VERDICT='not testable as registered', H4_H3_VERDICT='not testable as registered',
             H8_VERDICT='not testable as registered', N_NEW='0')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(v, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
    print(json.dumps(v, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
