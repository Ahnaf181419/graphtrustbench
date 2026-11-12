"""G7: evaluate every generated patch of the G6.2 patch readout in Docker (§5.2; H5, H6).

Per task: the competition snapshot (base tree = snapshot HEAD, as G4), the G4 environment (dependencies as of
created_at), then gtb6.repo_ops.evaluate for each generated reply: last diff block, test-file edits dropped,
git apply / patch --fuzz=3, the reference test part on top, F2P and P2P tests; resolved only if all pass.
F2P/P2P come from G4, with the re-derived sets (2026-11-11_g4_rederive) in place of the first ones.
Excluded tasks (install failed, no F2P test, a crashed derivation) are recorded with their reason for every
reply; no exclusion depends on model output. Resumable: one work file per task.

    python3 paper/scripts/v7_g7_evaluate.py --out paper/results/<RUNDATE>_g7_eval \
        --raw paper/results/<RUNDATE>_g6_patch/raw_patch_comp.jsonl ...
"""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import v7_g4_trace as G4                                   # noqa: E402
from gtb6 import repo_ops                                  # noqa: E402

KEEP = ('tag', 'id', 'cond', 'task', 'intervention', 'seed', 'rep', 'filler')


def load_tests(first, again=None):
    tests = json.loads(Path(first).read_text())
    if again and Path(again).exists():
        tests.update(json.loads(Path(again).read_text()))
    return tests


def exclusion(rec):
    """Why a task cannot be evaluated, or None. G4's trace-only reasons do not exclude it here."""
    if rec.get('install') != 'ok':
        return 'install failed'
    t = rec.get('tests') or {}
    if t.get('error'):
        return t['error']
    if not t.get('f2p'):
        return 'no F2P test'
    return None


def evaluate_rows(repo, base, rows, test_patch, f2p, p2p, run):
    """repo_ops.evaluate for each reply; replies with the same extracted diff are evaluated once."""
    seen, out = {}, []
    for r in rows:
        diff = repo_ops.extract_diff(r['raw'] or '')
        if diff is None:
            res = dict(has_diff=False, applied=None, resolved=False, failed_tests=[])
        elif diff in seen:
            res = dict(seen[diff], reused=True)
        else:
            res = repo_ops.evaluate(repo, base, r['raw'], test_patch, f2p, p2p, run)
            seen[diff] = res
        out.append({**{k: r[k] for k in KEEP if k in r}, **res})
    return out


def process(task, item, rec, rows, scratch, cache, out_file):
    t, t0 = task['instance_id'], time.perf_counter()
    why = exclusion(rec)
    if why:
        res = [{**{k: r[k] for k in KEEP if k in r}, 'excluded': why, 'resolved': False} for r in rows]
    else:
        repo = Path(scratch) / f'{t}_g7'
        shutil.rmtree(repo, ignore_errors=True)
        repo.mkdir(parents=True)
        try:
            with tarfile.open(ROOT / item['snapshot_path']) as tf:
                tf.extractall(repo, filter='data')
            base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
            docker = G4.docker_cmd(G4.IMAGE)
            ok, err = G4.prepare(repo, base, task['repo'], task['created_at'], docker, cache)
            if not ok:
                res = [{**{k: r[k] for k in KEEP if k in r}, 'excluded': 'install failed', 'resolved': False}
                       for r in rows]
            else:
                _, test, _ = repo_ops.split_reference(task['patch'] + '\n' + task['test_patch'])
                res = evaluate_rows(repo, base, rows, test, rec['tests']['f2p'], rec['tests']['p2p'],
                                    lambda r, tg: G4._outcomes(docker, r, tg))
        finally:
            shutil.rmtree(repo, ignore_errors=True)
    Path(out_file).write_text(json.dumps(dict(task=t, seconds=round(time.perf_counter() - t0, 1), rows=res)))
    return t, Counter('excluded' if x.get('excluded') else x['resolved'] for x in res)


def _job(args):
    try:
        return process(*args)
    except Exception as e:                                # recorded; the driver refuses to summarise errors
        Path(args[-1]).write_text(json.dumps(dict(task=args[0]['instance_id'], error=repr(e)[-500:], rows=[])))
        return args[0]['instance_id'], Counter(error=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--raw', type=Path, nargs='+', required=True)
    ap.add_argument('--tests', type=Path, default=ROOT / 'paper/results/2026-11-11_g4_trace/tests_competition.json')
    ap.add_argument('--rederived', type=Path, default=ROOT / 'paper/results/2026-11-11_g4_rederive/tests_competition.json')
    ap.add_argument('--selection', type=Path, default=ROOT / 'paper/data/heldout_full_2026-11-07.json')
    ap.add_argument('--scratch', type=Path, default=Path.home() / '.cache/gtb6')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    if not a.rederived.exists():
        sys.exit(f'{a.rederived} missing: run the G4 re-derivation first (deviations.md)')
    tests = load_tests(a.tests, a.rederived)
    items = {t['instance_id']: t for t in json.loads(a.selection.read_text())['tasks']}
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'tasks.jsonl')}
    rows = defaultdict(list)
    for f in a.raw:
        tag = f.stem.removeprefix('raw_')
        for line in open(f):
            r = json.loads(line)
            rows[r['task']].append(dict(r, tag=tag))
    work = a.out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    (a.out / '.gitignore').write_text('work/\n')
    jobs = [(tasks[t], items[t], tests[t], rs, a.scratch / 'repos', a.scratch / 'uvcache', work / f'{t}.json')
            for t, rs in sorted(rows.items()) if not (work / f'{t}.json').exists()
            or 'error' in json.loads((work / f'{t}.json').read_text())]
    print(f'{len(rows) - len(jobs)} tasks done earlier, {len(jobs)} to run, {sum(map(len, rows.values()))} replies',
          flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(_job, j) for j in jobs]), 1):
            t, c = f.result()
            print(f'[{i}/{len(jobs)}] {t} {dict(c)}', flush=True)
    recs = [json.loads((work / f'{t}.json').read_text()) for t in sorted(rows)]
    errors = [r for r in recs if 'error' in r]
    if errors:
        sys.exit(f'{len(errors)} tasks raised; rerun to retry them: ' + '; '.join(f"{r['task']}: {r['error']}" for r in errors[:5]))
    with open(a.out / 'evaluations.jsonl', 'w') as fh:
        for r in recs:
            for x in r['rows']:
                fh.write(json.dumps(x) + '\n')
    allrows = [x for r in recs for x in r['rows']]
    summary = dict(replies=len(allrows), tasks=len(recs),
                   excluded_tasks=dict(Counter(exclusion(tests[r['task']]) for r in recs if exclusion(tests[r['task']]))),
                   by_cond={c: dict(n=sum(x['cond'] == c and not x.get('excluded') for x in allrows),
                                    resolved=sum(x['cond'] == c and x['resolved'] for x in allrows))
                            for c in sorted({x['cond'] for x in allrows})},
                   crashed=sum(bool(x.get('crashed')) for x in allrows))
    (a.out / 'summary_g7.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
