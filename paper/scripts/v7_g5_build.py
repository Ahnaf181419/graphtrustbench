#!/usr/bin/env python3
"""G5 (second registration §2): build, select and freeze the post-cutoff tasks.

Phase A (network, gh api): candidate PRs per repository in merge-date windows, then PR/issue details,
cached in <out>/details_cache.json. Phase B: per candidate, in merge order: code + test change, merge
base, Docker environment as of the merge date, F2P/P2P tests (Docker), PyCG graph (a static
analyser, own Python 3.10 venv), definitions, BM25. Phase C: 40% corrupted graphs and one-hop
rankings (H8 features; no model output). Phase D: the manifest paper/prereg/postcutoff_tasks.json,
written once. Resumable: one result per candidate in <out>/work/.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import v7_g4_trace as G4
from context_stability import admitted, corrupted_pairs, rbo_ext
from gtb6 import builder as B, repo_ops, retrievers as R
from localization_retrieval import edge_key, one_hop, seed_nodes

CLEAN = ('clean', 0.0, 0)
IVS, SEEDS = ('missing', 'spurious'), (13, 37, 101)
WINDOWS = [(B.CUTOFF, '2025-08-31'), ('2025-09-01', '2026-02-28'), ('2026-03-01', None)]
# 0.0.8's wheel ships 'PyCG/' but imports 'pycg' (fails on case-sensitive filesystems); 0.0.7 needs pkg_resources
PYCG = ['pycg==0.0.7', 'setuptools==75.8.0']
PYCG_PATCH = ROOT / 'paper/pycg/pycg-0.0.7-posonly-defaults.patch'    # crash on positional-only defaults (requests)
PYCG_MAX_ITER = 2      # rich never reaches PyCG's fixed point; iteration 3 explodes (paper/pycg/STUDY.md)
MANIFEST_KEYS = ('task_id', 'repo', 'pr', 'issue', 'merged_at', 'issue_created', 'base_sha', 'gold_files', 'f2p', 'p2p')
STRICT, FALLBACK = 'issue and fix after the cutoff', 'fix after the cutoff (fallback)'


# ---------------------------------------------------------------- phase B: one candidate
def build_task(d, repo, base, run, callgraph, setup, pkg_name):
    """(task dict, 'ok') or (None, reason), in the registered order of checks."""
    code, test, gold = repo_ops.split_reference(d['patch'])
    if not gold or not test.strip():
        return None, 'needs a code and a test change'
    ok, _ = setup(repo, base)
    if not ok:
        return None, 'install failed'
    tests = repo_ops.derive_tests(repo, base, code, test, lambda r, tg: G4._outcomes(run, r, tg))
    if tests.get('error'):
        return None, tests['error']
    if not tests['f2p']:
        return None, 'no fail-to-pass test'
    repo_ops.reset(repo, base)
    pkg = [p for p in B.package_dirs(repo) if os.path.basename(p) == pkg_name]
    if not pkg:
        return None, 'package not found'
    cg = callgraph(pkg[0])
    if cg is None:
        return None, 'PyCG failed'
    defs = B.extract_definitions(repo, pkg[0])
    ids = [x['node_id'] for x in defs]
    files = {p: (Path(repo) / p).read_text(encoding='utf-8', errors='replace') for p in sorted({x['path'] for x in defs})}
    short = G4.PKG[d['repo']]
    return dict(task_id=f"{short}-{d['pr']}", repo=short, pr=d['pr'], issue=d['issue'], merged_at=d['merged_at'],
                issue_created=d['issue_created'], base_sha=base, problem_statement=d['problem_statement'],
                patch=d['patch'], gold_files=gold, f2p=tests['f2p'], p2p=tests['p2p'], succ=B.to_succ(cg, ids),
                bm25=B.candidate_bm25(d['problem_statement'], defs), node2file={x['node_id']: x['path'] for x in defs},
                defs={x['node_id']: dict(qualname=x['qualname'], source=x['source']) for x in defs},
                files=files), 'ok'


def select(details, outcome):
    """Registered selection: the strict pool (issue opened on/after the cutoff) in merge order until
    MAX_TASKS pass; if fewer than MIN_TASKS pass, every candidate in merge order instead."""
    memo = {}

    def build_all(pool):
        built, why = [], Counter()
        for d in pool:
            key = (d['repo'], d['pr'])
            if key not in memo:
                memo[key] = outcome(d)
            task, reason = memo[key]
            why[reason] += 1
            if task:
                built.append(task)
            if len(built) == B.MAX_TASKS:
                break
        return built, dict(why)

    by_merge = sorted(details, key=lambda d: d['merged_at'])
    tasks, why = build_all([d for d in by_merge if d['issue_created'][:10] >= B.CUTOFF])
    reasons = {'strict': why}
    if len(tasks) >= B.MIN_TASKS:
        return tasks, reasons, STRICT
    tasks, reasons['fallback'] = build_all(by_merge)
    return tasks, reasons, FALLBACK


def write_manifest(path, rule, reasons, tasks):
    path = Path(path)
    if path.exists():
        sys.exit(f'refusing to overwrite {path}: the post-cutoff tasks are frozen')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(rule=rule, reasons=reasons, tasks=[{k: t[k] for k in MANIFEST_KEYS} for t in tasks]),
                               indent=1) + '\n')


# ---------------------------------------------------------------- phase C
def succ_edges(succ):
    return sorted((dict(source=u, target=v, type='calls', key=0) for u, vs in succ.items() for v in vs), key=edge_key)


def cells_for(task, hop):
    """Clean and 40% corrupted one-hop rankings, and per-cell k_eff and 1 - RBO (H8 features)."""
    lexical, edges = task['bm25'], succ_edges(task['succ'])
    candidates = {n: dict(path=p) for n, p in task['node2file'].items()}
    seeds = seed_nodes(lexical, hop['seed_count'])
    base = admitted(lexical, edges, seeds, hop['edge_budget'])
    rank = lambda e: R.rank(one_hop(lexical, e, **hop)[0], lexical, candidates)
    rankings, cells = {CLEAN: rank(edges)}, []
    for iv in IVS:
        for sd in SEEDS:
            corrupted, _, _ = corrupted_pairs(iv, edges, 0.4, sd)
            r = rankings[(iv, 0.4, sd)] = rank(corrupted)
            cells.append(dict(task=task['task_id'], intervention=iv, fraction=0.4, seed=sd,
                              k_eff=sum(len(a ^ b) for a, b in zip(base, admitted(lexical, corrupted, seeds, hop['edge_budget']))),
                              one_minus_rbo=1 - rbo_ext(rankings[CLEAN], r)))
    return rankings, cells


# ---------------------------------------------------------------- driver
def fetch(cache_path):
    """Phase A, resumable: candidates per window, then details per candidate (None = not a task)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else dict(candidates=None, details={})
    if cache['candidates'] is None:
        cands, seen = [], set()
        for repo in B.REPOS:
            for lo, hi in WINDOWS:
                q = f'repo:{repo} is:pr is:merged ' + (f'merged:{lo}..{hi}' if hi else f'merged:>={lo}')
                total = B.gh('search/issues', q=q, per_page=1)['total_count']
                if total >= 1000:
                    sys.exit(f'{q}: {total} results, over the search API limit; narrow the window')
                time.sleep(3)
                for c in B.candidate_prs(repo, cutoff=lo, until=hi):
                    if (c['repo'], c['pr']) not in seen:
                        seen.add((c['repo'], c['pr'])); cands.append(c)
                time.sleep(3)
        cache['candidates'] = cands
        cache_path.write_text(json.dumps(cache))
    for i, c in enumerate(cache['candidates']):
        key = f"{c['repo']}#{c['pr']}"
        d = cache['details'].get(key, 'missing')
        if d != 'missing' and not (d and 'HTTP 5' in d.get('error', '')):     # retry server errors
            continue
        try:
            cache['details'][key] = B.pr_details(c['repo'], c['pr'], c['issue'])
        except subprocess.CalledProcessError as e:              # e.g. the linked issue does not exist
            cache['details'][key] = dict(error=(e.stderr or '')[-200:])
        time.sleep(0.8)
        if i % 20 == 0:
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    return [d for d in cache['details'].values() if d and 'error' not in d], cache


def _build_job(args):
    d, clone, work_root, pycg_py, uvcache, out = args
    t0 = time.perf_counter()
    tid = f"{G4.PKG[d['repo']]}-{d['pr']}"
    repo = Path(work_root) / tid
    shutil.rmtree(repo, ignore_errors=True)
    try:
        subprocess.run(['git', 'clone', '-q', '--shared', '--no-checkout', str(clone), str(repo)], check=True,
                       capture_output=True)
        run = G4.docker_cmd(G4.IMAGE)
        setup = lambda r, sha: G4.prepare(r, sha, d['repo'], d['merged_at'], run, uvcache)
        cg = lambda pkg: B.run_pycg(pkg, str(Path(work_root) / f'{tid}.pycg.json'), python=pycg_py, max_iter=PYCG_MAX_ITER)
        task, why = build_task(d, repo, d['base_sha'], run, cg, setup, G4.PKG[d['repo']])
    except Exception as e:                                          # logged as a reason, never dropped
        task, why = None, f'error: {type(e).__name__}: {e}'[:300]
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    with gzip.open(out, 'wt') as fh:
        json.dump(dict(task=task, reason=why, seconds=round(time.perf_counter() - t0, 1)), fh)
    return tid, why


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='paper/results/<RUNDATE>_g5_build')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'paper/prereg/postcutoff_tasks.json')
    ap.add_argument('--scratch', type=Path, default=Path.home() / '.cache/gtb6')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--phase', choices=('fetch', 'all'), default='all')
    ap.add_argument('--config', type=Path, default=ROOT / 'paper/config/pilot_v3.json')
    a = ap.parse_args()
    if a.manifest.exists():
        sys.exit(f'refusing to overwrite {a.manifest}: the post-cutoff tasks are frozen')
    work = a.out / 'work'
    for p in (work, a.scratch / 'clones', a.scratch / 'g5', a.scratch / 'uvcache'):
        p.mkdir(parents=True, exist_ok=True)
    (a.out / '.gitignore').write_text('work/\npostcutoff_full.json.gz\n')
    details, cache = fetch(a.out / 'details_cache.json')
    print(f"{len(cache['candidates'])} PRs that close an issue; {len(details)} with a real issue; "
          f"{sum(d['issue_created'][:10] >= B.CUTOFF for d in details)} issue opened after the cutoff", flush=True)
    if a.phase == 'fetch':
        return
    pycg = a.scratch / 'pycg'
    if not (pycg / 'bin/python').exists():
        subprocess.run(['uv', 'venv', '-q', '--python', '3.10', str(pycg)], check=True)
    # a fresh copy (not hardlinked into uv's cache), then the repository's patch
    subprocess.run(['uv', 'pip', 'install', '-q', '--reinstall', '--link-mode', 'copy', '--python',
                    str(pycg / 'bin/python'), *PYCG], check=True)
    site = next((pycg / 'lib').glob('python3.10/site-packages'))
    subprocess.run(['patch', '-s', '-p1', '--forward', '-d', str(site), '-i', str(PYCG_PATCH)], check=True)
    subprocess.run([str(pycg / 'bin/python'), '-c', 'import pycg.pycg'], check=True)   # 'PyCG failed' must mean the package
    (a.out / 'pycg_lock.txt').write_text(subprocess.check_output(
        ['uv', 'pip', 'freeze', '--python', str(pycg / 'bin/python')], text=True))
    clones = {}
    for repo in B.REPOS:
        c = clones[repo] = a.scratch / 'clones' / G4.PKG[repo]
        if not c.exists():
            subprocess.run(['git', 'clone', '-q', f'https://github.com/{repo}', str(c)], check=True)
    result = lambda d: work / f"{G4.PKG[d['repo']]}-{d['pr']}.json.gz"

    def prebuild(pool):
        """Build the pool in merge order, in parallel batches, until MAX_TASKS of it passed."""
        i = 0
        while i < len(pool):
            done = [json.load(gzip.open(result(d), 'rt')) for d in pool[:i] if result(d).exists()]
            if sum(r['task'] is not None for r in done) >= B.MAX_TASKS:
                return
            batch = [d for d in pool[i:i + 2 * a.workers] if not result(d).exists()]
            for d in batch:                                         # serial: git fetch of the PR head
                d['base_sha'] = B.merge_base(str(clones[d['repo']]), d['base_sha'], d['pr'])
            with ProcessPoolExecutor(a.workers) as ex:
                for tid, why in ex.map(_build_job, [(d, clones[d['repo']], a.scratch / 'g5', pycg / 'bin/python',
                                                     a.scratch / 'uvcache', result(d)) for d in batch]):
                    print(tid, why, flush=True)
            i += 2 * a.workers

    by_merge = sorted(details, key=lambda d: d['merged_at'])
    strict = [d for d in by_merge if d['issue_created'][:10] >= B.CUTOFF]
    prebuild(strict)
    load = lambda d: json.load(gzip.open(result(d), 'rt')) if result(d).exists() else None
    if sum(bool(r and r['task']) for r in map(load, strict)) < B.MIN_TASKS:
        prebuild(by_merge)

    def outcome(d):
        r = load(d)
        if r is None:
            raise RuntimeError(f"{d['repo']}#{d['pr']} was not built")
        return r['task'], r['reason']

    tasks, reasons, rule = select(details, outcome)
    errors = {k: v for pool in reasons.values() for k, v in pool.items() if k.startswith('error')}
    if errors:
        sys.exit(f'builds raised, fix and rerun before freezing: {errors}')
    hop = json.loads(a.config.read_text())['one_hop']
    all_rank, all_cells = [], []
    for t in tasks:
        rk, cells = cells_for(t, hop)
        all_rank += [dict(task=t['task_id'], key=list(k), ranking=v) for k, v in rk.items()]
        all_cells += cells
    with gzip.open(a.out / 'postcutoff_rankings.json.gz', 'wt') as fh:
        json.dump(all_rank, fh)
    keys = list(all_cells[0])
    (a.out / 'postcutoff_cells.csv').write_text(
        '\n'.join([','.join(keys)] + [','.join(str(c[k]) for k in keys) for c in all_cells]) + '\n')
    with gzip.open(a.out / 'postcutoff_full.json.gz', 'wt') as fh:       # for the GPU readouts; not committed
        json.dump(tasks, fh)
    fill = dict(N_NEW=str(len(tasks)), rule=rule, reasons=reasons, candidates=len(cache['candidates']),
                details=len(details), strict_pool=sum(d['issue_created'][:10] >= B.CUTOFF for d in details),
                by_repo=dict(Counter(t['repo'] for t in tasks)))
    (a.out / 'fill_g5.json').write_text(json.dumps(fill, indent=2) + '\n')
    write_manifest(a.manifest, rule, reasons, tasks)
    print(json.dumps(fill, indent=2))


if __name__ == '__main__':
    main()
