#!/usr/bin/env python3
"""G4 (second registration §4, §5.2, H7): F2P/P2P tests, traced call edges, miss kinds, repaired graphs.

Upstream code (dependency installs, test suites, the tracer) runs only in Docker containers
(`gtb-tasks:py3.12`, host uid, no network except for the dependency install). Each task's
competition snapshot (the frozen pre-fix tree the registered candidates come from) is extracted to a
scratch directory, dependencies are resolved as of the task's creation date (uv --exclude-newer),
and the scratch directory is removed after the task. Resumable: one record per task in <out>/work/.
"""
import argparse
import ast
import gzip
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'paper/scripts'
sys.path.insert(0, str(SCRIPTS))
import gemma_readout as GR
from gtb6 import docker_runner, misskinds, repo_ops, retrievers as R, stats as S
from localization_retrieval import edge_key, one_hop

IMAGE = 'gtb-tasks:py3.12'
VENV_PY = '/repo/.venv/bin/python'
INSTALL = {'fastapi/fastapi': ['-e', '.', '-r', 'requirements-tests.txt'],       # guide G4.2
           'Textualize/rich': ['-e', '.', 'pytest', 'attrs'],
           'psf/requests': ['-e', '.', '-r', 'requirements-dev.txt']}
# after fastapi moved its test requirements into pyproject dependency groups (2026-01-10, #14676)
INSTALL_GROUPS = {'fastapi/fastapi': ('requirements-tests.txt', ['-e', '.[all]', '--group', 'tests'])}
PKG = {'fastapi/fastapi': 'fastapi', 'Textualize/rich': 'rich', 'psf/requests': 'requests'}
KIND_WORDS = {'dunder': 'implicit dunder calls such as __init__', 'decorated': 'decorated functions',
              'method': 'method calls resolved at run time', 'function': 'plain function calls',
              'unresolved': 'definitions the index could not find'}


# ---------------------------------------------------------------- containers
def docker_cmd(image, python=VENV_PY, memory=docker_runner.MEMORY):
    """run(repo, argv, env=None, network=False, mounts=(), timeout=1800) inside a container.
    argv[0] == 'python' is replaced by the environment's interpreter."""
    def run(repo, argv, env=None, network=False, mounts=(), timeout=1800):
        cmd = ['docker', 'run', '--rm', '--memory', memory, '--memory-swap', memory,
               '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{Path(repo).resolve()}:/repo', '-v', f'{SCRIPTS}:/gtb:ro', '-w', '/repo',
               '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1']
        if not network:
            cmd += ['--network', 'none']
        for host, inside in mounts:
            cmd += ['-v', f'{host}:{inside}']
        for k, v in (env or {}).items():
            cmd += ['-e', f'{k}={v}']
        argv = [python if a == 'python' else a for a in argv]
        try:
            return subprocess.run(cmd + [image, 'timeout', str(timeout), *argv], capture_output=True, text=True,
                                  timeout=timeout + 120)
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(cmd, 124, '', f'timeout: {e}')
    return run


def install_line(repo, repo_name):
    """The guide's install arguments, or the repository's test dependency group once its requirements
    file is gone at that commit."""
    alt = INSTALL_GROUPS.get(repo_name)
    if alt and not (Path(repo) / alt[0]).exists():
        return alt[1]
    return INSTALL[repo_name]


def _require(run):
    if not callable(run):
        raise TypeError('a container runner is required (docker_cmd); upstream code never runs on the host')


def prepare(repo, base, repo_name, exclude_newer, run, cache=None):
    """Base tree plus a venv at /repo/.venv with the guide's dependencies as of exclude_newer."""
    _require(run)
    repo_ops.reset(repo, base)
    deps = ' '.join(shlex.quote(a) for a in install_line(repo, repo_name))
    script = (f'uv venv -q --python /usr/local/bin/python /repo/.venv && '
              f'uv pip install -q --python {VENV_PY} --exclude-newer {exclude_newer} {deps}')
    r = run(repo, ['sh', '-c', script], env=dict(UV_CACHE_DIR='/uvcache', UV_LINK_MODE='copy'), network=True,
            mounts=[(cache, '/uvcache')] if cache else (), timeout=1800)
    repo_ops.reset(repo, base)
    return r.returncode == 0, (r.stderr or '')[-800:]


def trace_task(repo, pkg_rel, targets, run):
    """Traced (caller module, qualname, callee module, qualname) edges, or None when no trace was written."""
    _require(run)
    out = Path(repo) / '.gtb_trace.json'
    run(repo, ['python', *repo_ops.pytest_args(repo, targets), '-p', 'gtb6.tracer'],
        env=dict(GTB_SRC=f'/repo/{pkg_rel}', GTB_OUT='/repo/.gtb_trace.json', PYTHONPATH='/gtb',
                 GTB_OUTCOMES='/repo/.gtb_outcomes.json'))
    repo_ops.read_outcomes(Path(repo) / '.gtb_outcomes.json')        # removes it
    try:
        return json.loads(out.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        if out.exists():
            out.unlink()


# ---------------------------------------------------------------- mapping and accounting
def _qualnames(path):
    """{(start, end): co_qualname} for every def/class in a file; start includes decorators."""
    try:
        tree = ast.parse(Path(path).read_text(encoding='utf-8', errors='replace'))
    except (SyntaxError, OSError):
        return {}
    out = {}

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + n.name
                out[(min([n.lineno] + [d.lineno for d in n.decorator_list]), n.end_lineno)] = q
                walk(n.body, q + ('.' if isinstance(n, ast.ClassDef) else '.<locals>.'))
            elif hasattr(n, 'body') and isinstance(n.body, list):
                walk(n.body + getattr(n, 'orelse', []) + getattr(n, 'finalbody', []), prefix)
    walk(tree.body, '')
    return out


def _modules(path):
    """Module names of a repo-relative file, as localization_data.extract_candidates derives them."""
    parts = list(PurePosixPath(path).with_suffix('').parts)
    parts = parts[1:] if parts[0] == 'src' else parts
    return ['.'.join(parts[:-1])] if parts[-1] == '__init__' else ['.'.join(parts)]


def make_resolver(repo, candidates):
    """resolve(module, qualname) -> node id: an exact ID, else the unique candidate bound to that definition."""
    index, by_path = {}, defaultdict(list)
    for n, c in candidates.items():
        by_path[c['path']].append(n)
    for path, nodes in by_path.items():
        quals = _qualnames(Path(repo) / path)
        for n in nodes:
            for s, e in candidates[n]['spans']:
                q = quals.get((s, e))
                for m in _modules(path) if q else ():
                    index[(m, q)] = n if index.get((m, q), n) == n else None
    return lambda module, qualname: (f'{module}.{qualname}' if f'{module}.{qualname}' in candidates
                                     else index.get((module, qualname)))


def summarize_trace(edges, resolve):
    """({(u, v): (callee module, qualname)} for mapped edges between distinct nodes, unmapped count)."""
    raw, unmapped = {}, 0
    for rm, rq, cm, cq in edges:
        u, v = resolve(rm, rq), resolve(cm, cq)
        if u and v and u != v:
            raw.setdefault((u, v), (cm, cq))
        else:
            unmapped += 1
    return raw, unmapped


def repaired(supplied, missing):
    """Supplied edges plus one 'calls' edge per missed pair, sorted by the registered edge key."""
    if not missing:
        return supplied
    return sorted(list(supplied) + [dict(source=u, target=v, type='calls', key=0) for u, v in missing], key=edge_key)


def exclusions(records):
    return Counter(r['reason'] for r in records if r['reason'])


def included(records):
    return [r for r in records if not r['reason']]


def h7(rep_rank, clean_rank, gold):
    """Recall@5 of one-hop on the repaired graph minus the supplied graph, per task (H7, one-sided)."""
    tasks = sorted(rep_rank)
    d = [GR.recall5(rep_rank[t], gold[t]) - GR.recall5(clean_rank[t], gold[t]) for t in tasks]
    m, lo, hi = S.ci(d)
    return dict(tasks=len(tasks), mean=m, lo=lo, hi=hi, p_one_sided=S.one_sided_p(d),
                membership_changed=sum(set(rep_rank[t][:5]) != set(clean_rank[t][:5]) for t in tasks))


# ---------------------------------------------------------------- one task
def pkg_dir(repo, name):
    p = PKG[name]
    return f'src/{p}' if (Path(repo) / 'src' / p).is_dir() else p


def process(item, task, config, gold_t, clean_committed, work_root, trace_dir, cache):
    from gtb6.corpus import load_task
    t, t0 = item['instance_id'], time.perf_counter()
    rec = dict(task=t, repo=task['repo'], base_commit=task['base_commit'], reason=None)
    repo = Path(work_root) / t
    shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir(parents=True)
    try:
        with tarfile.open(ROOT / item['snapshot_path']) as tf:
            tf.extractall(repo, filter='data')
        base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        rec['snapshot_head'] = base
        run = docker_cmd(IMAGE)
        ok, err = prepare(repo, base, task['repo'], task['created_at'], run, cache)
        rec['install'] = 'ok' if ok else err
        if not ok:
            rec['reason'] = 'install failed'
            return rec
        code, test, _ = repo_ops.split_reference(task['patch'] + '\n' + task['test_patch'])
        tests = repo_ops.derive_tests(repo, base, code, test, lambda r, tg: _outcomes(run, r, tg))
        rec['tests'] = tests
        if tests.get('error'):
            rec['reason'] = tests['error']
            return rec
        if not tests['f2p']:
            rec['reason'] = 'no F2P test'
            return rec
        repo_ops.reset(repo, base)
        misskinds.clear_cache()
        if repo_ops.apply_patch(repo, test) is None:
            rec['reason'] = 'test part did not apply'
            return rec
        pkg = pkg_dir(repo, task['repo'])
        edges_tr = trace_task(repo, pkg, tests['f2p'] + tests['p2p'], run)
        if edges_tr is None:
            rec['reason'] = 'no trace output'
            return rec
        with gzip.open(Path(trace_dir) / f'{t}.json.gz', 'wt') as fh:
            json.dump(edges_tr, fh)
        candidates, lexical, edges = load_task(ROOT, item, config)
        raw, unmapped = summarize_trace(edges_tr, make_resolver(repo, candidates))
        rec['trace'] = dict(raw_edges=len(edges_tr), executed=len(raw), unmapped=unmapped)
        if not raw:
            rec['reason'] = 'no mapped edges'
            return rec
        supplied = {(u, v) for u, vs in R._mapped(lexical, edges).items() for v in vs}
        miss = sorted(set(raw) - supplied)
        src_parent = str((repo / pkg).parent)
        rec['trace'].update(missing=len(miss), kinds=dict(Counter(misskinds.classify(src_parent, *raw[e]) for e in miss)))
        rec['missing_pairs'] = [list(e) for e in miss]
        hop = config['one_hop']
        clean = R.rank(one_hop(lexical, edges, **hop)[0], lexical, candidates)
        if clean != clean_committed:
            raise AssertionError(f'{t}: rebuilt clean one-hop ranking differs from the committed run')
        rec['clean_ranking'] = clean
        rec['repaired_ranking'] = R.rank(one_hop(lexical, repaired(edges, set(miss)), **hop)[0], lexical, candidates)
        return rec
    finally:
        misskinds.clear_cache()
        rec['seconds'] = round(time.perf_counter() - t0, 1)
        shutil.rmtree(repo, ignore_errors=True)


def _outcomes(run, repo, targets):
    """derive_tests' runner: pytest with the outcomes plugin, inside the task's container."""
    if not targets:
        return {}
    run(repo, ['python', *repo_ops.pytest_args(repo, targets)], env=dict(PYTHONPATH='/gtb', GTB_OUTCOMES='/repo/.gtb_outcomes.json'))
    return repo_ops.read_outcomes(Path(repo) / '.gtb_outcomes.json')


def _job(args):
    item, task, config, gold_t, clean_c, work_root, trace_dir, cache, out = args
    try:
        rec = process(item, task, config, gold_t, clean_c, work_root, trace_dir, cache)
    except Exception as e:                                       # logged, never silently dropped
        rec = dict(task=item['instance_id'], reason=f'error: {type(e).__name__}: {e}'[:300])
    Path(out).write_text(json.dumps(rec))
    return rec


# ---------------------------------------------------------------- analysis
def fill(records, gold):
    inc = included(records)
    ex = sum(r['trace']['executed'] for r in inc)
    ms = sum(r['trace']['missing'] for r in inc)
    kinds = Counter()
    for r in inc:
        kinds.update(r['trace']['kinds'])
    v = dict(TR_TASKS=len(inc), TR_EDGES=f'{ex:,}', TR_MISS=str(round(100 * ms / ex)) if ex else 'n/a',
             kinds=dict(kinds), exclusions=dict(exclusions(records)), tasks=len(records),
             f2p_tasks=sum(bool(r.get('tests', {}).get('f2p')) for r in records))
    if kinds:
        top = kinds.most_common(1)[0]
        v.update(TR_TOP_KIND=KIND_WORDS[top[0]], TR_TOP_SHARE=str(round(100 * top[1] / ms)))
    if inc:
        v['TR_MISS_MED'] = str(round(100 * statistics.median(r['trace']['missing'] / r['trace']['executed'] for r in inc)))
        out = h7({r['task']: r['repaired_ranking'] for r in inc}, {r['task']: r['clean_ranking'] for r in inc}, gold)
        fmt = lambda x: f'{x:+.4f}'.replace('-', '−')
        v.update(H7_DIFF=fmt(out['mean']), H7_LO=fmt(out['lo']), H7_HI=fmt(out['hi']), H7_p=out['p_one_sided'],
                 REP_M=str(out['membership_changed']), H7_TASKS=out['tasks'])
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='paper/results/<RUNDATE>_g4_trace')
    ap.add_argument('--tasks', nargs='*', help='only these instance ids (smoke run)')
    ap.add_argument('--workers', type=int, default=3)    # x MEMORY must fit in RAM
    ap.add_argument('--scratch', type=Path, default=Path.home() / '.cache/gtb6')
    ap.add_argument('--selection', type=Path, default=ROOT / 'paper/data/heldout_full_2026-11-07.json')
    ap.add_argument('--config', type=Path, default=ROOT / 'paper/config/pilot_v3.json')
    ap.add_argument('--per-task', type=Path, default=ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz')
    a = ap.parse_args()
    selection = [t for t in json.loads(a.selection.read_text())['tasks'] if not a.tasks or t['instance_id'] in a.tasks]
    config = json.loads(a.config.read_text())
    clean_c, _, gold = GR.rankings(GR.load_rows(a.per_task))
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'tasks.jsonl')}
    work, traces = a.out / 'work', a.out / 'traces'
    for d in (work, traces, a.scratch / 'repos', a.scratch / 'uvcache'):
        d.mkdir(parents=True, exist_ok=True)
    (a.out / '.gitignore').write_text('work/\n')
    jobs = [(it, tasks[it['instance_id']], config, gold[it['instance_id']], clean_c[it['instance_id']],
             a.scratch / 'repos', traces, a.scratch / 'uvcache', work / f"{it['instance_id']}.json")
            for it in selection if not (work / f"{it['instance_id']}.json").exists()]
    print(f'{len(selection) - len(jobs)} tasks done earlier, {len(jobs)} to run', flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(_job, j) for j in jobs]), 1):
            r = f.result()
            print(f"[{i}/{len(jobs)}] {r['task']} {r.get('reason') or 'ok'} {r.get('seconds', '')}s "
                  f"f2p={len(r.get('tests', {}).get('f2p', []))} {r.get('trace', '')}", flush=True)
    records = [json.loads((work / f"{it['instance_id']}.json").read_text()) for it in selection]
    errors = [r for r in records if (r['reason'] or '').startswith('error')]
    if errors:
        sys.exit(f'{len(errors)} tasks raised; fix and rerun (their work files are kept): ' +
                 '; '.join(f"{r['task']}: {r['reason']}" for r in errors[:5]))
    keep = ('task', 'repo', 'base_commit', 'snapshot_head', 'install', 'tests', 'reason')
    (a.out / 'tests_competition.json').write_text(json.dumps({r['task']: {k: r.get(k) for k in keep} for r in records},
                                                             indent=1) + '\n')
    (a.out / 'trace_summary.json').write_text(json.dumps(
        {r['task']: dict(trace=r.get('trace'), reason=r['reason'], seconds=r.get('seconds')) for r in records}, indent=1) + '\n')
    inc = included(records)
    with gzip.open(a.out / 'repaired_graphs.json.gz', 'wt') as fh:     # added edges; repaired = supplied + these
        json.dump({r['task']: r['missing_pairs'] for r in inc}, fh)
    with gzip.open(a.out / 'repaired_rankings.json.gz', 'wt') as fh:
        json.dump({r['task']: dict(repaired=r['repaired_ranking'], clean=r['clean_ranking']) for r in inc}, fh)
    v = fill(records, gold)
    (a.out / 'fill_g4.json').write_text(json.dumps(v, indent=2) + '\n')
    print(json.dumps(v, indent=2))


if __name__ == '__main__':
    main()
