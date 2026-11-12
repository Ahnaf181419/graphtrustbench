"""Q5a: build post-cutoff tasks from GitHub, with candidates, BM25 and a PyCG call graph.

Registered selection (see the second registration):
  repositories   fastapi/fastapi, Textualize/rich, psf/requests
  pull request   merged on or after CUTOFF, and its body says it closes an issue (GitHub closing forms)
  issue          opened on or after CUTOFF (fallback if fewer than MIN_TASKS qualify: merge date only)
  patch          changes at least one non-test .py file and at least one test file
  tests          at least one fail-to-pass test after derive_tests()
  order / cap    by merge date, first MAX_TASKS that pass every check

Substitutions from the guide (stated in the registration): GitHub access goes through `gh api`
(already authenticated, so no token in code), and BM25 is localization_retrieval.bm25 so both
task sets share one tokenizer.
"""
import ast
import json
import os
import re
import subprocess

from localization_retrieval import bm25 as _bm25

CUTOFF = '2025-03-01'
REPOS = ['fastapi/fastapi', 'Textualize/rich', 'psf/requests']
MIN_TASKS, MAX_TASKS = 40, 100
CLOSES = re.compile(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)', re.I)
# GitHub's closing-reference forms: KEYWORD[:] #N | OWNER/REPO#N | https://github.com/OWNER/REPO/issues/N
CLOSING = re.compile(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?):?\s+'
                     r'(?:#|(?:https?://github\.com/)?([\w.-]+/[\w.-]+)(?:#|/issues/))(\d+)', re.I)


def closed_issue(body, repo):
    """The first issue of `repo` the body says it closes, fixes or resolves, or None."""
    for other, n in CLOSING.findall(body or ''):
        if not other or other.lower() == repo.lower():
            return int(n)
    return None


# ---------- GitHub, through the authenticated gh CLI ----------

def _gh(path, params=(), accept=None):
    cmd = ['gh', 'api', path, '--method', 'GET']
    for k, v in params:
        cmd += ['-f', f'{k}={v}']
    if accept:
        cmd += ['-H', f'Accept: {accept}']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    r.check_returncode()
    return r.stdout


def gh(path, **params):
    """`gh api <path>` as JSON; params become query fields."""
    return json.loads(_gh(path, sorted(params.items())))


def candidate_prs(repo, cutoff=CUTOFF, max_pages=10, until=None):
    """Merged PRs on or after the cutoff (and on or before `until`, if given) that say they close an
    issue, candidates oldest-created first (search sort). The registered merge-date ordering happens
    later, at selection. The search API returns at most 1,000 results (10 pages); a repository with
    more is searched in merge-date windows."""
    out = []
    for page in range(1, max_pages + 1):
        q = f'repo:{repo} is:pr is:merged ' + (f'merged:{cutoff}..{until}' if until else f'merged:>={cutoff}')
        res = gh('search/issues', q=q, sort='created', order='asc', per_page=100, page=page)
        items = res.get('items', [])
        for it in items:
            issue = closed_issue(it.get('body'), repo)
            if issue is not None:
                out.append(dict(repo=repo, pr=it['number'], issue=issue))
        if len(items) < 100:
            break
    return out


def pr_details(repo, pr, issue):
    p = gh(f'repos/{repo}/pulls/{pr}')
    i = gh(f'repos/{repo}/issues/{issue}')
    if 'pull_request' in i:          # the "issue" is itself a PR: not a task
        return None
    diff = _gh(f'repos/{repo}/pulls/{pr}', accept='application/vnd.github.v3.diff')
    return dict(repo=repo, pr=pr, issue=issue, merged_at=p['merged_at'], issue_created=i['created_at'],
                base_sha=p['base']['sha'], head_sha=p['head']['sha'], merge_sha=p['merge_commit_sha'],
                problem_statement=(i['title'] or '') + '\n\n' + (i['body'] or ''), patch=diff)


def merge_base(repo_dir, base_sha, pr):
    """The commit the PR diff is computed against (GitHub diffs are three-dot). The PR head is
    fetched from refs/pull/N/head, which exists even when the branch lived in a fork."""
    subprocess.run(['git', 'fetch', '-q', 'origin', f'pull/{pr}/head:gtb-pr-{pr}'], cwd=repo_dir,
                   capture_output=True, text=True)
    r = subprocess.run(['git', 'merge-base', base_sha, f'gtb-pr-{pr}'], cwd=repo_dir, capture_output=True, text=True)
    return r.stdout.strip() or base_sha


# ---------- candidates ----------

def package_dirs(repo_root):
    """Top-level importable packages (directories with __init__.py), skipping tests and docs."""
    skip = {'tests', 'test', 'docs', 'docs_src', 'examples', 'scripts', 'benchmarks'}
    roots = [repo_root] + ([os.path.join(repo_root, 'src')] if os.path.isdir(os.path.join(repo_root, 'src')) else [])
    out = []
    for r in roots:
        for d in sorted(os.listdir(r)):
            full = os.path.join(r, d)
            if d not in skip and os.path.isfile(os.path.join(full, '__init__.py')):
                out.append(full)
    return out


def extract_definitions(repo_root, pkg_dir):
    """Every function, method and class definition, with node ID module.qualname."""
    parent = os.path.dirname(pkg_dir)
    defs = []
    for dp, _, files in os.walk(pkg_dir):
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            path = os.path.join(dp, f)
            src = open(path, encoding='utf-8', errors='replace').read()
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            mod = os.path.relpath(path, parent)[:-3].replace(os.sep, '.')
            mod = mod[:-9] if mod.endswith('.__init__') else mod
            rel = os.path.relpath(path, repo_root).replace(os.sep, '/')

            def walk(body, prefix):
                for n in body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        q = prefix + n.name
                        defs.append(dict(node_id=f'{mod}.{q}', module=mod, qualname=q, path=rel,
                                         lineno=n.lineno, source=ast.get_source_segment(src, n) or ''))
                        walk(n.body, q + '.' if isinstance(n, ast.ClassDef) else q + '.<locals>.')
            walk(tree.body, '')
    return [d for d in defs if '<locals>' not in d['qualname']]


# ---------- BM25: the registered tokenizer (localization_retrieval.bm25) ----------

def candidate_bm25(issue_text, defs):
    """{node_id: bm25} with the same tokenizer as the competition tasks."""
    candidates = {d['node_id']: dict(id=d['node_id'], path=d['path'], text=d['source']) for d in defs}
    return _bm25(issue_text, candidates)


# ---------- call graph ----------

def run_pycg(pkg_dir, out_json, python, timeout=3600, max_iter=None):
    """PyCG over every module in the package, under the given interpreter (PyCG needs <= 3.10).
    max_iter caps PyCG's fixed-point iterations; PYTHONHASHSEED is fixed so set order, and with it a
    capped graph, is reproducible. Returns {caller: [callees]} or None on failure."""
    entries = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(pkg_dir) for f in fs if f.endswith('.py'))
    cmd = [python, '-m', 'pycg', '--package', os.path.dirname(pkg_dir), *entries, '-o', out_json]
    if max_iter is not None:
        cmd[3:3] = ['--max-iter', str(max_iter)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, PYTHONHASHSEED='0'))
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(out_json):
        return None
    return json.load(open(out_json))


def to_succ(cg, node_ids):
    """Keep caller -> callee pairs where both are candidate definitions."""
    known = set(node_ids)
    return {u: sorted({v for v in vs if v in known and v != u}) for u, vs in cg.items() if u in known}
