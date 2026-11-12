"""Patches and tests (second registration §5.2): split a reference patch, derive fail-to-pass and
pass-to-pass tests, extract and apply a model's diff, and decide whether a cell is resolved.

Test execution is injected as run(repo, targets) -> {nodeid: outcome}; real runs use
docker_runner.make(...), so upstream test suites and model-written code never run on the host.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile


def sh(cmd, cwd, timeout=600):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ---------- patch text ----------

def is_test_path(path):
    parts = path.replace('\\', '/').split('/')
    name = parts[-1]
    return (any(p in ('tests', 'test', 'testing') for p in parts[:-1])
            or name.startswith('test_') or name.endswith('_test.py') or name == 'conftest.py')


def split_files(diff_text):
    """Split a unified diff into {path: chunk}; handles 'diff --git' and bare '---/+++' diffs."""
    chunks, cur, path = {}, [], None
    lines = diff_text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        starts_git = ln.startswith('diff --git ')
        starts_bare = (ln.startswith('--- ') and i + 1 < len(lines) and lines[i + 1].startswith('+++ ')
                       and (not cur or not cur[-1].startswith('diff --git ')))
        if starts_git or starts_bare:
            if path is not None:
                chunks[path] = ''.join(cur)
            cur, path = [], None
        cur.append(ln)
        if ln.startswith('+++ '):
            p = ln[4:].strip().split('\t')[0]
            if p == '/dev/null':                      # deleted file: take the old path
                p = next((x[4:].strip().split('\t')[0] for x in reversed(cur) if x.startswith('--- ')), p)
            path = re.sub(r'^[ab]/', '', p)
    if path is not None:
        chunks[path] = ''.join(cur)
    return chunks


def split_reference(diff_text):
    """(code_part, test_part, gold_code_files) of a reference patch."""
    files = split_files(diff_text)
    code = ''.join(c for p, c in files.items() if not is_test_path(p))
    test = ''.join(c for p, c in files.items() if is_test_path(p))
    return code, test, sorted(p for p in files if not is_test_path(p) and p.endswith('.py'))


def extract_diff(text):
    """The model's diff: the last ```diff fenced block, else from the first diff header to the end."""
    # Fences count only at the start of a line: diff lines start with a marker, so a docstring's
    # '+    ```python' stays inside the block.
    blocks = [b for b in re.findall(r'^```(?:diff|patch)?[ \t]*\n(.*?)^```[ \t]*$', text, re.S | re.M)
              if re.search(r'^(diff --git |--- )', b, re.M)]
    if blocks:
        d = blocks[-1]
    else:
        m = re.search(r'^(diff --git |--- )', text, re.M)
        if not m:
            return None
        d = text[m.start():]
    # Keep a final blank context line (' '): stripping spaces too would corrupt the last hunk.
    return d.rstrip('`').rstrip('\n') + '\n'


def strip_test_files(diff_text):
    """Drop the model's edits to test files; the reference test patch is applied on top."""
    return ''.join(c for p, c in split_files(diff_text).items() if not is_test_path(p))


def _normalize(repo, diff_text):
    """Final newline, and '/dev/null' headers for whole-file creations ('@@ -0,0', file absent)
    and deletions ('+0,0 @@'), which the competition patches write as a/ and b/ paths."""
    lines = diff_text.splitlines(keepends=True)
    for i in range(len(lines) - 2):
        old, new, hunk = lines[i], lines[i + 1], lines[i + 2]
        if not (old.startswith('--- ') and new.startswith('+++ ') and hunk.startswith('@@ ')):
            continue
        path = re.sub(r'^[ab]/', '', old[4:].strip().split('\t')[0])
        if hunk.startswith('@@ -0,0 ') and not os.path.exists(os.path.join(repo, path)):
            lines[i] = '--- /dev/null\n'
        elif re.match(r'@@ -\S+ \+0,0 @@', hunk):
            lines[i + 1] = '+++ /dev/null\n'
    text = ''.join(lines)
    return text if text.endswith('\n') else text + '\n'


def apply_patch(repo, diff_text):
    """git apply (-p1, then -p0), then GNU patch with fuzz 3. Returns the method used, or None."""
    if not diff_text or not diff_text.strip():
        return None
    with tempfile.NamedTemporaryFile('w', suffix='.diff', delete=False) as fh:
        fh.write(_normalize(repo, diff_text))
        tmp = fh.name
    try:
        for p in ('-p1', '-p0'):
            if sh(['git', 'apply', '--recount', '--whitespace=nowarn', p, tmp], repo).returncode == 0:
                return f'git{p}'
        for p in ('-p1', '-p0'):
            if sh(['patch', '--dry-run', '--batch', '--forward', '--fuzz=3', p, '-i', tmp], repo).returncode == 0:
                sh(['patch', '--batch', '--forward', '--fuzz=3', '--no-backup-if-mismatch', p, '-i', tmp], repo)
                return f'patch{p}'
        return None
    finally:
        os.unlink(tmp)


def reset(repo, commit):
    """Back to the base commit, removing untracked files and stale bytecode."""
    sh(['git', 'checkout', '-q', '-f', commit], repo)
    sh(['git', 'clean', '-q', '-fdx', '-e', '.venv'], repo)
    for dp, dirs, _ in os.walk(repo):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(dp, '__pycache__'), ignore_errors=True)
            dirs.remove('__pycache__')
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', 'venv', 'node_modules')]


# ---------- tests ----------

SELECT = '.gtb_select.json'


def pytest_args(repo, targets):
    """Interpreter arguments for one pytest run (from the repo root) that writes {nodeid: outcome} to
    $GTB_OUTCOMES. Only test files go on the command line: node ids can contain '::' inside parameters,
    which older pytest cannot parse, so node ids are selected by gtb6.outcomes from <repo>/.gtb_select.json."""
    files = list(dict.fromkeys(t.split('::', 1)[0] for t in targets))
    ids = [t for t in targets if '::' in t]
    select = []
    if ids:
        with open(os.path.join(repo, SELECT), 'w') as fh:
            json.dump(sorted(set(ids) | {f for f in files if f in targets}), fh)
        select = [f'--gtb-select={SELECT}']
    return ['-m', 'pytest', '-p', 'gtb6.outcomes', '-p', 'no:cacheprovider', '-q', '-o', 'addopts=', *select, *files]


def read_outcomes(path):
    """Outcomes a run wrote, or None after a crash, timeout or OOM kill (pytest never reached sessionfinish);
    the file is removed. A collection error still writes the file: its tests are absent, i.e. failed."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _require(run):
    if not callable(run):
        raise TypeError('a test runner is required (docker_runner.make); tests never run on the host by default')


def derive_tests(repo, base, code_patch, test_patch, run):
    """F2P: fail before the code fix, pass after. P2P: pass both times. Scope: the test files the
    reference test patch touches (registered). Returns dict(f2p, p2p, test_files[, error])."""
    _require(run)
    test_files = sorted(p for p in split_files(test_patch) if is_test_path(p) and p.endswith('.py')
                        and os.path.basename(p) != 'conftest.py')
    reset(repo, base)
    if test_patch.strip() and apply_patch(repo, test_patch) is None:
        return dict(f2p=[], p2p=[], test_files=test_files, error='test patch did not apply')
    before = run(repo, [t for t in test_files if os.path.exists(os.path.join(repo, t))])
    if before is None:
        reset(repo, base)
        return dict(f2p=[], p2p=[], test_files=test_files, error='base run produced no outcomes')
    if apply_patch(repo, code_patch) is None:
        reset(repo, base)
        return dict(f2p=[], p2p=[], test_files=test_files, error='code patch did not apply')
    after = run(repo, test_files)
    reset(repo, base)
    if after is None:
        return dict(f2p=[], p2p=[], test_files=test_files, error='after run produced no outcomes')
    f2p = sorted(t for t, o in after.items() if o == 'passed' and before.get(t, 'failed') == 'failed')
    p2p = sorted(t for t, o in after.items() if o == 'passed' and before.get(t) == 'passed')
    return dict(f2p=f2p, p2p=p2p, test_files=test_files)


def evaluate(repo, base, model_text, test_patch, f2p, p2p, run):
    """Resolved iff the model's diff applies and every F2P and P2P test passes with the reference
    test patch applied on top (SWE-bench convention)."""
    _require(run)
    reset(repo, base)
    diff = extract_diff(model_text or '')
    res = dict(has_diff=diff is not None, applied=None, resolved=False, failed_tests=[])
    if diff is None:
        return res
    res['applied'] = apply_patch(repo, strip_test_files(diff))
    if res['applied'] is None:
        reset(repo, base)
        return res
    if test_patch.strip() and apply_patch(repo, test_patch) is None:
        res['applied'] = 'conflict-with-test-patch'
        reset(repo, base)
        return res
    out = run(repo, sorted(set(f2p) | set(p2p)))
    res['crashed'] = out is None                    # timeout / OOM: every test counts as failed
    failed = [t for t in list(f2p) + list(p2p) if (out or {}).get(t) != 'passed']
    res.update(resolved=not failed and len(f2p) > 0, failed_tests=failed)
    reset(repo, base)
    return res
