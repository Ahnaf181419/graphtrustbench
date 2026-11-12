import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from gtb6 import repo_ops as O

BUGGY = 'def add(a, b):\n    return a - b\n\n\ndef other():\n    return 1\n'
# git anchors a hunk without trailing context to the end of the file, so the fixture carries context
# on both sides, as real diffs do.
CODE_FIX = ('--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n'
            '-    return a - b\n+    return a + b\n \n')
TEST_PATCH = ('--- /dev/null\n+++ b/tests/test_x.py\n@@ -0,0 +1,7 @@\n+from pkg.x import add, other\n+\n'
              '+def test_add():\n+    assert add(2, 3) == 5\n+\n+def test_other():\n+    assert other() == 1\n')


def local_run(repo, targets):
    """Test-only runner: the same pytest command the Docker runner uses, on the host, for toy repos."""
    out = Path(repo) / '.gtb_outcomes.json'
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS), PYTHONDONTWRITEBYTECODE='1', GTB_OUTCOMES=str(out))
    subprocess.run([sys.executable, *O.pytest_args(repo, targets)], cwd=repo, env=env, capture_output=True, timeout=120)
    return O.read_outcomes(out)


class RepoOps(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(f'{self.dir}/pkg'); os.makedirs(f'{self.dir}/tests')
        Path(f'{self.dir}/pkg/__init__.py').write_text('')
        Path(f'{self.dir}/pkg/x.py').write_text(BUGGY)
        Path(f'{self.dir}/tests/__init__.py').write_text('')
        git = lambda *a: subprocess.run(['git', *a], cwd=self.dir, check=True, capture_output=True)
        git('init', '-q'); git('add', '-A')
        git('-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '-qm', 'base')
        self.base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.dir, text=True).strip()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_split_reference_separates_code_and_tests(self):
        code, test, gold = O.split_reference(CODE_FIX + TEST_PATCH)
        self.assertIn('pkg/x.py', code)
        self.assertIn('tests/test_x.py', test)
        self.assertEqual(gold, ['pkg/x.py'])

    def test_extract_diff_takes_the_last_fenced_block(self):
        text = f'first try:\n```diff\n--- a/q.py\n+++ b/q.py\n```\nfinal:\n```diff\n{CODE_FIX}```\n'
        self.assertEqual(O.extract_diff(text), CODE_FIX)
        self.assertIsNone(O.extract_diff('no patch here'))

    def test_extract_diff_keeps_fences_inside_the_diff(self):
        # fastapi docstrings hold Markdown fences: a diff line '+    ```python' must not close the block
        diff = ('--- a/q.py\n+++ b/q.py\n@@ -1,2 +1,5 @@\n def f():\n+    """\n+    ```python\n'
                '+    ```\n+    """\n     return 1\n')
        self.assertEqual(O.extract_diff(f'Here:\n```diff\n{diff}```\nDone.\n'), diff)
        self.assertEqual(O.extract_diff(f'```diff\n{diff}```'), diff)

    def test_apply_patch_falls_back_to_fuzz(self):
        fuzzy = CODE_FIX.replace(' def add(a, b):', ' def add(a, bb):')    # context line no longer matches
        self.assertEqual(O.apply_patch(self.dir, CODE_FIX), 'git-p1')
        O.reset(self.dir, self.base)
        self.assertTrue(O.apply_patch(self.dir, fuzzy).startswith('patch'))
        self.assertIn('a + b', Path(f'{self.dir}/pkg/x.py').read_text())
        O.reset(self.dir, self.base)
        self.assertIsNone(O.apply_patch(self.dir, 'garbage\n'))

    def test_apply_patch_creates_and_deletes_files_with_a_slash_headers(self):
        # the competition patches have no final newline and write new and deleted files with
        # '--- a/<path>' / '+++ b/<path>' headers instead of '/dev/null'
        create = '--- a/tests/test_new.py\n+++ b/tests/test_new.py\n@@ -0,0 +1,2 @@\n+def test_n():\n+    pass'
        self.assertIsNotNone(O.apply_patch(self.dir, create))
        self.assertEqual(Path(f'{self.dir}/tests/test_new.py').read_text(), 'def test_n():\n    pass\n')
        Path(f'{self.dir}/pkg/gone.py').write_text('x = 1\n')
        subprocess.run(['git', 'add', '-A'], cwd=self.dir, check=True)
        gone = '--- a/pkg/gone.py\n+++ b/pkg/gone.py\n@@ -1 +0,0 @@\n-x = 1\n'
        self.assertIsNotNone(O.apply_patch(self.dir, gone))
        self.assertFalse(Path(f'{self.dir}/pkg/gone.py').exists())

    def test_derive_tests_finds_fail_to_pass_and_pass_to_pass(self):
        t = O.derive_tests(self.dir, self.base, CODE_FIX, TEST_PATCH, local_run)
        self.assertEqual(t['f2p'], ['tests/test_x.py::test_add'])
        self.assertEqual(t['p2p'], ['tests/test_x.py::test_other'])
        self.assertEqual(Path(f'{self.dir}/pkg/x.py').read_text(), BUGGY)          # left at the base commit

    def test_test_ids_are_relative_to_the_invocation_dir_not_the_rootdir(self):
        # rich keeps a pytest.ini in tests/, so pytest's rootdir (and its node ids) start there
        Path(f'{self.dir}/tests/pytest.ini').write_text('[pytest]\n')
        subprocess.run(['git', 'add', '-A'], cwd=self.dir, check=True)
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '-qm', 'ini'],
                       cwd=self.dir, check=True)
        base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.dir, text=True).strip()
        t = O.derive_tests(self.dir, base, CODE_FIX, TEST_PATCH, local_run)
        self.assertEqual(t['f2p'], ['tests/test_x.py::test_add'])
        f2p, p2p = t['f2p'], t['p2p']
        self.assertTrue(O.evaluate(self.dir, base, f'```diff\n{CODE_FIX}```', TEST_PATCH, f2p, p2p, local_run)['resolved'])

    def test_node_ids_with_colons_in_parameters_are_selected(self):
        # requests parametrizes with IPv6 URLs, so '::' appears inside node ids, which pytest cannot
        # take as command-line targets
        p_test = ('--- /dev/null\n+++ b/tests/test_p.py\n@@ -0,0 +1,6 @@\n+import pytest\n+from pkg.x import add\n+\n'
                  "+@pytest.mark.parametrize('a', [2, 9], ids=['http://fe80::1', 'plain'])\n"
                  '+def test_p(a):\n+    assert add(a, 3) == a + 3\n')
        t = O.derive_tests(self.dir, self.base, CODE_FIX, p_test, local_run)
        self.assertEqual(t['f2p'], ['tests/test_p.py::test_p[http://fe80::1]', 'tests/test_p.py::test_p[plain]'])
        good = O.evaluate(self.dir, self.base, f'```diff\n{CODE_FIX}```', p_test, t['f2p'][:1], [], local_run)
        self.assertTrue(good['resolved'], good)
        # older pytest (the dated task environments) rejects such ids as arguments: only files go on the line
        argv = O.pytest_args(self.dir, t['f2p'])
        self.assertIn('tests/test_p.py', argv)
        self.assertFalse(any('::' in a for a in argv))

    def test_evaluate_resolves_a_correct_fix_only(self):
        f2p, p2p = ['tests/test_x.py::test_add'], ['tests/test_x.py::test_other']
        good = O.evaluate(self.dir, self.base, f'```diff\n{CODE_FIX}```', TEST_PATCH, f2p, p2p, local_run)
        self.assertTrue(good['resolved'])
        noop = O.evaluate(self.dir, self.base, 'I cannot fix this.', TEST_PATCH, f2p, p2p, local_run)
        self.assertFalse(noop['resolved'])
        test_only = O.evaluate(self.dir, self.base, '```diff\n' + TEST_PATCH.replace('== 5', '== -1') + '```',
                               TEST_PATCH, f2p, p2p, local_run)
        self.assertFalse(test_only['resolved'])        # edits to test files are dropped

    def test_a_crashed_run_is_not_read_as_test_outcomes(self):
        # rich_4006's base run timed out: every test that passed after the fix became F2P and P2P was empty
        self.assertIsNone(O.read_outcomes(Path(self.dir) / 'missing.json'))
        crash_before = lambda repo, tg: None if Path(f'{repo}/pkg/x.py').read_text() == BUGGY else local_run(repo, tg)
        t = O.derive_tests(self.dir, self.base, CODE_FIX, TEST_PATCH, crash_before)
        self.assertEqual((t['f2p'], t['p2p'], t.get('error')), ([], [], 'base run produced no outcomes'))
        crash_after = lambda repo, tg: local_run(repo, tg) if Path(f'{repo}/pkg/x.py').read_text() == BUGGY else None
        t = O.derive_tests(self.dir, self.base, CODE_FIX, TEST_PATCH, crash_after)
        self.assertEqual((t['f2p'], t['p2p'], t.get('error')), ([], [], 'after run produced no outcomes'))
        # a test file that fails at collection before the fix still writes outcomes: its tests are F2P
        self.assertEqual(O.derive_tests(self.dir, self.base, CODE_FIX, TEST_PATCH, local_run)['f2p'],
                         ['tests/test_x.py::test_add'])
        crashed = O.evaluate(self.dir, self.base, f'```diff\n{CODE_FIX}```', TEST_PATCH,
                             ['tests/test_x.py::test_add'], [], lambda repo, tg: None)
        self.assertEqual((crashed['resolved'], crashed['crashed']), (False, True))

    def test_evaluate_never_runs_without_a_runner(self):
        with self.assertRaises(TypeError):
            O.evaluate(self.dir, self.base, CODE_FIX, TEST_PATCH, [], [], None)


@unittest.skipUnless(shutil.which('docker') and subprocess.run(['docker', 'image', 'inspect', 'gtb-tasks:py3.12'],
                                                                 capture_output=True).returncode == 0,
                     'Docker image gtb-tasks:py3.12 not built (paper/harness/Dockerfile.tasks)')
class DockerRunner(unittest.TestCase):
    def test_docker_runner_reports_outcomes_without_network(self):
        from gtb6 import docker_runner
        d = tempfile.mkdtemp()
        try:
            Path(f'{d}/test_n.py').write_text(
                'import socket\n\ndef test_ok():\n    assert 1\n\ndef test_net():\n'
                '    socket.create_connection(("1.1.1.1", 53), timeout=2)\n')
            got = docker_runner.make('gtb-tasks:py3.12')(d, ['test_n.py'])
            self.assertEqual(got, {'test_n.py::test_net': 'failed', 'test_n.py::test_ok': 'passed'})
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
