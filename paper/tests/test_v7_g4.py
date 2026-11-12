"""Plan 2 Task 2 (G4): Docker-only test runs, traced edges, miss accounting, repaired graphs and H7."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
sys.path.insert(0, str(ROOT / 'paper/tests'))
from test_gtb6_repo_ops import BUGGY, CODE_FIX, TEST_PATCH

IMAGE = 'gtb-tasks:py3.12'
needs_docker_image = unittest.skipUnless(
    shutil.which('docker') and subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True).returncode == 0,
    f'Docker image {IMAGE} not built (paper/harness/Dockerfile.tasks)')

TOY_SRC = '''\
def helper():
    return 1

class Box:
    def open(self):
        return helper()

def outer():
    def inner():
        return helper()
    return inner()
'''


def toy_repo(d):
    git = lambda *a: subprocess.run(['git', *a], cwd=d, check=True, capture_output=True)
    os.makedirs(f'{d}/pkg'); os.makedirs(f'{d}/tests')
    Path(f'{d}/pkg/__init__.py').write_text('')
    Path(f'{d}/pkg/x.py').write_text(BUGGY)
    Path(f'{d}/tests/__init__.py').write_text('')
    git('init', '-q'); git('add', '-A')
    git('-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '-qm', 'base')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d, text=True).strip()


class G4(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @needs_docker_image
    def test_derive_tests_through_docker_on_a_toy_repo(self):
        from gtb6 import docker_runner, repo_ops
        base = toy_repo(self.dir)
        t = repo_ops.derive_tests(self.dir, base, CODE_FIX, TEST_PATCH, docker_runner.make(IMAGE))
        self.assertEqual((t['f2p'], t['p2p']), (['tests/test_x.py::test_add'], ['tests/test_x.py::test_other']))

    @needs_docker_image
    def test_trace_task_runs_the_tracer_in_docker(self):
        import v7_g4_trace as G
        os.makedirs(f'{self.dir}/pkg')
        Path(f'{self.dir}/pkg/__init__.py').write_text('')
        Path(f'{self.dir}/pkg/m.py').write_text(TOY_SRC)
        Path(f'{self.dir}/test_m.py').write_text('import pkg.m\n\ndef test_m():\n    pkg.m.Box().open()\n')
        edges = G.trace_task(self.dir, 'pkg', ['test_m.py'], G.docker_cmd(IMAGE, python='python'))
        self.assertIn(['pkg.m', 'Box.open', 'pkg.m', 'helper'], edges)
        self.assertFalse(Path(f'{self.dir}/.gtb_trace.json').exists())

    @needs_docker_image
    def test_containers_have_a_memory_cap(self):
        # rich_3480's base commit grew a test process to 19 GB and the host OOM killer stepped in
        import v7_g4_trace as G
        from gtb6 import docker_runner
        hog = 'x = b"x" * (512 * 2 ** 20)'
        p = G.docker_cmd(IMAGE, python='python', memory='256m')(self.dir, ['python', '-c', hog])
        self.assertEqual(p.returncode, 137)
        Path(f'{self.dir}/test_hog.py').write_text(f'def test_hog():\n    {hog}\n')
        self.assertIsNone(docker_runner.make(IMAGE, memory='256m')(self.dir, ['test_hog.py']))   # killed: no outcomes

    def test_missing_trace_file_is_an_exclusion_not_zero(self):
        import v7_g4_trace as G
        ran = []
        silent = lambda repo, argv, env=None, **kw: ran.append(argv) or subprocess.CompletedProcess(argv, 1)
        self.assertIsNone(G.trace_task(self.dir, 'pkg', ['tests/test_a.py::t'], silent))
        self.assertEqual(len(ran), 1)
        recs = [dict(task='a', reason='no trace output'), dict(task='b', reason=None), dict(task='c', reason='install failed')]
        self.assertEqual(G.exclusions(recs), Counter({'no trace output': 1, 'install failed': 1}))
        self.assertEqual([r['task'] for r in G.included(recs)], ['b'])

    def test_no_host_pytest_path(self):
        import v7_g4_trace as G
        with self.assertRaises(TypeError):
            G.trace_task(self.dir, 'pkg', ['t.py'], None)
        with self.assertRaises(TypeError):
            G.prepare(self.dir, 'HEAD', 'psf/requests', '2025-01-01T00:00:00Z', None)

    def test_install_line_follows_the_commit(self):
        # fastapi moved its test requirements into pyproject dependency groups (2026-01-10, #14676)
        import v7_g4_trace as G
        self.assertEqual(G.install_line(self.dir, 'fastapi/fastapi'), ['-e', '.[all]', '--group', 'tests'])
        Path(f'{self.dir}/requirements-tests.txt').write_text('pytest\n')
        self.assertEqual(G.install_line(self.dir, 'fastapi/fastapi'), ['-e', '.', '-r', 'requirements-tests.txt'])
        self.assertEqual(G.install_line(self.dir, 'Textualize/rich'), ['-e', '.', 'pytest', 'attrs'])

    def test_resolver_follows_the_candidate_binding(self):
        import v7_g4_trace as G
        os.makedirs(f'{self.dir}/src/pkg')
        Path(f'{self.dir}/src/pkg/__init__.py').write_text('')
        Path(f'{self.dir}/src/pkg/m.py').write_text(TOY_SRC)
        cand = {'pkg.m.helper': dict(path='src/pkg/m.py', spans=[[1, 2]]),
                'pkg.Box.open': dict(path='src/pkg/m.py', spans=[[5, 6]]),        # alias-bound node id
                'pkg.m.outer.inner': dict(path='src/pkg/m.py', spans=[[9, 10]])}
        resolve = G.make_resolver(self.dir, cand)
        self.assertEqual(resolve('pkg.m', 'helper'), 'pkg.m.helper')
        self.assertEqual(resolve('pkg.m', 'Box.open'), 'pkg.Box.open')
        self.assertEqual(resolve('pkg.m', 'outer.<locals>.inner'), 'pkg.m.outer.inner')
        self.assertIsNone(resolve('pkg.m', 'outer'))
        self.assertIsNone(resolve('pkg.other', 'helper'))

    def test_summarize_and_repaired_on_toy_edges(self):
        import v7_g4_trace as G
        ids = {('m', 'f'): 'm.f', ('m', 'g'): 'm.g', ('m', 'C.__init__'): 'm.C.__init__'}
        resolve = lambda mod, q: ids.get((mod, q))
        trace = [['m', 'f', 'm', 'g'], ['m', 'f', 'm', 'C.__init__'], ['m', '<module>', 'm', 'f'],
                 ['m', 'f.<locals>.h', 'm', 'g'], ['m', 'g', 'm', 'g']]
        raw, unmapped = G.summarize_trace(trace, resolve)
        self.assertEqual(raw, {('m.f', 'm.g'): ('m', 'g'), ('m.f', 'm.C.__init__'): ('m', 'C.__init__')})
        self.assertEqual(unmapped, 3)
        supplied = [dict(source='m.f', target='m.g', type='calls', key=0)]
        rep = G.repaired(supplied, {('m.f', 'm.C.__init__')})
        self.assertEqual([(e['source'], e['target']) for e in rep], [('m.f', 'm.C.__init__'), ('m.f', 'm.g')])
        self.assertEqual(G.repaired(supplied, set()), supplied)

    def test_h7_delta_on_toy_gold(self):
        import v7_g4_trace as G
        clean = {'a': ['x', 'y', 'z', 'w', 'v', 'g'], 'b': ['g', 'y', 'z', 'w', 'v', 'x']}
        rep = {'a': ['g', 'y', 'z', 'w', 'v', 'x'], 'b': list(clean['b'])}
        out = G.h7(rep, clean, {'a': {'g'}, 'b': {'g'}})
        self.assertEqual((out['tasks'], out['mean'], out['membership_changed']), (2, 0.5, 1))
        self.assertLessEqual(out['lo'], out['mean'])
        self.assertGreaterEqual(out['hi'], out['mean'])
        self.assertTrue(0 < out['p_one_sided'] < 0.5)       # bootstrap means <= 0 only when 'b' is drawn twice


if __name__ == '__main__':
    unittest.main()
