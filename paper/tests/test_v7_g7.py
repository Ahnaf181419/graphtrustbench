"""Plan 4 G7: evaluate every generated patch in Docker (H5, H6)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
sys.path.insert(0, str(ROOT / 'paper/tests'))
from test_gtb6_repo_ops import CODE_FIX, TEST_PATCH
from test_v7_g4 import IMAGE, needs_docker_image, toy_repo


class G7(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @needs_docker_image
    def test_evaluate_rows_on_a_toy_repo_reuses_identical_diffs(self):
        import v7_g4_trace as G4
        import v7_g7_evaluate as G
        base = toy_repo(self.dir)
        good = f'Here is the fix.\n```diff\n{CODE_FIX}```'
        rows = [dict(id='C0|t', raw=good), dict(id='X|t|missing|13', raw='I cannot help.'),
                dict(id='C0r1|t', raw=good + '\n')]
        docker = G4.docker_cmd(IMAGE, python='python')
        calls = []
        run = lambda r, tg: calls.append(tg) or G4._outcomes(docker, r, tg)
        out = G.evaluate_rows(self.dir, base, rows, TEST_PATCH, ['tests/test_x.py::test_add'],
                              ['tests/test_x.py::test_other'], run)
        self.assertEqual([(o['id'], o['resolved'], o['has_diff']) for o in out],
                         [('C0|t', True, True), ('X|t|missing|13', False, False), ('C0r1|t', True, True)])
        self.assertEqual(len(calls), 1)                 # no diff: no run; the repeated diff is reused
        self.assertTrue(out[2]['reused'])

    def test_tests_overlay_and_exclusions(self):
        import v7_g7_evaluate as G
        first = {'a': dict(install='ok', tests=dict(f2p=['t::x'], p2p=[]), reason=None),
                 'b': dict(install='ok', tests=dict(f2p=['t::y'], p2p=[]), reason=None),
                 'c': dict(install='boom', tests=None, reason='install failed'),
                 'd': dict(install='ok', tests=dict(f2p=[], p2p=['t::z']), reason='no F2P test')}
        again = {'b': dict(install='ok', tests=dict(f2p=[], p2p=[], error='base run produced no outcomes'),
                           reason='base run produced no outcomes')}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.json').write_text(json.dumps(first)); (Path(d) / 'b.json').write_text(json.dumps(again))
            tests = G.load_tests(Path(d) / 'a.json', Path(d) / 'b.json')
        self.assertEqual({t: G.exclusion(v) for t, v in tests.items()},
                         {'a': None, 'b': 'base run produced no outcomes', 'c': 'install failed', 'd': 'no F2P test'})
        # trace-only exclusions of G4 do not exclude a task from patch evaluation
        self.assertIsNone(G.exclusion(dict(install='ok', tests=dict(f2p=['t::x'], p2p=[]), reason='no trace output')))


if __name__ == '__main__':
    unittest.main()
