"""Task 10: offline parts of the post-cutoff task builder (constants, CLOSES, definitions, to_succ)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))

TOY_M = '''\
class C:
    def f(self):
        return 1

def outer():
    def inner():
        return 2
    return inner
'''


class Builder(unittest.TestCase):
    def test_registered_constants(self):
        from gtb6 import builder as B
        self.assertEqual(B.CUTOFF, '2025-03-01')
        self.assertEqual(B.REPOS, ['fastapi/fastapi', 'Textualize/rich', 'psf/requests'])
        self.assertEqual((B.MIN_TASKS, B.MAX_TASKS), (40, 100))

    def test_closes_matches_close_fix_resolve_with_number(self):
        from gtb6 import builder as B
        self.assertEqual(B.CLOSES.findall('Fixes #12'), ['12'])
        self.assertEqual(B.CLOSES.findall('this closes #3 now'), ['3'])
        self.assertEqual(B.CLOSES.findall('resolved #9'), ['9'])
        self.assertEqual(B.CLOSES.findall('see #12 please'), [])
        self.assertEqual(B.CLOSES.findall('fixes issue twelve'), [])

    def test_closed_issue_accepts_githubs_closing_reference_forms(self):
        # the registration: "its body says it closes, fixes or resolves an issue"; GitHub also closes
        # issues by URL, by owner/repo#N and with a colon after the keyword (rich writes URLs)
        from gtb6 import builder as B
        r = 'Textualize/rich'
        self.assertEqual(B.closed_issue('Fixes https://github.com/Textualize/rich/issues/3762\r\n', r), 3762)
        self.assertEqual(B.closed_issue('closes textualize/rich#41', r), 41)
        self.assertEqual(B.closed_issue('Fixes: #12', r), 12)
        self.assertEqual(B.closed_issue('fixes https://github.com/other/repo/issues/5 and resolves #6', r), 6)
        self.assertIsNone(B.closed_issue('see https://github.com/Textualize/rich/issues/9', r))
        self.assertIsNone(B.closed_issue('Fix the documentation search', r))

    def test_candidate_prs_searches_a_merge_date_window(self):
        # the search API returns at most 1,000 results, so busy repositories are searched in windows
        from unittest import mock
        from gtb6 import builder as B
        calls = []
        def fake(path, **params):
            calls.append(params['q'])
            return dict(items=[dict(number=5, body='Fixes #4'), dict(number=6, body='no link')])
        with mock.patch.object(B, 'gh', fake):
            self.assertEqual(B.candidate_prs('psf/requests', until='2025-08-31'),
                             [dict(repo='psf/requests', pr=5, issue=4)])
            B.candidate_prs('psf/requests')
        self.assertEqual(calls, ['repo:psf/requests is:pr is:merged merged:2025-03-01..2025-08-31',
                                 'repo:psf/requests is:pr is:merged merged:>=2025-03-01'])

    @unittest.skipUnless((Path.home() / '.cache/gtb6/pycg/bin/python').exists(), 'PyCG venv not built (v7_g5_build.py)')
    def test_run_pycg_builds_a_call_graph(self):
        # the PyPI wheel of PyCG 0.0.8 ships 'PyCG/' but imports 'pycg' (case-sensitive filesystems fail)
        from gtb6 import builder as B
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / 'toy'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'm.py').write_text('def g():\n    return 1\n\ndef f():\n    return g()\n')
            cg = B.run_pycg(str(pkg), str(Path(d) / 'cg.json'), python=str(Path.home() / '.cache/gtb6/pycg/bin/python'), max_iter=2)
        self.assertIsNotNone(cg)
        self.assertIn('toy.m.g', cg['toy.m.f'])

    def test_run_pycg_caps_iterations_and_fixes_the_hash_seed(self):
        # PyCG does not reach its fixed point on rich, and set iteration order follows the hash seed
        from unittest import mock
        from gtb6 import builder as B
        seen = {}
        def fake(cmd, **kw):
            seen.update(cmd=cmd, env=kw.get('env'))
            return mock.Mock(returncode=1)
        with mock.patch.object(B.subprocess, 'run', fake):
            self.assertIsNone(B.run_pycg('/x/pkg', '/x/cg.json', python='py', max_iter=3))
        self.assertEqual(seen['cmd'][seen['cmd'].index('--max-iter') + 1], '3')
        self.assertEqual(seen['env']['PYTHONHASHSEED'], '0')
        with mock.patch.object(B.subprocess, 'run', fake):
            B.run_pycg('/x/pkg', '/x/cg.json', python='py')
        self.assertNotIn('--max-iter', seen['cmd'])

    def test_extract_definitions_maps_nodes_and_drops_locals(self):
        from gtb6 import builder as B
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / 'pkg'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'm.py').write_text(TOY_M)
            defs = B.extract_definitions(d, str(pkg))
        ids = {x['node_id'] for x in defs}
        self.assertEqual(ids, {'pkg.m.C', 'pkg.m.C.f', 'pkg.m.outer'})
        self.assertFalse(any('<locals>' in i for i in ids))
        self.assertEqual(next(x for x in defs if x['node_id'] == 'pkg.m.C.f')['path'], 'pkg/m.py')

    def test_to_succ_keeps_known_pairs_and_drops_self_loops(self):
        from gtb6 import builder as B
        cg = {'pkg.m.f': ['pkg.m.f', 'pkg.m.g', 'pycg.unknown'],
              'pycg.unknown': ['pkg.m.f'],
              'pkg.m.g': []}
        self.assertEqual(B.to_succ(cg, {'pkg.m.f', 'pkg.m.g'}),
                         {'pkg.m.f': ['pkg.m.g'], 'pkg.m.g': []})


if __name__ == '__main__':
    unittest.main()
