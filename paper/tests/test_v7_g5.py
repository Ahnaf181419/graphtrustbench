"""Plan 2 Task 3 (G5): build, select and freeze the post-cutoff tasks."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
sys.path.insert(0, str(ROOT / 'paper/tests'))
from test_gtb6_repo_ops import BUGGY, CODE_FIX, TEST_PATCH
from test_v7_g4 import IMAGE, needs_docker_image, toy_repo


def detail(pr, merged, issue_created, repo='psf/requests'):
    return dict(repo=repo, pr=pr, issue=pr + 1000, merged_at=merged, issue_created=issue_created,
                base_sha='b', problem_statement='add returns the wrong sum', patch='')


class G5(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @needs_docker_image
    def test_build_task_on_a_toy_repo(self):
        import v7_g4_trace as G4
        import v7_g5_build as G
        base = toy_repo(self.dir)
        d = detail(7, '2025-05-01T00:00:00Z', '2025-04-01T00:00:00Z')
        d['patch'] = CODE_FIX + TEST_PATCH
        cg = lambda pkg: {'pkg.x.add': ['pkg.x.other', 'builtins.print'], 'pkg.x.other': []}
        task, why = G.build_task(d, self.dir, base, G4.docker_cmd(IMAGE, python='python'), cg,
                                 setup=lambda repo, sha: (True, ''), pkg_name='pkg')
        self.assertEqual(why, 'ok')
        self.assertEqual(task['task_id'], 'requests-7')
        self.assertEqual(task['gold_files'], ['pkg/x.py'])
        self.assertEqual(task['f2p'], ['tests/test_x.py::test_add'])
        self.assertEqual(task['succ'], {'pkg.x.add': ['pkg.x.other'], 'pkg.x.other': []})
        self.assertEqual(set(task['bm25']), {'pkg.x.add', 'pkg.x.other'})
        self.assertEqual(task['node2file'], {'pkg.x.add': 'pkg/x.py', 'pkg.x.other': 'pkg/x.py'})
        self.assertEqual(task['files'], {'pkg/x.py': BUGGY})
        self.assertEqual(Path(f'{self.dir}/pkg/x.py').read_text(), BUGGY)

    def test_build_task_rejects_a_patch_without_a_test_change(self):
        import v7_g5_build as G
        d = detail(8, '2025-05-01T00:00:00Z', '2025-04-01T00:00:00Z')
        d['patch'] = CODE_FIX
        self.assertEqual(G.build_task(d, self.dir, 'HEAD', None, None, setup=None, pkg_name='pkg'),
                         (None, 'needs a code and a test change'))

    def test_fallback_order(self):
        import v7_g5_build as G
        details = [detail(i, f'2025-06-{i:02d}T00:00:00Z', '2025-04-01T00:00:00Z' if i < 4 else '2024-12-01T00:00:00Z')
                   for i in range(1, 60)]
        seen = []
        def outcome(d):
            seen.append(d['pr'])
            return dict(task_id=f"requests-{d['pr']}"), 'ok'
        tasks, reasons, rule = G.select(details, outcome)
        self.assertEqual(rule, 'fix after the cutoff (fallback)')
        self.assertEqual(seen[:3], [1, 2, 3])                     # the strict pool is tried first
        self.assertEqual([t['task_id'] for t in tasks][:5], [f'requests-{i}' for i in range(1, 6)])
        self.assertEqual(len(tasks), 59)
        self.assertEqual(reasons['strict'], {'ok': 3})
        strict = [detail(i, f'2025-06-{i % 28 + 1:02d}T{i % 24:02d}:00:00Z', '2025-04-01T00:00:00Z') for i in range(1, 130)]
        tasks, reasons, rule = G.select(strict, lambda d: (dict(task_id=d['pr']), 'ok'))
        self.assertEqual((rule, len(tasks)), ('issue and fix after the cutoff', 100))
        self.assertEqual([t['task_id'] for t in tasks], [d['pr'] for d in sorted(strict, key=lambda d: d['merged_at'])][:100])

    def test_manifest_refuses_overwrite(self):
        import v7_g5_build as G
        p = Path(self.dir) / 'postcutoff_tasks.json'
        task = dict(task_id='requests-7', repo='requests', pr=7, issue=1007, merged_at='m', issue_created='i',
                    base_sha='b', gold_files=['x.py'], f2p=['t::a'], p2p=[], succ={}, files={})
        G.write_manifest(p, 'issue and fix after the cutoff', {'strict': {'ok': 1}}, [task])
        frozen = p.read_text()
        self.assertEqual(json.loads(frozen)['tasks'][0]['task_id'], 'requests-7')
        self.assertNotIn('succ', json.loads(frozen)['tasks'][0])
        with self.assertRaises(SystemExit) as cm:
            G.write_manifest(p, 'other', {}, [])
        self.assertNotEqual(cm.exception.code, 0)
        self.assertEqual(p.read_text(), frozen)

    def test_fetch_retries_transient_errors_but_keeps_missing_issues(self):
        # Textualize/rich#3935 got "gh: HTTP 500" once and was cached as not-a-task
        from unittest import mock
        import v7_g5_build as G
        p = Path(self.dir) / 'details_cache.json'
        cands = [dict(repo='r/x', pr=1, issue=2), dict(repo='r/x', pr=3, issue=4)]
        p.write_text(json.dumps(dict(candidates=cands, details={'r/x#1': dict(error='gh: HTTP 500\n'),
                                                                 'r/x#3': dict(error='gh: Not Found (HTTP 404)\n')})))
        got = dict(repo='r/x', pr=1, issue=2, merged_at='m', issue_created='i')
        calls = []
        with mock.patch.object(G.B, 'pr_details', lambda repo, pr, issue: calls.append(pr) or got), \
             mock.patch.object(G.time, 'sleep', lambda s: None):
            details, cache = G.fetch(p)
        self.assertEqual(calls, [1])
        self.assertEqual(details, [got])
        self.assertIn('error', json.loads(p.read_text())['details']['r/x#3'])

    def test_postcutoff_cells_k_eff_and_rbo(self):
        import v7_g5_build as G
        from context_stability import admitted
        nodes = [f'm.f{i}' for i in range(12)]
        succ = {n: [m for m in nodes if m != n][:4] for n in nodes}
        task = dict(task_id='rich-1', bm25={n: float(12 - i) for i, n in enumerate(nodes)},
                    node2file={n: f'f{i % 6}.py' for i, n in enumerate(nodes)}, succ=succ)
        hop = dict(seed_count=5, edge_budget=100, restart=0.15)
        rankings, cells = G.cells_for(task, hop)
        self.assertEqual(len(cells), 6)
        self.assertEqual(set(rankings), {('clean', 0.0, 0)} | {(iv, 0.4, sd) for iv in ('missing', 'spurious') for sd in (13, 37, 101)})
        self.assertEqual(sorted(rankings[('clean', 0.0, 0)]), [f'f{i}.py' for i in range(6)])
        for c in cells:
            self.assertEqual(set(c), {'task', 'intervention', 'fraction', 'seed', 'k_eff', 'one_minus_rbo'})
            self.assertTrue(0 <= c['one_minus_rbo'] <= 1)
            self.assertGreaterEqual(c['k_eff'], 0)
        edges = G.succ_edges(succ)
        seeds = sorted(task['bm25'], key=lambda n: (-task['bm25'][n], n))[:5]
        self.assertEqual(sum(len(a) for a in admitted(task['bm25'], edges, seeds, 100)), 20)


if __name__ == '__main__':
    unittest.main()
