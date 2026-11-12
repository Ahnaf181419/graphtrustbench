"""Deterministic, group-aware held-out selection that never shares a base commit with development."""
import importlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
selection = importlib.import_module('select_heldout')


def task(i, repo, commit):
    return {'instance_id': i, 'repo': repo, 'base_commit': commit}


class TestSelectHeldOut(unittest.TestCase):
    def setUp(self):
        self.dev = [task('dev_a', 'o/one', 'c1'), task('dev_b', 'o/two', 'c2')]
        # shares a base commit with dev_a: must be excluded with it
        self.tasks = self.dev + [
            task('shares_with_dev', 'o/one', 'c1'),
            task('one_ok1', 'o/one', 'c3'), task('one_ok2', 'o/one', 'c4'),
            task('two_ok1', 'o/two', 'c5'), task('two_ok2', 'o/two', 'c5'),  # c5 collides internally
            task('three_ok', 'o/three', 'c6'),
        ]

    def test_dev_tasks_and_their_base_commit_sharers_are_excluded(self):
        chosen = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 6, 2027)
        ids = [t['instance_id'] for t in chosen]
        self.assertNotIn('dev_a', ids)
        self.assertNotIn('dev_b', ids)
        self.assertNotIn('shares_with_dev', ids)

    def test_never_selects_two_tasks_sharing_a_base_commit(self):
        chosen = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 6, 2027)
        commits = [f"{t['repo']}@{t['base_commit']}" for t in chosen]
        self.assertEqual(len(commits), len(set(commits)))

    def test_covers_every_repo_with_available_tasks(self):
        chosen = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 6, 2027)
        self.assertEqual({t['repo'] for t in chosen}, {'o/one', 'o/two', 'o/three'})

    def test_selection_is_deterministic(self):
        a = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 6, 2027)
        b = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 6, 2027)
        self.assertEqual(a, b)

    def test_fewer_available_than_limit_is_not_an_error(self):
        chosen = selection.select(self.tasks, {t['instance_id'] for t in self.dev}, 99, 2027)
        self.assertLessEqual(len(chosen), 5)

    def test_real_audit_yields_a_clean_grouped_split(self):
        ROOT = Path(__file__).resolve().parents[2]
        audit = json.loads((ROOT / 'paper/reports/competition_audit_full_2026-11-06.json').read_text())
        pilot = json.loads((ROOT / 'paper/data/pilot_2026-11-06.json').read_text())
        usable = [t for t in audit['tasks'] if t['schema_usable']]
        chosen = selection.select(usable, {t['instance_id'] for t in pilot['tasks']}, 24, 2027)
        commits = {f"{t['repo']}@{t['base_commit']}" for t in chosen}
        dev_commits = {f"{t['repo']}@{t['base_commit']}" for t in pilot['tasks']}
        self.assertEqual(len(chosen), 24)
        self.assertTrue(commits.isdisjoint(dev_commits))
        self.assertEqual(len(commits), 24)
        self.assertEqual({t['repo'] for t in chosen}, {'fastapi/fastapi', 'Textualize/rich', 'psf/requests'})


if __name__ == '__main__':
    unittest.main()
