"""Graph-name conventions must not guess between incompatible source definitions."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from localization_data import extract_candidates


class TestCandidateAliases(unittest.TestCase):
    def test_module_root_alias_requires_matching_source(self):
        sources = {'docs_src/examples/demo.py': 'def answer():\n    return 42\n'}
        node = {'id': 'examples.demo.answer', 'text': 'def answer():\n    return 42'}
        candidates, report = extract_candidates(sources, [node])
        self.assertIn('examples.demo.answer', candidates)
        self.assertEqual(candidates[node['id']]['path'], 'docs_src/examples/demo.py')
        self.assertEqual(report['alias_mapped_node_ids'], [node['id']])

    def test_flattened_function_scope_preserves_class_scope(self):
        sources = {'pkg/a.py': 'class C:\n    def outer(self):\n        def helper():\n            return 1\n        return helper()\n'}
        node = {'id': 'pkg.a.C.helper', 'text': 'def helper():\n    return 1'}
        candidates, _ = extract_candidates(sources, [node])
        self.assertIn(node['id'], candidates)
        self.assertEqual(candidates[node['id']]['spans'], [[3, 4]])

    def test_body_distinguishes_flattened_name_collision(self):
        sources = {'pkg/a.py': 'def first():\n    def helper():\n        return 1\ndef second():\n    def helper():\n        return 2\n'}
        node = {'id': 'pkg.a.helper', 'text': 'def helper():\n    return 2'}
        candidates, _ = extract_candidates(sources, [node])
        self.assertIn(node['id'], candidates)
        self.assertEqual(candidates[node['id']]['spans'], [[5, 6]])

    def test_identical_aliases_in_two_files_remain_unresolved(self):
        sources = {path: 'def f():\n    return 1\n' for path in ('one/pkg/a.py', 'two/pkg/a.py')}
        node = {'id': 'pkg.a.f', 'text': 'def f():\n    return 1'}
        candidates, report = extract_candidates(sources, [node])
        self.assertNotIn(node['id'], candidates)
        self.assertIn(node['id'], report.get('ambiguous_alias_node_ids', []))

    def test_unique_alias_with_wrong_body_is_not_guessed(self):
        node = {'id': 'pkg.a.f', 'text': 'def f():\n    return 2'}
        candidates, _ = extract_candidates({'other/pkg/a.py': 'def f():\n    return 1\n'}, [node])
        self.assertNotIn(node['id'], candidates)

    def test_exact_name_conflict_uses_source_matching_nested_alias(self):
        source = 'def helper():\n    return 1\n\ndef outer():\n    def helper():\n        return 2\n'
        node = {'id': 'pkg.a.helper', 'text': 'def helper():\n    return 2'}
        candidates, report = extract_candidates({'pkg/a.py': source}, [node])
        self.assertEqual(candidates[node['id']]['spans'], [[5, 6]])
        self.assertIn(node['id'], report['conflicting_exact_node_ids'])

    def test_exact_name_with_conflicting_body_and_no_alias_is_unresolved(self):
        node = {'id': 'pkg.a.f', 'text': 'def f():\n    return 2'}
        candidates, report = extract_candidates({'pkg/a.py': 'def f():\n    return 1\n'}, [node])
        self.assertNotIn(node['id'], candidates)
        self.assertIn(node['id'], report['conflicting_exact_node_ids'])

    def test_class_stub_can_bind_unique_module_alias(self):
        sources = {'docs_src/examples/a.py': 'class Settings(Base):\n    value = 7\n'}
        node = {'id': 'examples.a.Settings', 'text': 'class Settings(Base):\n    None'}
        candidates, _ = extract_candidates(sources, [node])
        self.assertIn(node['id'], candidates)
        self.assertEqual(candidates[node['id']]['path'], 'docs_src/examples/a.py')

    def test_nested_class_export_can_omit_outer_scopes(self):
        sources = {'pkg/a.py': 'class Outer:\n    def make(self):\n        class Inner:\n            def ping(self):\n                return 7\n'}
        nodes = [{'id': 'pkg.a.Inner', 'text': 'class Inner:\n    None'},
                 {'id': 'pkg.a.Inner.ping', 'text': 'def ping(self):\n    return 7'}]
        candidates, _ = extract_candidates(sources, nodes)
        self.assertEqual(set(candidates), {'pkg.a.Inner', 'pkg.a.Inner.ping'})


    def test_duplicate_unmapped_node_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate graph ID'):
            extract_candidates({}, [{'id': 'missing'}, {'id': 'missing'}])


if __name__ == '__main__':
    unittest.main()
