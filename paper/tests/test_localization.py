"""Synthetic ground-truth, source mapping and leakage tests."""
import importlib
import json
from pathlib import Path
import sys
import io
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
data = importlib.import_module('localization_data')
evaluation = importlib.import_module('localization_evaluation')


class TestLocalizationData(unittest.TestCase):
    def test_snapshot_tree_matches_native_git_hash(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'repo.tgz'
            with tarfile.open(path, 'w:gz') as archive:
                for name, payload in [('a.py', b'pass\n'), ('.git/config', b'ignored\n')]:
                    entry = tarfile.TarInfo('./' + name)
                    entry.mode = 0o644
                    entry.size = len(payload)
                    archive.addfile(entry, io.BytesIO(payload))
            blob = subprocess.check_output(['git','hash-object','--stdin'],input=b'pass\n').strip()
            expected = subprocess.check_output(['git','hash-object','-t','tree','--stdin'],input=b'100644 a.py\0'+bytes.fromhex(blob.decode())).decode().strip()
            self.assertEqual(data.snapshot_tree(path)['tree_sha'], expected)
            self.assertEqual(data.snapshot_tree(path)['files'], 1)

    def test_allowlist_never_exposes_patches_or_test_labels(self):
        task = dict(instance_id='pkg_1', repo='o/pkg', base_commit='a'*40,
                    problem_statement='fix query', patch='SECRET_GOLD', test_patch='SECRET_TEST', hints_text='SECRET_HINT')
        self.assertEqual(data.retrieval_task(task), {k: task[k] for k in ('instance_id','repo','base_commit','problem_statement')})
        self.assertNotIn('SECRET', json.dumps(data.retrieval_task(task)))

    def test_candidates_map_src_layout_and_nested_scopes(self):
        sources = {'src/pkg/mod.py': 'class A:\n    def method(self):\n        return 1\n\ndef other():\n    return 2\n'}
        nodes = [{'id':'pkg.mod.A.method'}, {'id':'pkg.mod.other'}, {'id':'pkg.missing'}]
        candidates, report = data.extract_candidates(sources, nodes)
        self.assertEqual(candidates['pkg.mod.A.method']['path'], 'src/pkg/mod.py')
        self.assertEqual(candidates['pkg.mod.A.method']['spans'], [[2,3]])
        self.assertEqual(report['missing_node_ids'], ['pkg.missing'])
        self.assertEqual(len(candidates), 2)

    def test_duplicate_symbols_preserve_ambiguity(self):
        sources = {'pkg/a.py': 'def f():\n    return 1\ndef f():\n    return 2\n'}
        candidates, report = data.extract_candidates(sources, [{'id':'pkg.a.f'}])
        self.assertEqual(candidates['pkg.a.f']['spans'], [[1,2],[3,4]])
        self.assertEqual(report['ambiguous_node_ids'], ['pkg.a.f'])


class TestEvaluation(unittest.TestCase):
    def test_independent_insertion_is_not_lost_in_replacement_hunk(self):
        patch = '--- a/a.py\n+++ b/a.py\n@@ -1,6 +1,7 @@\n def a():\n-    return 1\n+    return 2\n \n def b():\n+    x = 3\n     return 4\n \n'
        sources = {'a.py': 'def a():\n    return 1\n\ndef b():\n    return 4\n\n'}
        candidates = {'a': {'path': 'a.py', 'spans': [[1, 2]]}, 'b': {'path': 'a.py', 'spans': [[4, 5]]}}
        self.assertEqual(evaluation.map_gold(patch, sources, candidates)['gold_nodes'], ['a', 'b'])

    def test_mixed_text_and_unsupported_patch_sections_are_rejected(self):
        text = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n'
        for extra in ('diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n',
                      'diff --git a/image.bin b/image.bin\nBinary files a/image.bin and b/image.bin differ\n'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                evaluation.parse_patch(text + extra)

    def test_rename_with_text_hunks_uses_pre_fix_path(self):
        patch = 'diff --git a/old.py b/new.py\nrename from old.py\nrename to new.py\n--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-pass\n+print(1)\n'
        self.assertEqual(evaluation.map_gold(patch, {'old.py': 'pass\n'}, {})['gold_files'], ['old.py'])

    def test_diff_preimage_mapping_and_insertion(self):
        patch = '--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,3 +1,4 @@\n def f():\n-    return 1\n+    x = 2\n+    return x\n \n'
        sources = {'pkg/a.py': 'def f():\n    return 1\n\n'}
        candidates, _ = data.extract_candidates(sources, [{'id':'pkg.a.f'}])
        labels = evaluation.map_gold(patch, sources, candidates)
        self.assertEqual(labels['gold_files'], ['pkg/a.py'])
        self.assertEqual(labels['gold_nodes'], ['pkg.a.f'])
        self.assertEqual(labels['status'], 'precise')
        self.assertEqual(labels['preimage_errors'], [])

    def test_mismatched_preimage_blocks(self):
        patch = '--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-before\n+after\n'
        labels = evaluation.map_gold(patch, {'a.py':'wrong\n'}, {})
        self.assertEqual(labels['status'], 'failed')
        self.assertTrue(labels['preimage_errors'])

    def test_new_file_is_not_mapped_to_existing_symbol(self):
        patch = '--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+pass\n'
        labels = evaluation.map_gold(patch, {}, {})
        self.assertEqual(labels['gold_files'], ['new.py'])
        self.assertEqual(labels['unretrievable_files'], ['new.py'])
        self.assertEqual(labels['status'], 'file_only')

    def test_dataset_new_file_without_dev_null_header(self):
        patch = '--- a/new.py\n+++ b/new.py\n@@ -0,0 +1 @@\n+pass\n'
        labels = evaluation.map_gold(patch, {}, {})
        self.assertEqual(labels['preimage_errors'], [])
        self.assertTrue(labels['files'][0]['inferred_new_file'])
        self.assertEqual(labels['unretrievable_files'], ['new.py'])

    def test_deleted_file_uses_old_path(self):
        patch = '--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-pass\n'
        self.assertEqual(evaluation.map_gold(patch, {'old.py':'pass\n'}, {})['gold_files'], ['old.py'])

    def test_null_sides_cannot_contain_hunk_lines(self):
        for patch in ('--- /dev/null\n+++ b/new.py\n@@ -1 +1 @@\n-fake_old\n+new\n',
                      '--- a/old.py\n+++ /dev/null\n@@ -1 +1 @@\n-old\n+fake_new\n'):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                evaluation.map_gold(patch, {'old.py': 'old\n'}, {})

    def test_hunk_counts_and_unsafe_paths_are_rejected(self):
        for patch in [
            '--- a/x.py\n+++ b/x.py\n@@ -1,2 +1 @@\n-a\n+b\n',
            '--- a/../../outside\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n',
            'not a unified diff',
        ]:
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError): evaluation.parse_patch(patch)

    def test_metrics_use_multi_gold_denominator(self):
        metrics = evaluation.metrics(['wrong','b','a'], {'a','b'})
        self.assertEqual(metrics['recall@1'],0)
        self.assertEqual(metrics['recall@5'],1)
        self.assertEqual(metrics['success@1'],0)
        self.assertEqual(metrics['success@5'],1)
        self.assertEqual(metrics['mrr'],0.5)
        with self.assertRaises(ValueError): evaluation.metrics(['a','a'], {'a'})
        with self.assertRaises(ValueError): evaluation.metrics(['a'], set())


if __name__ == '__main__': unittest.main()
