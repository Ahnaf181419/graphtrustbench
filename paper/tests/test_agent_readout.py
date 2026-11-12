"""Paired contexts for a downstream agent read-out: identical issue, two retrieved file sets."""
import importlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
readout = importlib.import_module('agent_readout')

ROOT = Path(__file__).resolve().parents[2]

# Gold labels (per_task.jsonl, mapping.jsonl) are derived from competition patches and are not in the
# public release; these tests run where the labels were regenerated from the competition data.
needs_labels = unittest.skipUnless((ROOT / 'paper/results/2026-11-07_heldout_v3/per_task.jsonl').exists(),
                                   'gold labels are not released; regenerate them from the competition data')


class TestAgentReadout(unittest.TestCase):
    def test_only_tasks_whose_context_actually_changed_are_paired(self):
        rows = [
            {'instance_id': 'a', 'method': 'one_hop', 'corruption': 'clean', 'severity': 0, 'seed': 0,
             'ranked_files': ['x.py', 'y.py'], 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'a', 'method': 'one_hop', 'corruption': 'spurious', 'severity': 0.4, 'seed': 13,
             'ranked_files': ['x.py', 'z.py'], 'metrics': {'recall@5': 0.0}},
            {'instance_id': 'b', 'method': 'one_hop', 'corruption': 'clean', 'severity': 0, 'seed': 0,
             'ranked_files': ['p.py'], 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'b', 'method': 'one_hop', 'corruption': 'spurious', 'severity': 0.4, 'seed': 13,
             'ranked_files': ['p.py'], 'metrics': {'recall@5': 1.0}},
        ]
        pairs = readout.paired_contexts(rows, top_k=2)
        self.assertEqual([p['instance_id'] for p in pairs], ['a'])
        self.assertEqual(pairs[0]['clean_files'], ['x.py', 'y.py'])
        self.assertEqual(pairs[0]['corrupted_files'], ['x.py', 'z.py'])

    def test_prompt_is_fixed_contains_the_issue_and_truncated_files(self):
        prompt = readout.build_prompt('the parser drops comments', ['a.py', 'b.py'],
                                      {'a.py': 'x' * 5000, 'b.py': 'def f(): pass'}, per_file_chars=100)
        self.assertIn('the parser drops comments', prompt)
        self.assertIn('a.py', prompt)
        self.assertIn('b.py', prompt)
        self.assertLess(len(prompt), 1200)
        self.assertNotIn('x' * 500, prompt)
        self.assertIn('truncated', prompt.lower())

    def test_answer_parsing_accepts_bare_paths_and_rejects_free_text(self):
        self.assertEqual(readout.parse_answer('a/b/c.py', ['a/b/c.py', 'd.py']), 'a/b/c.py')
        self.assertEqual(readout.parse_answer('  d.py  ', ['a/b/c.py', 'd.py']), 'd.py')
        self.assertIsNone(readout.parse_answer('I think the bug is in d.py', ['a/b/c.py']))
        self.assertIsNone(readout.parse_answer('unknown.py', ['a/b/c.py', 'd.py']))

    def test_score_answers_reports_accuracy_discordance_and_faithfulness(self):
        from agent_readout import score_answers
        records = [
            {'instance_id': 'a', 'condition': 'clean', 'files': ['g.py', 'x.py'], 'gold_files': ['g.py']},
            {'instance_id': 'a', 'condition': 'corrupted', 'files': ['y.py', 'x.py'], 'gold_files': ['g.py']},
            {'instance_id': 'b', 'condition': 'clean', 'files': ['g.py'], 'gold_files': ['g.py']},
            {'instance_id': 'b', 'condition': 'corrupted', 'files': ['g.py'], 'gold_files': ['g.py']},
            # task c: the model finds the gold file but ignores the retriever's first choice
            {'instance_id': 'c', 'condition': 'clean', 'files': ['z.py', 'g.py'], 'gold_files': ['g.py']},
            {'instance_id': 'c', 'condition': 'corrupted', 'files': ['z.py', 'g.py'], 'gold_files': ['g.py']},
        ]
        answers = {('a', 'clean'): 'g.py', ('a', 'corrupted'): 'y.py',
                   ('b', 'clean'): 'g.py', ('b', 'corrupted'): 'g.py',
                   ('c', 'clean'): 'g.py', ('c', 'corrupted'): 'g.py'}
        summary = score_answers(records, answers)
        self.assertEqual(summary['arms']['clean']['accuracy'], 1.0)
        self.assertAlmostEqual(summary['arms']['corrupted']['accuracy'], 2 / 3)
        # faithfulness = the answer equals the retriever's own first choice (task c deviates)
        self.assertAlmostEqual(summary['arms']['clean']['faithfulness'], 2 / 3)
        self.assertAlmostEqual(summary['arms']['corrupted']['faithfulness'], 2 / 3)
        self.assertEqual(summary['discordant'], {'clean_only': 1, 'corrupted_only': 0,
                                                 'total': 1, 'sign_test_p': 1.0})

    def test_score_answers_counts_unanswerable_responses(self):
        from agent_readout import score_answers
        records = [{'instance_id': 'a', 'condition': 'clean', 'files': ['g.py'], 'gold_files': ['g.py']},
                   {'instance_id': 'a', 'condition': 'corrupted', 'files': ['g.py'], 'gold_files': ['g.py']}]
        summary = score_answers(records, {('a', 'clean'): 'g.py', ('a', 'corrupted'): 'I think g.py'})
        self.assertEqual(summary['arms']['corrupted']['unanswerable'], 1)
        self.assertIsNone(summary['arms']['corrupted']['accuracy'])

    @needs_labels
    def test_paired_contexts_from_the_committed_heldout_run(self):
        run = ROOT / 'paper/results/2026-11-07_heldout_v3'
        rows = [json.loads(line) for line in (run / 'per_task.jsonl').read_text().splitlines()]
        pairs = readout.paired_contexts(rows, top_k=5)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            self.assertNotEqual(pair['clean_files'], pair['corrupted_files'])
            self.assertEqual(len(pair['clean_files']), 5)
            self.assertEqual(len(pair['corrupted_files']), 5)


if __name__ == '__main__':
    unittest.main()
