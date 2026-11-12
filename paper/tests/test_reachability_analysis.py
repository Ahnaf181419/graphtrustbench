"""Reachability analysis: scope counting, delivery and the dose/sensitivity relation."""
import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
analysis = importlib.import_module('analyze_reachability')


class TestReachabilityAnalysis(unittest.TestCase):
    def setUp(self):
        self.edges = [
            {'source': 'a', 'target': 'x', 'type': 'calls', 'key': 0},
            {'source': 'a', 'target': 'y', 'type': 'calls', 'key': 0},
            {'source': 'b', 'target': 'z', 'type': 'calls', 'key': 0},
        ]

    def test_scope_counts_only_edges_leaving_the_seeds(self):
        self.assertEqual(analysis.scope_edge_count(self.edges, {'a'}), 2)
        self.assertEqual(analysis.scope_edge_count(self.edges, set()), 0)

    def test_delivered_count_detects_changed_or_removed_scope_edges(self):
        scope = [e for e in self.edges if e['source'] == 'a']
        self.assertEqual(analysis.delivered_count(scope, self.edges), 0)
        mutated = [e for e in self.edges if not (e['source'] == 'a' and e['target'] == 'x')]
        self.assertEqual(analysis.delivered_count(scope, mutated), 1)
        rewired = [dict(e) for e in self.edges]
        rewired[0]['target'] = 'z'
        self.assertEqual(analysis.delivered_count(scope, rewired), 1)

    def test_zero_delivery_probability_matches_the_binomial_prediction(self):
        self.assertAlmostEqual(analysis.zero_delivery_probability(0, 0.1), 1.0)
        self.assertAlmostEqual(analysis.zero_delivery_probability(8, 0.1), 0.9 ** 8)
        self.assertAlmostEqual(analysis.zero_delivery_probability(100, 0.4), 0.6 ** 100)

    def test_reachability_contingency_separates_delivered_from_undelivered(self):
        from analyze_reachability import reachability_contingency
        cells = [
            # corruption never reached the reachable neighbourhood
            {'delivered': 0, 'outcomes': {'recall5_changed': False, 'top5_set_changed': False,
                                           'top5_order_changed': False, 'full_ranking_changed': False}},
            # corruption reached it and changed the ranking but not the accuracy
            {'delivered': 4, 'outcomes': {'recall5_changed': False, 'top5_set_changed': True,
                                           'top5_order_changed': True, 'full_ranking_changed': True}},
            {'delivered': 2, 'outcomes': {'recall5_changed': True, 'top5_set_changed': False,
                                           'top5_order_changed': False, 'full_ranking_changed': False}},
        ]
        table = reachability_contingency(cells)
        self.assertEqual(table[0]['cells'], 1)
        self.assertEqual(table[0]['full_ranking_changed'], 0)
        self.assertEqual(table[1]['cells'], 2)
        self.assertEqual(table[1]['full_ranking_changed'], 1)
        self.assertEqual(table[1]['recall5_changed'], 1)
        self.assertEqual(table[1]['top5_order_changed'], 1)

    def test_separators_report_feature_means_for_changed_and_unchanged_tasks(self):
        from analyze_reachability import separators
        tasks = [{'instance_id': 'a', 'scope_edges': 10, 'distinct_seed_neighbours': 5, 'candidates': 100},
                 {'instance_id': 'b', 'scope_edges': 20, 'distinct_seed_neighbours': 9, 'candidates': 200},
                 {'instance_id': 'c', 'scope_edges': 30, 'distinct_seed_neighbours': 15, 'candidates': 300}]
        changed = {'a': 3, 'b': 0, 'c': 0}
        result = separators(tasks, changed)
        self.assertIn('scope_edges', result)
        self.assertAlmostEqual(result['scope_edges']['changed_mean'], 10.0)
        self.assertAlmostEqual(result['scope_edges']['unchanged_mean'], 25.0)
        self.assertIn('point_biserial', result['scope_edges'])

    def test_dose_bins_separate_landed_cells_from_unlanded_ones(self):
        cells = [{'delivered': 0, 'delivered_fraction': 0.0, 'changed': 0},
                 {'delivered': 2, 'delivered_fraction': 0.5, 'changed': 1},
                 {'delivered': 1, 'delivered_fraction': 0.1, 'changed': 0}]
        bins = analysis.delivered_dose_bins(cells)
        self.assertEqual(bins['at_or_above_threshold'], {'cells': 1, 'changed': 1})
        self.assertEqual(bins['below_threshold'], {'cells': 1, 'changed': 0})
        self.assertEqual(bins['no_corruption_delivered'], {'cells': 1, 'changed': 0})


if __name__ == '__main__':
    unittest.main()
