"""Regression tests for artifact provenance and fail-closed experiment execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

# Gold labels (per_task.jsonl, mapping.jsonl) are derived from competition patches and are not in the
# public release; these tests run where the labels were regenerated from the competition data.
needs_labels = unittest.skipUnless((ROOT / 'paper/results/2026-11-07_heldout_v3/per_task.jsonl').exists(),
                                   'gold labels are not released; regenerate them from the competition data')
sys.path.insert(0, str(ROOT / 'paper/scripts'))


class TestPilotIntegrity(unittest.TestCase):
    def test_missing_input_is_recorded_by_both_runners(self):
        for script in ('prepare_pilot.py', 'run_localization_pilot.py'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = ['--data-root', str(root), '--pilot', str(root/'missing'),
                        '--audit', str(root/'missing'), '--output-dir', str(root/'run')]
                args += (['--upstream-trees', str(root/'missing')] if script == 'prepare_pilot.py'
                         else ['--mapping-dir', str(root/'missing')])
                result = subprocess.run([sys.executable, str(ROOT/'paper/scripts'/script), *args],
                                        capture_output=True, text=True, cwd=ROOT)
                self.assertNotEqual(result.returncode, 0)
                manifest = json.loads((root/'run/manifest.json').read_text())
                self.assertEqual(manifest['status'], 'failed')
                self.assertIn('FileNotFoundError', manifest['error'])

    def test_pilot_selection_rejects_empty_duplicate_or_mismatched_assets(self):
        from pilot_integrity import validate_selection
        pilot = json.loads((ROOT/'paper/data/pilot_2026-11-06.json').read_text())
        audit = json.loads((ROOT/'paper/reports/competition_audit_full_2026-11-06.json').read_text())
        validate_selection(pilot, audit)
        empty = copy.deepcopy(pilot); empty['tasks'] = []
        duplicate = copy.deepcopy(pilot); duplicate['tasks'].append(duplicate['tasks'][0])
        swapped = copy.deepcopy(pilot); swapped['tasks'][0]['graph_path'] = swapped['tasks'][1]['graph_path']
        wrong_repo = copy.deepcopy(pilot); wrong_repo['tasks'][0]['repo'] = 'other/repo'
        for bad in (empty, duplicate, swapped, wrong_repo):
            with self.subTest(tasks=len(bad['tasks'])), self.assertRaises(ValueError):
                validate_selection(bad, audit)

    def test_config_rejects_duplicate_conditions_and_invalid_budgets(self):
        from pilot_integrity import conditions
        config = json.loads((ROOT/'paper/config/pilot.json').read_text())
        self.assertEqual(len(conditions(config)), 10)
        changes = [('seeds', [13, 13]), ('seeds', []), ('seeds', [True]),
                   ('severities', [0.1, 0.1]), ('severities', [0]),
                   ('severities', [float('nan')]), ('bm25', {'k1': -1, 'b': 0.75}),
                   ('one_hop', {'seed_count': 5, 'edge_budget': -1, 'restart': 0.15})]
        for key, value in changes:
            bad = copy.deepcopy(config); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                conditions(bad)

    def test_protocol_v3_adds_spurious_conditions_and_keeps_legacy_configs(self):
        from pilot_integrity import conditions
        legacy = json.loads((ROOT/'paper/config/pilot_v2.json').read_text())
        v3 = json.loads((ROOT/'paper/config/pilot_v3.json').read_text())
        self.assertEqual(conditions(legacy),
                         [('clean', 0.0, 0)] + [('missing', f, s) for f in (0.1, 0.2, 0.4) for s in (13, 37, 101)])
        self.assertEqual(conditions(v3),
                         [('clean', 0.0, 0)] + [('missing', f, s) for f in (0.1, 0.2, 0.4) for s in (13, 37, 101)]
                         + [('spurious', f, s) for f in (0.1, 0.2, 0.4) for s in (13, 37, 101)])
        for corruption in ('edge_addition', [], ['missing', 'missing'], ['missing', 'bogus'], 7):
            bad = copy.deepcopy(v3); bad['corruption'] = corruption
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                conditions(bad)

    def test_stale_extractor_requires_new_mapping(self):
        from pilot_integrity import validate_mapping_sources
        from audit_competition_data import digest
        sources = {name: digest(ROOT/'paper/scripts'/name)
                   for name in ('localization_data.py', 'localization_evaluation.py')}
        validate_mapping_sources({'source_hashes': sources}, sources)
        stale = dict(sources); stale['localization_data.py'] = 'obsolete'
        with self.assertRaisesRegex(ValueError, 'Rebuild mapping'):
            validate_mapping_sources({'source_hashes': stale}, sources)

    @needs_labels
    def test_renderer_rejects_unrelated_committed_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, str(ROOT/'paper/scripts/render_pilot_report.py'),
                '--audit', 'paper/reports/competition_audit_full_2026-11-06.json',
                '--run-dir', 'paper/results/2026-11-06_localization_final',
                '--mapping-dir', 'paper/results/2026-11-06_mapping_pilot_v2',
                '--audit-output', str(root/'audit.md'), '--results-output', str(root/'results.md')],
                capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Mapping', result.stderr)
            self.assertFalse((root/'audit.md').exists())
            self.assertFalse((root/'results.md').exists())

    @needs_labels
    def test_report_checks_prediction_labels_and_complete_condition_matrix(self):
        from render_pilot_report import validate_observations
        run = ROOT/'paper/results/2026-11-06_localization_final'
        load_lines = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
        manifest = json.loads((run/'manifest.json').read_text())
        pilot = json.loads((ROOT/'paper/data/pilot_2026-11-06.json').read_text())
        predictions = load_lines(run/'predictions.jsonl')
        rows = load_lines(run/'per_task.jsonl')
        mappings = load_lines(ROOT/'paper/results/2026-11-06_mapping_final/mapping.jsonl')
        validate_observations(manifest, pilot, predictions, rows, mappings)
        wrong_ranking = copy.deepcopy(predictions)
        wrong_ranking[0]['ranked_files'].reverse()
        wrong_gold = copy.deepcopy(mappings)
        wrong_gold[0]['mapping']['gold_files'] = ['wrong.py']
        wrong_query = copy.deepcopy(predictions)
        wrong_query[1]['query_sha256'] = 'different'
        cases = [(wrong_ranking, rows, mappings), (predictions, rows, wrong_gold),
                 (wrong_query, rows, mappings), (predictions[:-1], rows[:-1], mappings),
                 (predictions + predictions[:1], rows + rows[:1], mappings)]
        for predicted, evaluated, labels in cases:
            with self.subTest(count=len(predicted)), self.assertRaises(ValueError):
                validate_observations(manifest, pilot, predicted, evaluated, labels)

    def test_degradation_cells_are_counted_per_cell(self):
        from render_pilot_report import degradation_cells
        rows = [
            {'instance_id': 'a', 'method': 'graph', 'corruption': 'clean', 'severity': 0, 'seed': 0, 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'a', 'method': 'graph', 'corruption': 'missing', 'severity': 0.4, 'seed': 13, 'metrics': {'recall@5': 0.5}},
            {'instance_id': 'a', 'method': 'graph', 'corruption': 'missing', 'severity': 0.4, 'seed': 37, 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'b', 'method': 'graph', 'corruption': 'missing', 'severity': 0, 'seed': 0, 'metrics': {'recall@5': 0.0}},
            {'instance_id': 'b', 'method': 'graph', 'corruption': 'missing', 'severity': 0.4, 'seed': 13, 'metrics': {'recall@5': 0.0}},
        ]
        # Only the one changed cell is reported; identical-to-clean cells are counted, not listed.
        self.assertEqual(degradation_cells(rows),
                         {('graph', 'missing'): {'cells': 3, 'changed': [('a', 13, 0.4, -0.5)]}})

    def test_footprint_sentence_derives_from_recorded_costs(self):
        from render_pilot_report import footprint_text
        rows = [
            {'method': 'one_hop', 'severity': 0, 'cost': {'edges_incident_to_seeds': 5, 'edges_scanned': 100, 'distinct_seed_neighbors': 4}},
            {'method': 'one_hop', 'severity': 0, 'cost': {'edges_incident_to_seeds': 25, 'edges_scanned': 100, 'distinct_seed_neighbors': 20}},
            {'method': 'bm25', 'severity': 0, 'cost': {}},
        ]
        text = footprint_text(rows)
        self.assertIn('5.0%–25.0%', text)
        self.assertIn('15.0%', text)
        # Runs recorded before instrumentation simply omit the sentence.
        self.assertIsNone(footprint_text([{'method': 'one_hop', 'severity': 0, 'cost': {}}]))

    def test_cell_listing_rounds_severity_percentages(self):
        from render_pilot_report import cell_listing
        cells = {('one_hop', 'spurious'): {'cells': 108, 'changed': [('t', 13, 0.29, -0.5), ('u', 37, 0.4, -1.0)]},
                 ('bm25', 'missing'): {'cells': 108, 'changed': []}}
        text = cell_listing(cells)
        self.assertIn('29% spurious', text)
        self.assertIn('40% spurious', text)
        self.assertIn('bm25/missing changed 0 of 108', text)
        self.assertNotIn('28%', text)

    def test_bootstrap_intervals_are_stable_when_a_new_row_type_is_added(self):
        """A per-group RNG keeps existing intervals unchanged when other rows appear."""
        from render_pilot_report import bootstrap_deltas
        def row(task, kind, sev, value):
            return {'instance_id': task, 'method': 'g', 'corruption': kind, 'severity': sev,
                    'seed': 13, 'metrics': {'recall@5': value}}
        base = []
        for index in range(12):
            base.append(row(f't{index}', 'clean', 0, 0.5))
            base.append(row(f't{index}', 'missing', 0.4, 0.5 - 0.01 * index))
        before = bootstrap_deltas(base)
        self.assertLess(before[('g', 'missing', 0.4)]['low'],
                        before[('g', 'missing', 0.4)]['high'], 'the interval must be non-degenerate')
        # A new group that sorts BEFORE the existing one shifts a shared RNG stream.
        extended = base + [row('x', 'clean', 0, 0.5), row('x', 'aaa_early', 0.4, 0.25)]
        after = bootstrap_deltas(extended)
        self.assertEqual(before[('g', 'missing', 0.4)]['low'], after[('g', 'missing', 0.4)]['low'])
        self.assertEqual(before[('g', 'missing', 0.4)]['high'], after[('g', 'missing', 0.4)]['high'])

    def test_bootstrap_intervals_and_sign_test_on_synthetic_rows(self):
        from render_pilot_report import interval_text, sign_test
        def row(task, kind, sev, delta):
            return {'instance_id': task, 'method': 'g', 'corruption': kind, 'severity': sev,
                    'seed': 13, 'metrics': {'recall@5': 0.5 + delta}}
        clean = [row(t, 'clean', 0, 0.0) for t in ('a', 'b', 'c', 'd')]
        # every task degrades by exactly 0.1 -> the CI must collapse onto -0.1
        rows = clean + [row(t, 'missing', 0.4, -0.1) for t in ('a', 'b', 'c', 'd')]
        text = interval_text(rows)
        self.assertIn('missing', text)
        self.assertIn('-0.100', text)
        # sign test: 4 of 4 negative, zero ties -> exact binomial p = 2*0.5^4 = 0.125
        self.assertAlmostEqual(sign_test([-0.1] * 4), 0.125)
        # one-sided degradation is not significant when tasks split evenly
        self.assertGreater(sign_test([-0.1, 0.1, -0.1, 0.1]), 0.99)
        # no nonzero deltas -> no evidence either way
        self.assertIsNone(sign_test([0.0, 0.0]))

    @needs_labels
    def test_generated_report_discloses_that_one_cell_drives_the_40pct_aggregate(self):
        from render_pilot_report import degradation_cells
        run = ROOT / 'paper/results/2026-11-07_localization_v2'
        rows = [json.loads(line) for line in (run / 'per_task.jsonl').read_text().splitlines()]
        cells = degradation_cells(rows)
        self.assertEqual(cells[('bm25', 'missing')], {'cells': 108, 'changed': []})
        self.assertEqual(cells[('one_hop', 'missing')], {'cells': 108,
                                                         'changed': [('rich_4075', 13, 0.4, -1.0)]})

    def test_degradation_cells_separate_corruption_kinds(self):
        from render_pilot_report import degradation_cells
        rows = [
            {'instance_id': 'a', 'method': 'g', 'corruption': 'clean', 'severity': 0, 'seed': 0, 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'a', 'method': 'g', 'corruption': 'missing', 'severity': 0.4, 'seed': 13, 'metrics': {'recall@5': 1.0}},
            {'instance_id': 'a', 'method': 'g', 'corruption': 'spurious', 'severity': 0.4, 'seed': 13, 'metrics': {'recall@5': 0.5}},
        ]
        self.assertEqual(degradation_cells(rows),
                         {('g', 'missing'): {'cells': 1, 'changed': []},
                          ('g', 'spurious'): {'cells': 1, 'changed': [('a', 13, 0.4, -0.5)]}})

    def test_protocol_v4_adds_targeted_kinds_and_keeps_every_legacy_config(self):
        from pilot_integrity import conditions
        v4 = json.loads((ROOT/'paper/config/pilot_v4.json').read_text())
        expected = [('clean', 0.0, 0)]
        for kind in ('missing', 'spurious', 'missing_targeted', 'spurious_targeted'):
            expected += [(kind, f, s) for f in (0.1, 0.2, 0.4) for s in (13, 37, 101)]
        self.assertEqual(conditions(v4), expected)
        for corruption in (['missing', 'targeted'], ['missing', 'spurious', 'missing'], 'bogus_kind', 7):
            bad = copy.deepcopy(v4); bad['corruption'] = corruption
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                conditions(bad)
        # A single targeted kind is a valid protocol, as list or legacy-style string.
        only_targeted = copy.deepcopy(v4); only_targeted['corruption'] = ['missing_targeted']
        self.assertEqual(len(conditions(only_targeted)), 10)
        shorthand = copy.deepcopy(v4); shorthand['corruption'] = 'missing_targeted'
        self.assertEqual(conditions(shorthand), conditions(only_targeted))

    def test_a_config_cannot_gain_an_intervention_without_bumping_its_version(self):
        from pilot_integrity import conditions
        v3 = json.loads((ROOT/'paper/config/pilot_v3.json').read_text())
        smuggle = copy.deepcopy(v3); smuggle['corruption'] = ['missing', 'spurious', 'missing_targeted']
        with self.assertRaises(ValueError):
            conditions(smuggle)
        v4 = json.loads((ROOT/'paper/config/pilot_v4.json').read_text())
        self.assertEqual(len(conditions(v4)), 37)

    @needs_labels
    def test_aggregate_reproduces_the_stored_v2_summary_exactly(self):
        from run_localization_pilot import aggregate
        run = ROOT / 'paper/results/2026-11-07_localization_v2'
        rows = [json.loads(line) for line in (run / 'per_task.jsonl').read_text().splitlines()]
        self.assertEqual(aggregate(rows), json.loads((run / 'summary.json').read_text()))

    def test_aggregate_never_pools_corruption_kinds(self):
        from run_localization_pilot import aggregate
        def row(kind, severity, task, value):
            return {'method': 'one_hop', 'corruption': kind, 'severity': severity, 'instance_id': task,
                    'repo': 'o/r', 'seed': 13, 'metrics': {'recall@5': value}}
        rows = [row('clean', 0, 'a', 1.0), row('missing', 0.4, 'a', 1.0), row('spurious', 0.4, 'a', 0.0)]
        summaries = aggregate(rows)['summaries']
        means = {(s.get('corruption'), s['severity']): s['mean']['recall@5']
                 for s in summaries if s['repo'] == 'all'}
        self.assertEqual(means[('missing', 0.4)], 1.0)
        self.assertEqual(means[('spurious', 0.4)], 0.0)


if __name__ == '__main__':
    unittest.main()
