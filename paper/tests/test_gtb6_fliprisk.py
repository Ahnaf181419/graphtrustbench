import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gtb6 import fliprisk as FR


def synthetic(n=200):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] - 2 * x[:, 1] + 0.5 * x[:, 2] + rng.normal(size=n) > 0).astype(int)
    rows = []
    for i in range(n):
        f = dict.fromkeys(FR.FEATURES, 0.0)
        f.update(n_incoming=x[i, 0], one_minus_rbo=x[i, 1], log_k_eff=x[i, 2])
        rows.append(dict(f, flip=int(y[i]), task=f't{i % 20}'))
    return rows


class FlipRisk(unittest.TestCase):
    def test_features_for_unshown_and_non_file_choices(self):
        f = FR.features(['a', 'b', 'c', 'd', 'e', 'f'], ['a', 'b', 'c', 'd', 'f'], 'NONE', 0.2, 3)
        self.assertEqual((f['n_incoming'], f['c0_pos_corr'], f['incoming_above_c0'], f['c0_pos_shift']), (1, 6, 1, 0))
        self.assertAlmostEqual(f['log_min_incoming_rank'], math.log1p(6))
        self.assertAlmostEqual(f['log_k_eff'], math.log1p(3))

    def test_features_track_the_first_choice_moving(self):
        f = FR.features(['a', 'b', 'c', 'd', 'e', 'x'], ['x', 'a', 'b', 'c', 'd'], 'b', 0.3, 0)
        self.assertEqual((f['c0_pos_corr'], f['c0_pos_shift'], f['incoming_above_c0']), (3, 1, 1))

    def test_fit_matches_the_l2_optimum(self):
        rows = synthetic()
        m = FR.fit(rows)
        X = np.array([[r[k] for k in FR.FEATURES] for r in rows], float)
        z = (X - np.asarray(m['mean'])) / np.asarray(m['scale'])
        y = np.array([r['flip'] for r in rows], float)
        w = np.asarray(m['coef'])
        p = 1 / (1 + np.exp(-(z @ w + m['intercept'])))
        self.assertLess(np.max(np.abs(m['C'] * z.T @ (y - p) - w)), 1e-8)
        self.assertLess(abs(np.sum(y - p)), 1e-8)
        self.assertGreater(w[FR.FEATURES.index('n_incoming')], 0)
        self.assertLess(w[FR.FEATURES.index('one_minus_rbo')], 0)

    def test_fit_is_deterministic_and_json_serialisable(self):
        rows = synthetic()
        m = FR.fit(rows)
        self.assertEqual(m, FR.fit(rows))
        self.assertEqual(json.loads(json.dumps(m)), m)

    def test_score_orders_cells_by_risk(self):
        rows = synthetic()
        s = FR.score(FR.fit(rows), rows)
        self.assertEqual(len(s), len(rows))
        flips = np.array([r['flip'] for r in rows]) == 1
        self.assertGreater(s[flips].mean(), s[~flips].mean())

    def test_coefficient_intervals_cover_every_feature(self):
        ci = FR.coef_intervals(synthetic(), n=50)
        self.assertEqual(set(ci), set(FR.FEATURES))
        self.assertTrue(all(lo <= hi for lo, hi in ci.values()))


if __name__ == '__main__':
    unittest.main()
