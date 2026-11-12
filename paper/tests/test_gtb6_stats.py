import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gtb6 import stats as S


class Stats(unittest.TestCase):
    def test_holm_is_step_down(self):
        self.assertEqual(S.holm([0.01, 0.04, 0.03]), [True, False, False])
        self.assertEqual(S.holm([0.001, 0.02, 0.04]), [True, True, True])

    def test_h3_reading_rule(self):
        self.assertEqual(S.h3_reading(True, 0, 0, 0, 0), 'corruption-specific')
        self.assertEqual(S.h3_reading(False, -0.2, -0.01, -0.15, -0.02), 'placebo-heavier')
        self.assertEqual(S.h3_reading(False, -0.057, 0.030, -0.049, 0.023), 'generic')   # registration-1 H3
        self.assertEqual(S.h3_reading(False, -0.2, 0.2, -0.15, 0.15), 'inconclusive')

    def test_auc_counts_ties_half(self):
        self.assertEqual(S.auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(S.auc([1, 1], [0, 1]), 0.5)

    def test_bootstrap_matches_registration_one(self):
        # same generator and draw shape as gemma_readout.analyze's boot()
        v = [0.1, -0.2, 0.3, 0.0, 0.5]
        b = np.random.default_rng(2026).integers(0, 5, (10_000, 5))
        lo = float(np.quantile(np.asarray(v)[b].mean(1), 0.025))
        self.assertAlmostEqual(S.ci(v)[1], lo)

    def test_equivalence_uses_the_90_interval(self):
        self.assertTrue(S.equivalence([0.01, -0.01] * 20, 0.05)['equivalent'])
        self.assertFalse(S.equivalence([0.3, -0.3] * 3, 0.05)['equivalent'])

    def test_one_sided_p_direction(self):
        self.assertEqual(S.one_sided_p([1.0, 2.0, 3.0]), 0.0)
        self.assertEqual(S.one_sided_p([1.0, 2.0, 3.0], direction='less'), 1.0)


if __name__ == '__main__':
    unittest.main()
