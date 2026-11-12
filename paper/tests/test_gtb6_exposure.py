import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gtb6 import exposure as EX


class Exposure(unittest.TestCase):
    def test_no_read_edges_means_no_dose(self):
        self.assertEqual(EX.dose_distribution(100, 0, 0.4)['p_zero'], 1.0)

    def test_every_edge_read_means_the_dose_is_the_draw(self):
        self.assertEqual(EX.dose_distribution(10, 10, 0.4), dict(n=4, mean=4.0, p_zero=0.0))

    def test_mean_dose_is_f_times_read_edges(self):
        self.assertAlmostEqual(EX.dose_distribution(1000, 30, 0.2)['mean'], 200 * 30 / 1000)

    def test_predicted_share_averages_cells(self):
        self.assertAlmostEqual(EX.predicted_unchanged_share([(100, 0, 0.4), (10, 10, 0.4)]), 0.5)


if __name__ == '__main__':
    unittest.main()
