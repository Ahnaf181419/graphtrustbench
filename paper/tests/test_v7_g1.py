"""G1: the flip-risk score is fitted on exactly the registered cells and the committed freeze is reproducible."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
EXT = ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'
CELLS = ROOT / 'paper/reports/context_stability/cells_onehop.jsonl'
needs_labels = unittest.skipUnless(EXT.exists() and CELLS.exists(),
                                   'gold labels are not released; regenerate them from the competition data')


@needs_labels
class G1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import v7_g1_freeze as g1
        cls.g1 = g1
        cls.cells = g1.fit_cells(ROOT / 'paper/results/2026-11-10_gemma_readout', EXT, CELLS)

    def test_fit_set_is_the_530_non_forced_valid_cells(self):
        self.assertEqual(len(self.cells), 530)
        self.assertEqual(Counter(c['kind'] for c in self.cells), {'H2': 421, 'order-only': 77, 'recall-changed': 32})
        self.assertFalse(any(c['forced'] for c in self.cells))

    def test_committed_freeze_is_reproducible(self):
        from gtb6 import fliprisk as FR
        frozen = json.loads((ROOT / 'paper/prereg/fliprisk_frozen.json').read_text())
        self.assertEqual(frozen, json.loads(json.dumps(FR.fit(self.cells))))
        self.assertEqual(frozen['features'], FR.FEATURES)


if __name__ == '__main__':
    unittest.main()
