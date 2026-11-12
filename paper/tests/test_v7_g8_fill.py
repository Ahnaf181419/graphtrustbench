"""G8 fill: patch values from G7-shaped rows (labels needed for the cell coding)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
EXT = ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'


@unittest.skipUnless(EXT.exists(), 'labels not present')
class Fill(unittest.TestCase):
    def test_patch_values_when_every_cell_resolves_like_clean(self):
        import gemma_readout as g
        import v7_g8_fill as F
        clean, _, gold = g.rankings(g.load_rows(EXT))
        r1, r2, r3, _ = g.load_stages(F.V5)
        ev = [dict(tag='patch_comp', cond='C0', task=t, resolved=True) for t in clean]
        ev += [dict(tag='patch_comp', cond='X', task=r['task'], intervention=r['intervention'], seed=r['seed'],
                    resolved=True) for r in r2]
        ev += [dict(tag='patch_comp', cond='P', task=r['task'], intervention=r['intervention'], seed=r['seed'],
                    resolved=True) for r in r3]
        ev += [dict(tag='patch_floor_rep1', cond='C0r', task=t, resolved=True) for t in clean]
        v = F.patch_values(ev, r1, r2, clean, gold, h7_p=0.0436, rep_tasks=list(clean)[:3])
        self.assertEqual((v['RES_CLEAN'], v['H5_VERDICT'], v['H5_CELLS']), ('100.0', 'equivalent', 421))
        self.assertEqual((v['H6_VERDICT'], v['H7_VERDICT']), ('not established', 'not established'))
        self.assertEqual((v['PL_RES_DIFF'], v['PATCH_FLOOR_AGREE'], v['REP_RES_DIFF']),
                         ('+0.0 [+0.0, +0.0]', '100.0', '+0.0 [+0.0, +0.0]'))

    def test_floor_descriptives_apply_rate_and_informative_h5_tasks(self):
        # at a low resolve rate most H5 tasks are 0 in both contexts and narrow the interval by themselves
        import gemma_readout as g
        import v7_g8_fill as F
        clean, _, gold = g.rankings(g.load_rows(EXT))
        r1, r2, _, _ = g.load_stages(F.V5)
        first = sorted(clean)[0]
        x = [dict(tag='patch_comp', cond='X', task=r['task'], intervention=r['intervention'], seed=r['seed'],
                  resolved=False) for r in r2]
        none = [dict(tag='patch_comp', cond='C0', task=t, resolved=False, applied='git-p1' if t == first else None)
                for t in clean]
        v = F.patch_values(none + x, r1, r2, clean, gold, h7_p=0.0436)
        self.assertEqual((v['APPLY_CLEAN'], v['H5_INFORMATIVE']), (f'{100 / len(clean):.1f}', 0))
        every = [dict(e, resolved=True) for e in none]
        v = F.patch_values(every + x, r1, r2, clean, gold, h7_p=0.0436)
        self.assertEqual(v['H5_INFORMATIVE'], v['H5_TASKS'])

    def test_registered_cutoff_split_counts(self):
        import json
        import gemma_readout as g
        import v7_g8_fill as F
        meta = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'tasks.jsonl')}
        v = F.cutoff_split(g.load_rows(EXT), g.load_stages(F.V5), meta)
        self.assertEqual((v['SPLIT_PRE_N'], v['SPLIT_POST_N']), (28, 87))          # registered in prereg v2 §1
        self.assertEqual(v['SPLIT_POST_H3_READING'], 'generic')

    def test_settings_not_run_when_files_are_absent(self):
        import v7_g8_fill as F
        v = F.settings_values([])
        self.assertEqual({v[f'{k}_D'] for k in F.SET_ROWS}, {'not run'})


if __name__ == '__main__':
    unittest.main()
