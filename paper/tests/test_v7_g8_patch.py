"""G8 for the patch readout: H5 (equivalence), H6 (Recall@5 drop), Holm over {H6, H7}."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))


def world(n_tasks, h2_resolved, drop_resolved):
    """n tasks; each has C0 resolved, one H2-type cell, one Recall@5-drop cell and one order-only cell."""
    clean, gold, r1, r2, ev = {}, {}, [], [], []
    for i in range(n_tasks):
        t = f't{i}'
        clean[t] = ['g.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py', 'h.py']
        gold[t] = {'g.py'}
        r1.append(dict(task=t, cond='C0', choice='g.py'))
        cells = [('missing', 13, ['g.py', 'b.py', 'c.py', 'd.py', 'f.py'], h2_resolved(i)),    # H2-type
                 ('spurious', 13, ['f.py', 'b.py', 'c.py', 'd.py', 'e.py'], drop_resolved(i)),  # drop, forced
                 ('missing', 37, ['b.py', 'g.py', 'c.py', 'd.py', 'e.py'], True)]               # order only
        for iv, sd, shown, res in cells:
            r2.append(dict(task=t, intervention=iv, seed=sd, shown=shown, choice='g.py' if 'g.py' in shown else 'b.py'))
            ev.append(dict(tag='patch_comp', id=f'X|{t}|{iv}|{sd}', cond='X', task=t, intervention=iv, seed=sd,
                           resolved=res))
        ev.append(dict(tag='patch_comp', id=f'C0|{t}', cond='C0', task=t, resolved=True))
    return ev, r1, r2, clean, gold


class G8Patch(unittest.TestCase):
    def test_h5_equivalent_when_cells_resolve_like_clean(self):
        import v7_g8_patch as P
        out = P.patch_hypotheses(*world(40, lambda i: True, lambda i: False))
        self.assertEqual((out['H5']['tasks'], out['H5']['mean'], out['H5']['verdict']), (40, 0.0, 'equivalent'))
        self.assertEqual((out['H6']['tasks'], out['H6']['mean']), (40, -1.0))
        self.assertLess(out['H6']['p_one_sided'], 0.05)

    def test_h5_underpowered_and_not_equivalent(self):
        import v7_g8_patch as P
        wide = P.patch_hypotheses(*world(30, lambda i: i % 2 == 0, lambda i: True))['H5']
        self.assertEqual(wide['verdict'], 'not established (underpowered)')
        far = P.patch_hypotheses(*world(400, lambda i: i % 4 != 0, lambda i: True))['H5']
        self.assertEqual(far['verdict'], 'not equivalent')
        few = P.patch_hypotheses(*world(10, lambda i: True, lambda i: True))['H5']
        self.assertEqual(few['verdict'], 'descriptive (fewer than 20 tasks)')

    def test_excluded_rows_and_tasks_without_c0_are_dropped(self):
        import v7_g8_patch as P
        ev, r1, r2, clean, gold = world(25, lambda i: True, lambda i: False)
        ev = [dict(e, excluded='install failed', resolved=False) if e['task'] == 't0' else e for e in ev]
        self.assertEqual(P.patch_hypotheses(ev, r1, r2, clean, gold)['H5']['tasks'], 24)

    def test_holm_over_h6_and_h7(self):
        import v7_g8_patch as P
        self.assertEqual(P.holm_family({'H6': 0.01, 'H7': 0.0436}), {'H6': True, 'H7': True})
        self.assertEqual(P.holm_family({'H6': 0.2, 'H7': 0.0436}), {'H6': False, 'H7': False})   # 0.0436 > 0.05/2
        self.assertEqual(P.holm_family({'H6': 0.02, 'H7': 0.0436}), {'H6': True, 'H7': True})
        self.assertEqual(P.holm_family({'H6': 0.03, 'H7': 0.0436}), {'H6': False, 'H7': False})


if __name__ == '__main__':
    unittest.main()
