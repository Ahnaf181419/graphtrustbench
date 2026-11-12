"""G8.5: Table 3 rows from a setting's own run (guide G8.5)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))


def rows(c0, cells):
    out = [dict(id='C0|t', cond='C0', task='t', shown=['a', 'b'], choice=c0)]
    for i, (shown, x, p) in enumerate(cells):
        out.append(dict(id=f'X|t|missing|{i}', cond='X', task='t', intervention='missing', seed=i, shown=shown, choice=x))
        out.append(dict(id=f'P|t|missing|{i}', cond='P', task='t', intervention='missing', seed=i, shown=shown, choice=p))
    return out


class Settings(unittest.TestCase):
    def test_code_setting_uses_its_own_c0_and_drops_forced_and_invalid(self):
        import v7_g8_settings as S
        r = rows('a', [(['a', 'c'], 'c', 'a'),        # flip, placebo stays
                       (['c', 'd'], 'c', 'c'),        # forced under this setting (own C0 'a' not shown): dropped
                       (['a', 'c'], 'INVALID', 'a'),  # invalid: dropped
                       (['a', 'c'], 'a', 'c')])       # no flip, placebo flips
        pairs = S.code_setting(r)
        self.assertEqual([(p['flip'], p['pflip']) for p in pairs], [(1.0, 0.0), (0.0, 1.0)])
        h2, pl, (m, lo, hi) = S.setting_row([r])
        self.assertEqual((h2, pl, m), (0.5, 0.5, 0.0))

    def test_modal_rows_take_the_modal_choice_per_item(self):
        import v7_g8_settings as S
        draws = [rows('a', [(['a', 'c'], x, 'a')]) for x in ('c', 'a', 'c', 'a', 'b')]
        draws[1][0]['choice'] = 'b'
        m = S.modal_rows(draws)
        self.assertEqual([r['choice'] for r in m], ['a', 'c', 'a'])     # C0: a,b,a,a,a -> a; X: c,a,c,a,b -> c (earliest)


if __name__ == '__main__':
    unittest.main()
