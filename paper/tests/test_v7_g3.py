"""Plan 2 Task 1 (G3): graph conditions, scorer rankings and cell tables for PPR, the mitigation and low rates."""
import gzip
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
PRED = ROOT / 'paper/results/2026-11-08_extension_v3/predictions.jsonl.gz'
needs_corpus = unittest.skipUnless((ROOT / 'tasks.jsonl').exists() and PRED.exists(),
                                   'competition corpus not present locally')


class G3(unittest.TestCase):
    def test_registered_constants(self):
        import v7_g3_retrievers as G
        self.assertEqual(G.FRACS, (0.01, 0.02, 0.05, 0.1, 0.2, 0.4))
        self.assertEqual(G.SEEDS, (13, 37, 101))
        self.assertEqual(G.IVS, ('missing', 'spurious'))
        self.assertEqual(G.CLEAN, ('clean', 0.0, 0))

    def test_cell_table_on_a_toy_task(self):
        import v7_g3_retrievers as G
        from context_stability import rbo_ext
        clean = ['a.py', 'b.py', 'c.py', 'd.py', 'e.py', 'f.py']
        moved = ['f.py', 'b.py', 'c.py', 'd.py', 'e.py', 'a.py']
        rows = G.cell_table('t', {G.CLEAN: clean, ('spurious', 0.4, 13): moved, ('missing', 0.4, 13): list(clean)}, {'a.py'})
        by = {(r['intervention'], r['seed']): r for r in rows}
        self.assertEqual(len(rows), 2)
        r = by[('spurious', 13)]
        self.assertEqual((r['task'], r['fraction'], r['d_recall'], r['set_changed']), ('t', 0.4, -1.0, True))
        self.assertAlmostEqual(r['one_minus_rbo'], 1 - rbo_ext(clean, moved))
        self.assertGreater(r['one_minus_rbo'], 0)
        r = by[('missing', 13)]
        self.assertEqual((r['d_recall'], r['set_changed']), (0.0, False))
        self.assertAlmostEqual(r['one_minus_rbo'], 0.0)

    def test_rank_all_matches_the_registered_tie_order(self):
        import v7_g3_retrievers as G
        lex = {'x.f': 1.0, 'y.g': 1.0, 'z.h': 2.0}
        cand = {'x.f': {'path': 'x.py'}, 'y.g': {'path': 'y.py'}, 'z.h': {'path': 'z.py'}}
        got = G.rank_all({G.CLEAN: [], ('missing', 0.1, 13): []}, lex, cand,
                         lambda lexical, edges: {'x.f': 0.5})
        self.assertEqual(got, {G.CLEAN: ['x.py', 'z.py', 'y.py'], ('missing', 0.1, 13): ['x.py', 'z.py', 'y.py']})

    def test_build_conditions_rejects_an_intervention_that_differs_from_the_run(self):
        import v7_g3_retrievers as G
        edges = [dict(source=u, target=v, type='calls', key=0) for u, v in (('a', 'b'), ('b', 'c'), ('c', 'a'))]
        conds = G.build_conditions(edges)
        self.assertEqual(len(conds), 1 + 2 * 6 * 3)
        with self.assertRaises(ValueError):
            G.build_conditions(edges, recorded={('missing', 0.4, 13): {'removed_edges': 99}})

    @needs_corpus
    def test_three_task_anchor_covers_every_severity_and_seed(self):
        import v7_g3_retrievers as G
        from gtb6.corpus import load_task
        from localization_retrieval import one_hop
        config = json.loads((ROOT / 'paper/config/pilot_v3.json').read_text())
        selection = json.loads((ROOT / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']
        picks = [next(t for t in selection if t['instance_id'] == 'fastapi_11194')]
        picks += [next(t for t in selection if t['repo'] == r) for r in ('Textualize/rich', 'psf/requests')]
        ids = {p['instance_id'] for p in picks}
        want, info = {}, {}
        with gzip.open(PRED, 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                if r['method'] == 'one_hop' and r['instance_id'] in ids:
                    key = (r['corruption'], r['severity'], r['seed'])
                    want[(r['instance_id'], key)] = r['ranked_files']
                    if r['corruption'] != 'clean':
                        info.setdefault(r['instance_id'], {})[key] = r['intervention']
        for item in picks:
            t = item['instance_id']
            candidates, lexical, edges = load_task(ROOT, item, config)
            conds = G.build_conditions(edges, recorded=info[t], fracs=(0.1, 0.2, 0.4))
            got = G.rank_all(conds, lexical, candidates, lambda l, e: one_hop(l, e, **config['one_hop'])[0])
            self.assertEqual(len(got), 19)
            for key, ranking in got.items():
                with self.subTest(task=t, key=key):
                    self.assertEqual(ranking, want[(t, key)])


if __name__ == '__main__':
    unittest.main()
