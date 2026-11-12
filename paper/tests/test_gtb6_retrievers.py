import gzip
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
from gtb6 import retrievers as R
from localization_retrieval import one_hop, delete_edges, rewire_edges

edge = lambda u, v: dict(source=u, target=v, type='calls', key=0)


class Retrievers(unittest.TestCase):
    def test_ppr_conserves_mass_and_prefers_reachable(self):
        lex = {'a': 3.0, 'b': 1.0, 'c': 0.5, 'd': 0.0}
        s = R.ppr(lex, [edge('a', 'c'), edge('c', 'a')], seed_count=2)
        self.assertAlmostEqual(sum(s.values()), 1.0, places=6)
        self.assertGreater(s['c'], s.get('d', 0.0))

    def test_ppr_ignores_unmapped_targets_and_self_loops(self):
        lex = {'a': 1.0, 'b': 0.5}
        with_noise = R.ppr(lex, [edge('a', 'b'), edge('a', 'zz'), edge('a', 'a')], seed_count=1)
        self.assertEqual(with_noise, R.ppr(lex, [edge('a', 'b')], seed_count=1))

    def test_dropout_drops_a_shared_edge_in_the_same_members_of_every_graph(self):
        seen = {}
        def recorder(tag):
            def scorer(lexical, edges):
                seen.setdefault(tag, []).append(('a', 'b') in {(e['source'], e['target']) for e in edges})
                return dict.fromkeys(lexical, 0.0)
            return scorer
        lex = {'a': 1.0, 'b': 0.5, 'c': 0.1}
        R.dropout_ensemble(lex, [edge('a', 'b')], 't1', recorder('clean'))
        R.dropout_ensemble(lex, [edge('a', 'b'), edge('b', 'c')], 't1', recorder('corrupted'))
        self.assertEqual(len(seen['clean']), 10)
        self.assertEqual(seen['clean'], seen['corrupted'])

    def test_dropout_rate_zero_reproduces_the_scorer(self):
        lex = {'a': 3.0, 'b': 1.0, 'c': 0.5}
        edges = [edge('a', 'c'), edge('b', 'c')]
        base = one_hop(lex, edges, 2, 100, 0.15)[0]
        ens = R.dropout_ensemble(lex, edges, 't', lambda l, e: one_hop(l, e, 2, 100, 0.15)[0], p=0.0)
        for n in lex:
            self.assertAlmostEqual(ens[n], base[n])

    def test_rank_breaks_ties_like_the_registered_one_hop_and_keeps_unscored_candidates(self):
        lex = {'x.f': 1.0, 'y.g': 1.0, 'z.h': 2.0}
        cand = {'x.f': {'path': 'x.py'}, 'y.g': {'path': 'y.py'}, 'z.h': {'path': 'z.py'}}
        self.assertEqual(R.rank({'x.f': 0.5}, lex, cand), ['x.py', 'z.py', 'y.py'])
        self.assertEqual(R.rank({}, lex, cand), ['z.py', 'x.py', 'y.py'])

    def test_one_hop_anchor_on_committed_rankings(self):
        if not (ROOT / 'tasks.jsonl').exists():
            self.skipTest('competition corpus not present locally')
        from gtb6.corpus import load_task
        config = json.loads((ROOT / 'paper/config/pilot_v3.json').read_text())
        selection = json.loads((ROOT / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']
        picks = [next(t for t in selection if t['instance_id'] == 'fastapi_11194')]
        picks += [next(t for t in selection if t['repo'] == r) for r in ('Textualize/rich', 'psf/requests')]
        want = {}
        with gzip.open(ROOT / 'paper/results/2026-11-08_extension_v3/predictions.jsonl.gz', 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                if r['method'] == 'one_hop' and (r['corruption'], r['severity'], r['seed']) in (('clean', 0.0, 0), ('spurious', 0.4, 13)):
                    want[(r['instance_id'], r['corruption'])] = r['ranked_files']
        for item in picks:
            candidates, lexical, edges = load_task(ROOT, item, config)
            for kind, graph in (('clean', delete_edges(edges, 0.0, 0)[0]), ('spurious', rewire_edges(edges, 0.4, 13)[0])):
                with self.subTest(task=item['instance_id'], kind=kind):
                    got = R.rank(one_hop(lexical, graph, **config['one_hop'])[0], lexical, candidates)
                    self.assertEqual(got, want[(item['instance_id'], kind)])
                    for scores in (R.ppr(lexical, graph), R.dropout_ensemble(
                            lexical, graph, item['instance_id'], lambda l, e: one_hop(l, e, **config['one_hop'])[0])):
                        self.assertEqual(sorted(R.rank(scores, lexical, candidates)), sorted(got))


if __name__ == '__main__':
    unittest.main()
