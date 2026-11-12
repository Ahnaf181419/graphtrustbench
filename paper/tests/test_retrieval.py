import copy
import importlib
import math
from collections import Counter
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
r=importlib.import_module('localization_retrieval')

class TestRetrieval(unittest.TestCase):
    def test_aggregation_averages_seeds_before_tasks(self):
        from run_localization_pilot import aggregate
        rows=[{'method':'bm25','severity':0,'instance_id':task,'repo':'o/r','seed':seed,
               'metrics':{'recall@5':score}} for task,seed,score in [('a',13,1),('a',37,1),('b',13,0)]]
        summary=next(s for s in aggregate(rows)['summaries'] if s['repo']=='all')
        self.assertEqual(summary['tasks'],2)
        self.assertEqual(summary['mean']['recall@5'],0.5)

    def test_bm25_known_two_document_score(self):
        candidates={'a':{'text':'cat cat','path':'','id':''},'b':{'text':'dog','path':'','id':''}}
        scores=r.bm25('cat',candidates,k1=1.2,b=0.75)
        expected=math.log(2)*2*2.2/(2+1.2*(0.25+0.75*2/1.5))
        self.assertAlmostEqual(scores['a'],expected)
        self.assertEqual(scores['b'],0)

    def test_deletion_is_deterministic_nested_and_immutable(self):
        edges=[{'source':str(i),'target':str(i+1),'type':'calls','key':0} for i in range(10)]
        original=copy.deepcopy(edges)
        low, info=r.delete_edges(edges,0.2,13)
        high,_=r.delete_edges(edges,0.4,13)
        self.assertEqual(edges,original)
        self.assertEqual(len(low),8);self.assertEqual(len(high),6)
        self.assertEqual(info['removed_edges'],2)
        self.assertEqual(low,r.delete_edges(edges[::-1],0.2,13)[0])
        self.assertTrue({e['source'] for e in high}.issubset({e['source'] for e in low}))
        self.assertEqual(r.delete_edges(edges,0,13)[0],edges)
        with self.assertRaises(ValueError):r.delete_edges(edges,1.1,13)

    def test_one_hop_propagates_only_from_frozen_seeds(self):
        lexical={'a':2.0,'b':1.0,'c':0.0}
        edges=[{'source':'a','target':'c','type':'calls','key':0}]
        scores,cost=r.one_hop(lexical,edges,seed_count=1,edge_budget=100,restart=0.15)
        self.assertAlmostEqual(scores['a'],0.15)
        self.assertAlmostEqual(scores['c'],0.85)
        self.assertEqual(cost['expanded_nodes'],1)
        empty,_=r.one_hop(lexical,[],seed_count=1,edge_budget=100,restart=0.15)
        self.assertEqual(empty['a'],1)
        self.assertEqual(lexical,{'a':2.0,'b':1.0,'c':0.0})

    def test_one_hop_reports_the_edge_footprint_it_can_possibly_use(self):
        lexical={'a':2.0,'b':1.0,'c':0.0,'d':0.0}
        edges=[{'source':'a','target':'c','type':'calls','key':0},
               {'source':'b','target':'c','type':'calls','key':0},
               {'source':'c','target':'d','type':'calls','key':0}]
        _,cost=r.one_hop(lexical,edges,seed_count=1,edge_budget=100,restart=0.15)
        # Only edges leaving the chosen seed are reachable by construction; the rest of the
        # graph is invisible to the traversal however corrupted it becomes.
        self.assertEqual(cost['edges_incident_to_seeds'],1)
        self.assertEqual(cost['distinct_seed_neighbors'],1)
        self.assertEqual(cost['edges_scanned'],3)

    def test_graph_changes_do_not_change_lexical_inputs_or_scores(self):
        candidates={n:{'id':n,'text':n,'path':n+'.py'} for n in ['cat','dog']}
        saved=copy.deepcopy(candidates)
        before=r.bm25('cat',candidates)
        edges=[{'source':'cat','target':'dog','type':'calls','key':0}]
        for fraction in [0,0.1,0.2,0.4,1]:
            changed,_=r.delete_edges(edges,fraction,13)
            r.one_hop(before,changed,seed_count=1,edge_budget=1,restart=0.15)
            self.assertEqual(r.bm25('cat',candidates),before)
            self.assertEqual(candidates,saved)

    def test_rewire_is_deterministic_immutable_and_count_preserving(self):
        edges=[{'source':str(i),'target':str(i+1),'type':'calls','key':0} for i in range(10)]
        original=copy.deepcopy(edges)
        low,info=r.rewire_edges(edges,0.2,13)
        self.assertEqual(edges,original)
        self.assertEqual(low,r.rewire_edges(edges,0.2,13)[0])
        self.assertEqual(info,r.rewire_edges(edges,0.2,13)[1])
        self.assertEqual(len(low),10)
        self.assertEqual(info['original_edges'],10)
        self.assertEqual(info['rewired_edges'],2)
        self.assertAlmostEqual(info['actual_fraction'],0.2)
        self.assertEqual(r.rewire_edges(edges,0,13)[0],edges)

    def test_rewire_preserves_sources_and_types_without_self_loops_or_duplicates(self):
        edges=[{'source':str(i),'target':str((i+1)%10),'type':'calls','key':0} for i in range(10)]
        ident=lambda e:(e['source'],e.get('type',''),str(e.get('key',0)))
        before={ident(e):e for e in edges}
        rewired,info=r.rewire_edges(edges,0.4,37)
        self.assertEqual(Counter((e['source'],e.get('type','')) for e in rewired),
                         Counter((e['source'],e.get('type','')) for e in edges))
        self.assertTrue(all(e['source']!=e['target'] for e in rewired))
        keys=[r.edge_key(e) for e in rewired]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(info['rewired_edges'],
                         sum(e['target']!=before[ident(e)]['target'] for e in rewired))

    def test_rewire_rejects_bad_fraction(self):
        edges=[{'source':'a','target':'b','type':'calls','key':0}]
        for fraction in (-0.1,1.1):
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):
                r.rewire_edges(edges,fraction,13)

    def test_seed_nodes_matches_the_one_hop_selection_rule(self):
        lexical={'a':2.0,'b':1.0,'c':3.0,'d':0.0}
        self.assertEqual(r.seed_nodes(lexical,2),['c','a'])
        self.assertEqual(r.seed_nodes(lexical,1),['c'])
        # ties break by node id, as in the existing propagation
        self.assertEqual(r.seed_nodes({'x':1.0,'y':1.0},1),['x'])

    def test_scoped_deletion_touches_only_edges_leaving_the_scope(self):
        edges=[{'source':s,'target':str(i),'type':'calls','key':0} for i,s in enumerate('aabbc')]
        original=copy.deepcopy(edges)
        kept,info=r.delete_scoped(edges,0.5,13,scope={'a'})
        self.assertEqual(edges,original)
        self.assertEqual(info['scope_edges'],2)
        self.assertEqual(info['removed_edges'],1)
        self.assertTrue(all(e['source'] in {'a','b','c'} for e in kept))
        # untouched sources keep every edge
        self.assertEqual(sum(e['source']=='b' for e in kept),2)

    def test_scoped_rewire_preserves_invariants_and_reports_shortfall(self):
        edges=[{'source':'a','target':str(i),'type':'calls','key':0} for i in range(4)]+\
              [{'source':'b','target':'x','type':'calls','key':0}]
        rewired,info=r.rewire_scoped(edges,1.0,37,scope={'a'})
        ident=lambda e:(e['source'],e.get('type',''),str(e.get('key',0)))
        self.assertEqual(len(rewired),len(edges))
        self.assertEqual(Counter((e['source'],e.get('type','')) for e in rewired),
                         Counter((e['source'],e.get('type','')) for e in edges))
        self.assertTrue(all(e['source']!=e['target'] for e in rewired))
        keys=[r.edge_key(e) for e in rewired]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertLessEqual(info['rewired_edges'],info['scope_edges'])

    def test_empty_scope_corrupts_nothing(self):
        edges=[{'source':'a','target':'b','type':'calls','key':0}]
        for fn in (r.delete_scoped,r.rewire_scoped):
            with self.subTest(fn=fn.__name__):
                out,info=fn(edges,0.4,13,scope=set())
                self.assertEqual(out,edges)
                self.assertEqual(info['scope_edges'],0)
                self.assertEqual(info.get('removed_edges',info.get('rewired_edges')),0)

    def test_scoped_selection_is_deterministic_and_nested(self):
        edges=[{'source':'a','target':str(i),'type':'calls','key':0} for i in range(10)]
        low,il=r.delete_scoped(edges,0.2,13,scope={'a'})
        high,ih=r.delete_scoped(edges,0.4,13,scope={'a'})
        self.assertEqual(il['removed_edges'],2)
        self.assertEqual(ih['removed_edges'],4)
        self.assertEqual(low,r.delete_scoped(edges,0.2,13,scope={'a'})[0])
        removed_low={e['target'] for e in [{'source':'a','target':t} for t in [] ]}
        # nesting: the 20% removed set is a subset of the 40% one
        low_targets={e['target'] for e in [{'target':t} for t in []]}
        kept_low={e['target'] for e in low}
        kept_high={e['target'] for e in high}
        self.assertTrue(kept_high <= kept_low)

if __name__=='__main__':unittest.main()
