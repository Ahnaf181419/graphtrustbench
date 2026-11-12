import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READOUT = ROOT / 'paper/results/2026-11-10_gemma_readout'
EXT = ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'
needs_labels = unittest.skipUnless(EXT.exists() and (ROOT / 'tasks.jsonl').exists(),
                                   'gold labels or corpus not present; regenerate them from the competition data')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gemma_readout as g

enc = lambda s: re.findall(r'\w+|[^\w\s]|\s+', s)


class Readout(unittest.TestCase):
    def test_parse_codes_choices(self):
        shown = ['a.py', 'b.py']
        self.assertEqual(g.parse('think {"files": ["x"]} then {"files": ["./b.py", "a.py"]}', shown)[0], 'b.py')
        self.assertEqual(g.parse('{"files": [], "none_relevant": true}', shown)[0], 'NONE')
        self.assertEqual(g.parse('{"files": ["z.py"]}', shown)[0], 'OFF-LIST')
        self.assertEqual(g.parse('no json', shown)[0], 'INVALID')
        self.assertEqual(g.parse('{"files": ["z.py"]}', None)[0], 'z.py')
        self.assertTrue(g.is_forced('c.py', shown))
        self.assertFalse(g.is_forced('NONE', shown))

    def test_placebo_replaces_only_incoming_with_non_gold_draws(self):
        clean = [f'{i}.py' for i in range(20)]
        shown = ['0.py', 'x.py', '2.py', 'y.py', '4.py']
        p = g.placebo('t', 'missing', 13, clean, shown, gold={'5.py'})
        self.assertEqual([p[0], p[2], p[4]], ['0.py', '2.py', '4.py'])
        self.assertTrue(set(p[1::2]) <= set(clean[6:20]))
        self.assertEqual(p, g.placebo('t', 'missing', 13, clean, shown, gold={'5.py'}))
        self.assertIsNone(g.placebo('t', 'missing', 13, clean[:6], shown, gold=set()))

    def test_render_orders_by_bm25_skips_nested_and_respects_budget(self):
        defs = [dict(id='m.A', spans=[[1, 4]], text='class A:\n    def f(self):\n        return 1\n    x = 2\n', bm25=1.0),
                dict(id='m.A.f', spans=[[2, 3]], text='    def f(self):\n        return 1\n', bm25=5.0),
                dict(id='m.g', spans=[[6, 7]], text='def g():\n    return 3\n', bm25=0.5)]
        text, spans = g.render_definitions(defs, enc, ''.join, budget=10_000)
        self.assertTrue(text.startswith('# m.A.f'))
        self.assertEqual(text.count('return 1'), 2)          # f alone, then inside A (not contained in f)
        self.assertIn('# m.g', text)
        short, kept = g.render_definitions(defs, enc, ''.join, budget=18)
        self.assertLessEqual(len(enc(short)), 18)
        self.assertEqual(kept, [[2, 2]])

    def test_edited_lines_and_visibility(self):
        patch = '--- a/x.py\n+++ b/x.py\n@@ -3,3 +3,3 @@\n a\n-b\n+B\n c\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+n\n'
        self.assertEqual(g.edited_lines(patch), {'x.py': [4]})
        self.assertEqual(g.visibility([4], [[1, 5]]), 'all')
        self.assertEqual(g.visibility([4, 9], [[1, 5]]), 'some')
        self.assertEqual(g.visibility([4], []), 'none')

    @needs_labels
    def test_analysis_reproduces_the_committed_readout_values(self):
        committed = json.loads((ROOT / 'paper/reports/gemma_readout/readout_values.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            got = g.analyze(READOUT, EXT, Path(d), Path(d) / 'f2.png', ROOT / 'tasks.jsonl')
            self.assertEqual(json.loads((Path(d) / 'readout_values.json').read_text()), committed)
        self.assertEqual(json.loads(json.dumps(got)), committed)

    @needs_labels
    def test_core_analysis_takes_any_task_set(self):
        clean, corr, gold = g.rankings(g.load_rows(EXT))
        rows = g.load_rows(EXT)
        retrievable = {r['instance_id']: set(r['gold_files']) - set(r['unretrievable_gold_files'])
                       for r in rows if r['method'] == 'one_hop' and r['severity'] == 0}
        stages = g.load_stages(READOUT)
        full = g.analyze_core(*stages, clean, corr, gold, retrievable)
        committed = json.loads((ROOT / 'paper/reports/gemma_readout/readout_values.json').read_text())
        committed.pop('gold_visibility_C0')
        self.assertEqual(json.loads(json.dumps(full)), committed)
        # a subset of tasks is analysed on its own
        keep = sorted(clean)[:40]
        sub = [[r for r in st if r['task'] in keep] for st in g.load_stages(READOUT)]
        part = g.analyze_core(*sub, {t: clean[t] for t in keep}, {k: v for k, v in corr.items() if k[0] in keep},
                              gold, retrievable)
        self.assertEqual(part['table3'][0]['tasks'], 40)


if __name__ == '__main__':
    unittest.main()
