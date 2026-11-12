"""Plan 3 (G6): GPU readouts, tested on CPU with fake generators."""
import gzip
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
READOUT = ROOT / 'paper/results/2026-11-10_gemma_readout'
EXT = ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'
G5 = ROOT / 'paper/results/2026-11-11_g5_build'
needs_labels = unittest.skipUnless(EXT.exists(), 'gold labels not present')
needs_g5 = unittest.skipUnless((G5 / 'postcutoff_full.json.gz').exists(), 'G5 full tasks not built (v7_g5_build.py)')

enc = lambda s: re.findall(r'\w+|[^\w\s]|\s+', s)
dec = ''.join

TOY_FILE = '''\
def helper():
    return 1


class Box:
    def open(self):
        return helper()
'''


def raw(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


class G6Decision(unittest.TestCase):
    @needs_labels
    def test_decision_stages_reproduce_the_committed_v5_items(self):
        import gemma_readout as g
        clean, corr, gold = g.rankings(g.load_rows(EXT))
        corr = {k: v for k, v in corr.items() if k[2] == 0.4}
        committed = {tag: raw(READOUT / f'raw_{tag}.jsonl')
                     for tag in ('stage1', 'stage2', 'stage3', 'stage4_rep1', 'stage4_rep2', 'stage4_rep3')}
        reply = {(tag, r['id']): r for tag, rows in committed.items() for r in rows}
        got, notes = {}, []
        def stage(items, tag):
            got[tag] = [it['id'] for it in items]
            return [dict({k: v for k, v in it.items() if k != 'conv'}, choice=reply[(tag, it['id'])]['choice'])
                    for it in items]
        g.decision_stages(sorted(clean), clean, corr, gold, lambda t, files: None, lambda t: None, stage, notes.append)
        for tag, rows in committed.items():
            self.assertEqual(got[tag], [r['id'] for r in rows], tag)
        self.assertIn('placebo pool too small: 0', notes)

    def test_pc_contexts_match_competition_rendering_on_a_toy_task(self):
        import gemma_readout as g
        import v7_g6_readouts as G
        full = [dict(task_id='rich-1', repo='rich', problem_statement='Box open fails',
                     files={'pkg/m.py': TOY_FILE},
                     defs={'pkg.m.helper': dict(qualname='helper', source='def helper():\n    return 1\n'),
                           'pkg.m.Box': dict(qualname='Box', source=TOY_FILE.split('\n\n\n')[1]),
                           'pkg.m.Box.open': dict(qualname='Box.open', source='def open(self):\n    return helper()\n')},
                     node2file={'pkg.m.helper': 'pkg/m.py', 'pkg.m.Box': 'pkg/m.py', 'pkg.m.Box.open': 'pkg/m.py'},
                     bm25={'pkg.m.helper': 1.0, 'pkg.m.Box': 3.0, 'pkg.m.Box.open': 2.0})]
        rendered, spans, issues, repos = G.pc_contexts(full, {'rich-1': {'pkg/m.py'}}, enc, dec)
        defs = [dict(id='pkg.m.Box', path='pkg/m.py', spans=[[5, 7]], text=TOY_FILE.split('\n\n\n')[1], bm25=3.0),
                dict(id='pkg.m.Box.open', path='pkg/m.py', spans=[[6, 7]], text='    def open(self):\n        return helper()\n', bm25=2.0),
                dict(id='pkg.m.helper', path='pkg/m.py', spans=[[1, 2]], text='def helper():\n    return 1\n', bm25=1.0)]
        self.assertEqual((rendered[('rich-1', 'pkg/m.py')], spans[('rich-1', 'pkg/m.py')]),
                         g.render_definitions(defs, enc, dec))
        self.assertEqual(spans[('rich-1', 'pkg/m.py')], [[5, 7], [1, 2]])
        self.assertEqual((issues['rich-1'], repos['rich-1']), ('Box open fails', 'Textualize/rich'))

    @needs_g5
    def test_pc_inputs_keys_and_counts(self):
        import v7_g6_readouts as G
        full = json.load(gzip.open(G5 / 'postcutoff_full.json.gz', 'rt'))
        tasks, clean, corr, gold = G.pc_inputs(full, json.load(gzip.open(G5 / 'postcutoff_rankings.json.gz', 'rt')))
        frozen = json.loads((ROOT / 'paper/prereg/postcutoff_tasks.json').read_text())['tasks']
        self.assertEqual(tasks, sorted(t['task_id'] for t in frozen))
        self.assertEqual(len(corr), 2 * 3 * len(tasks))
        self.assertEqual({(k[1], k[2]) for k in corr}, {('missing', 0.4), ('spurious', 0.4)})
        self.assertEqual(gold, {t['task_id']: set(t['gold_files']) for t in frozen})

    def test_stage_is_atomic_and_resumable(self):
        import v7_g6_readouts as G
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            calls = []
            def gen(convs):
                calls.append(len(convs))
                if len(calls) == 2:
                    raise RuntimeError('session died')
                return [('{"files": ["a.py"]}', 7)] * len(convs)
            stage = G.make_stage(gen, out, lambda m: None)
            items = [dict(id='C0|t', cond='C0', task='t', shown=['a.py'], conv=[])]
            self.assertEqual(stage(items, 's1')[0]['choice'], 'a.py')
            with self.assertRaises(RuntimeError):
                stage(items, 's2')
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['raw_s1.jsonl'])
            stage = G.make_stage(lambda convs: [('{"files": ["a.py"]}', 7)] * len(convs), out, lambda m: None)
            calls.clear()
            self.assertEqual(stage(items, 's1')[0]['n_prompt_tokens'], 7)      # read back, not regenerated
            self.assertEqual(calls, [])


APPENDIX_A_USER = """Repository: psf/requests

Issue:
add is wrong

Relevant files (in the order given):

[1] a.py
```python
x = 1
```

[2] b.py
```python
y = 2
```

Write a patch that resolves the issue. Reply with a single unified diff in a ```diff block, with paths relative to the repository root and a/ and b/ prefixes. Change only what is needed."""


class G6Patch(unittest.TestCase):
    def test_patch_prompt_is_appendix_a_verbatim(self):
        import v7_g6_readouts as G
        conv = G.patch_conv('psf/requests', 'add is wrong', ['a.py', 'b.py'], {'a.py': 'x = 1', 'b.py': 'y = 2'})
        self.assertEqual(conv, [{'role': 'system', 'content': 'You are a software engineer. Write a patch that resolves the GitHub issue.'},
                                {'role': 'user', 'content': APPENDIX_A_USER}])

    def test_render_full_cuts_at_a_line_boundary_within_budget(self):
        import v7_g6_readouts as G
        text = ''.join(f'line_{i} = {i}\n' for i in range(100))
        self.assertEqual(G.render_full(text, enc, dec, budget=10_000), text)
        cut = G.render_full(text, enc, dec, budget=50)
        self.assertLessEqual(len(enc(cut)), 50)
        self.assertTrue(text.startswith(cut + '\n'))
        self.assertEqual(cut.count('\n') + 1, len(cut.splitlines()))

    def test_patch_items_and_floor(self):
        import v7_g6_readouts as G
        clean = {'a': ['1', '2', '3', '4', '5', '6'], 'b': ['1', '2', '3', '4', '5', '6']}
        r2 = [dict(task='a', intervention='missing', seed=13, shown=['6', '2', '3', '4', '5']),
              dict(task='b', intervention='missing', seed=13, shown=['6', '2', '3', '4', '5'])]
        r3 = [dict(task='a', intervention='missing', seed=13, shown=['1', '6', '3', '4', '5'])]
        rep = {'a': ['2', '1', '3', '4', '5'], 'b': ['6', '2', '3', '4', '5']}
        items = G.patch_items(['a', 'b'], clean, r2, r3, lambda t: t == 'a', rep)
        self.assertEqual([(x['id'], x['shown']) for x in items],
                         [('C0|a', clean['a'][:5]), ('X|a|missing|13', r2[0]['shown']),
                          ('P|a|missing|13', r3[0]['shown']), ('REP|a', rep['a'])])
        same = G.patch_items(['a'], clean, [], [], lambda t: True, {'a': clean['a'][:5] + ['9']})
        self.assertEqual([x['cond'] for x in same], ['C0'])
        many = [dict(id=f'X|t{i}', cond='X', task=f't{i}') for i in range(300)] + [dict(id='C0|a', cond='C0', task='a')]
        f1, f1b, f2 = G.floor_batches(many, 1), G.floor_batches(many, 1), G.floor_batches(many, 2)
        self.assertEqual(f1, f1b)
        self.assertNotEqual(f1, f2)
        self.assertEqual(len(f1), 201)
        self.assertEqual([x['id'] for x in f1 if x['cond'] == 'C0r'], ['C0r1|a'])
        self.assertTrue(all(x.get('filler') for x in f1 if x['cond'] != 'C0r'))

    def test_overlong_prompt_aborts_before_generation(self):
        import v7_g6_readouts as G
        called = []
        items = [dict(id='C0|a', conv=[{'role': 'user', 'content': 'w ' * 20_000}])]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                G.run_raw(items, 'patch_x', lambda c: called.append(c) or [('', 0)], enc, Path(d), lambda m: None)
            self.assertEqual(called, [])
            self.assertEqual(list(Path(d).iterdir()), [])
            G.run_raw(items[:0] + [dict(id='C0|b', conv=[{'role': 'user', 'content': 'short'}])], 'patch_y',
                      lambda c: [('```diff\n```', 3)] * len(c), enc, Path(d), lambda m: None)
            row = raw(Path(d) / 'raw_patch_y.jsonl')[0]
            self.assertEqual((row['id'], row['raw'], row['n_prompt_tokens']), ('C0|b', '```diff\n```', 3))

    @needs_labels
    def test_no_gold_patch_line_in_competition_prompts(self):
        # prompts carry pre-fix snapshot text only: no added line of the reference patch may appear
        import gemma_readout as g
        import v7_g6_readouts as G
        if not (ROOT / 'snapshots/fastapi_11194.tgz').exists():
            self.skipTest('competition snapshots not present')
        clean, corr, gold = g.rankings(g.load_rows(EXT))
        tasks = sorted(clean)[:6]
        r2 = [r for r in raw(READOUT / 'raw_stage2.jsonl') if r['task'] in tasks]
        items = G.patch_items(tasks, clean, r2, [], lambda t: True)
        with open(ROOT / 'tasks.jsonl') as fh:
            comp = {t['instance_id']: t for t in map(json.loads, fh) if t['instance_id'] in tasks}
        sel = {t['instance_id']: t for t in json.loads((ROOT / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']}
        texts = G.comp_patch_texts(ROOT, sel, items, enc, dec)
        convs = G.attach_patch_convs(items, texts, {t: comp[t]['repo'] for t in tasks},
                                     {t: comp[t]['problem_statement'] for t in tasks})
        from localization_data import read_snapshot
        pre = {t: read_snapshot(ROOT / sel[t]['snapshot_path'], {p for x in items if x['task'] == t for p in x['shown']})
               for t in tasks}
        checked = 0
        for x in convs:
            prompt = x['conv'][1]['content']
            before = '\n'.join(pre[x['task']].get(p, '') for p in x['shown'])
            for line in comp[x['task']]['patch'].splitlines():
                body = line[1:].strip()
                if line.startswith('+') and not line.startswith('+++') and len(body) > 30 and body not in before \
                        and body not in comp[x['task']]['problem_statement']:
                    self.assertNotIn(body, prompt, x['id'])
                    checked += 1
        self.assertGreater(checked, 20)


class G6Settings(unittest.TestCase):
    @needs_labels
    def test_settings_items_are_c0_h2_cells_and_their_placebos(self):
        import gemma_readout as g
        import v7_g6_readouts as G
        clean, corr, gold = g.rankings(g.load_rows(EXT))
        r1, r2, r3, _ = g.load_stages(READOUT)
        items = G.settings_items(r1, r2, r3, clean, gold)
        conds = [x['cond'] for x in items]
        self.assertEqual((conds.count('C0'), conds.count('X')), (115, 421))       # 421 H2 cells (registration)
        xk = {(x['task'], x['intervention'], x['seed']) for x in items if x['cond'] == 'X'}
        pk = {(x['task'], x['intervention'], x['seed']) for x in items if x['cond'] == 'P'}
        self.assertTrue(pk <= xk)
        self.assertEqual(len({x['id'] for x in items}), len(items))

    def test_paraphrase_replaces_the_question_and_keeps_the_json_lines(self):
        import gemma_readout as g
        import v7_g6_readouts as G
        for i, line in enumerate(G.PARA, 1):
            ask = G.ask_for(f'para{i}')
            self.assertTrue(ask.startswith(line + '\n'))
            self.assertEqual(ask.split('\n')[1:], g.ASK.split('\n')[1:])
            conv = g.conversation('r/x', 'issue', ['a.py'], {'a.py': 'x'}, ask=ask)
            self.assertTrue(conv[1]['content'].endswith(ask))
        self.assertEqual(G.ask_for('think'), g.ASK)
        self.assertEqual(g.conversation('r/x', 'i', ['a.py'], {'a.py': 'x'}), g.conversation('r/x', 'i', ['a.py'], {'a.py': 'x'}, ask=g.ASK))

    def test_settings_that_share_an_engine(self):
        import v7_g6_readouts as G
        self.assertEqual(len({G.engine_for(x) for x in ('para1', 'para2', 'para3', 'sampled')}), 1)
        self.assertNotEqual(G.engine_for('m26'), G.engine_for('para1'))           # another model
        with self.assertRaises(AssertionError):
            G.run_settings(['para1', 'm26'], None, None, None, None)
        self.assertEqual(G.engine_for('think')[:3], (4096, 32768, True))
        self.assertEqual(G.engine_for('b8000')[1:5], (49152, False, 4, 8000))
        self.assertEqual(G.engine_for('e4')[3], 1)
        with self.assertRaises(AssertionError):
            G.run_settings(['think', 'para1'], None, None, None, None)

    def test_modal_choice_ties_go_to_the_earliest_draw(self):
        import v7_g6_readouts as G
        self.assertEqual(G.modal_choice(['b', 'a', 'a', 'b', 'c']), 'b')
        self.assertEqual(G.modal_choice(['c', 'a', 'a', 'b', 'b']), 'a')
        self.assertEqual(G.modal_choice(['INVALID', 'a', 'INVALID', 'b', 'c']), 'INVALID')


if __name__ == '__main__':
    unittest.main()
