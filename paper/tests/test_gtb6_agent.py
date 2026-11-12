"""Task 9: the agentic-retriever and agent loops, driven by a scripted fake generate."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))


def script(replies):
    """generate(list_of_convs) -> list_of_texts, popping one scripted reply per active state."""
    it = iter(replies)
    return lambda convs: [next(it) for _ in convs]


BM25 = {'a': 3.0, 'b': 2.0, 'c': 1.0, 'd': 0.5}
N2F = {'a': 'a.py', 'b': 'b.py', 'c': 'c.py', 'd': 'd.py'}
SRC = {'a': 'def a():\n    pass\n', 'b': 'def b(x):\n    return x\n'}
SUCC = {'a': ['b', 'c'], 'b': ['d']}


class AgenticRetriever(unittest.TestCase):
    def test_constants_are_registered(self):
        from gtb6 import agent as A
        self.assertEqual((A.AG_STEPS, A.AG_SHOW, A.AGENT_TURNS, A.READ_LINES), (8, 20, 20, 200))

    def test_stops_at_eight_calls_then_ranks_listed_files_first(self):
        from gtb6 import agent as A
        replies = ['{"tool": "neighbours", "node": "a"}'] * 9 + \
                  ['{"tool": "done", "files": ["c.py", "a.py", "zz.py", "a.py"]}']
        state = A.ag_start('t', 'the issue', ['a', 'b'], N2F, SRC)
        A.run_loop([state], script(replies),
                   lambda s, call, text: A.ag_step(s, call, SUCC, BM25, N2F, SRC), 20)
        self.assertTrue(state['done'])
        self.assertEqual(state['steps'], 8)                     # the 9th call did not count
        self.assertTrue(any('used all 8' in m['content'] for m in state['conv'] if m['role'] == 'user'))
        self.assertEqual(state['files'], ['c.py', 'a.py', 'zz.py', 'a.py'])
        self.assertEqual(A.ag_final_ranking(state, BM25, N2F),
                         ['c.py', 'a.py', 'b.py', 'd.py'])      # listed candidates first, then BM25

    def test_two_invalid_replies_end_with_no_files(self):
        from gtb6 import agent as A
        state = A.ag_start('t', 'the issue', ['a', 'b'], N2F, SRC)
        A.run_loop([state], script(['no json here', 'still no tool']),
                   lambda s, call, text: A.ag_step(s, call, SUCC, BM25, N2F, SRC), 20)
        self.assertTrue(state['done'])
        self.assertEqual(state['files'], [])
        self.assertEqual(A.ag_final_ranking(state, BM25, N2F), ['a.py', 'b.py', 'c.py', 'd.py'])


class AgentReadout(unittest.TestCase):
    def test_records_reads_extracts_diff_and_detects_recovery(self):
        from gtb6 import agent as A
        files = {'a.py': '\n'.join(f'line{i}' for i in range(1, 251)), 'b.py': 'return 2\n'}
        read_file = lambda task, p: files.get(p)
        search = lambda task, q: [('a', 'a.py', 'def a():')]
        diff_reply = ('Here is the patch:\n```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n'
                      '+new\n```\n{"tool": "submit"}')
        state = A.agent_start('t', 'the issue', ['a.py', 'c.py', 'd.py'], 'clean')
        A.run_loop([state], script(['{"tool": "read_file", "path": "b.py"}', diff_reply]),
                   lambda s, call, text: A.agent_step(s, call, text, read_file, search, SUCC, N2F), 20)
        self.assertTrue(state['done'])
        self.assertEqual(state['reads'], ['b.py'])
        self.assertTrue(A.recovered(state))                     # b.py is outside the top five
        self.assertIsNotNone(state['diff'])
        self.assertIn('+new', state['diff'])

    def test_read_file_is_capped_at_200_lines(self):
        from gtb6 import agent as A
        files = {'a.py': '\n'.join(f'line{i}' for i in range(1, 301))}
        state = A.agent_start('t', 'the issue', ['a.py'], 'clean')
        A.run_loop([state], script(['{"tool": "read_file", "path": "a.py"}',
                                    '{"tool": "submit"}']),
                   lambda s, call, text: A.agent_step(
                       s, call, text, lambda t, p: files.get(p), lambda t, q: [], {}, {}), 20)
        obs = next(m['content'] for m in state['conv'] if m['role'] == 'user' and 'a.py lines' in m['content'])
        self.assertIn('200: line200', obs)
        self.assertNotIn('201:', obs)


if __name__ == '__main__':
    unittest.main()
