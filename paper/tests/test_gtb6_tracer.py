"""Task 8: the tracer records real call edges, closures stay unmapped, and misses are classified."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'paper/scripts'
sys.path.insert(0, str(SCRIPTS))

TOY_M = '''\
def g():
    return 1

def f():
    return g()

def closure_user():
    def inner():
        return g()
    return inner()

class C:
    def __init__(self):
        self.x = f()
    def method(self):
        return g()

def deco(fn):
    return fn

@deco
def decorated():
    return g()
'''

TOY_TEST = '''\
import pkg.m

def test_toy():
    pkg.m.f()
    pkg.m.closure_user()
    pkg.m.C().method()
    pkg.m.decorated()
'''


class Tracer(unittest.TestCase):
    def test_plugin_records_edges_and_closure_is_unmapped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pkg = root / 'pkg'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'm.py').write_text(TOY_M)
            (root / 'test_toy.py').write_text(TOY_TEST)
            out = root / 'trace.json'
            env = dict(os.environ, GTB_SRC=str(pkg), GTB_OUT=str(out),
                       PYTHONPATH=str(SCRIPTS), PYTHONDONTWRITEBYTECODE='1')
            r = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'gtb6.tracer',
                                'test_toy.py', '-q'], cwd=root, env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            edges = {tuple(e) for e in json.loads(out.read_text())}
            self.assertIn(('pkg.m', 'f', 'pkg.m', 'g'), edges)
            self.assertIn(('pkg.m', 'C.__init__', 'pkg.m', 'f'), edges)
            self.assertIn(('pkg.m', 'decorated', 'pkg.m', 'g'), edges)
            self.assertIn(('pkg.m', 'closure_user.<locals>.inner', 'pkg.m', 'g'), edges)

            from gtb6.tracer import to_node_ids
            resolve = lambda m, q: None if '<locals>' in q else f'{m}.{q}'
            mapped, unmapped = to_node_ids(edges, resolve)
            self.assertIn(('pkg.m.f', 'pkg.m.g'), mapped)
            self.assertIn(('pkg.m.decorated', 'pkg.m.g'), mapped)
            self.assertGreaterEqual(unmapped, 1)          # the closure edge

    def test_classify_returns_each_kind_in_priority_order(self):
        from gtb6 import misskinds as M
        self.assertEqual(M.KINDS, ('dunder', 'decorated', 'method', 'function', 'unresolved'))
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / 'pkg'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'm.py').write_text(TOY_M)
            try:
                self.assertEqual(M.classify(d, 'pkg.m', 'C.__init__'), 'dunder')
                self.assertEqual(M.classify(d, 'pkg.m', 'decorated'), 'decorated')
                self.assertEqual(M.classify(d, 'pkg.m', 'C.method'), 'method')
                self.assertEqual(M.classify(d, 'pkg.m', 'g'), 'function')
                self.assertEqual(M.classify(d, 'pkg.m', 'nope'), 'unresolved')
            finally:
                M.clear_cache()


if __name__ == '__main__':
    unittest.main()
