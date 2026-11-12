"""Plan 3 Task 3: the private Kaggle input dataset and the G6 notebook."""
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))


class Package(unittest.TestCase):
    def test_build_flattens_hashes_and_restores(self):
        import v7_kaggle_package as K
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'repo'
            (src / 'paper/x').mkdir(parents=True)
            (src / 'paper/x/a.py').write_text('print(1)\n')
            (src / 'paper/x/b.json.gz').write_bytes(gzip.compress(b'{"k": 1}'))
            man = K.build(src, ['paper/x/a.py', 'paper/x/b.json.gz'], Path(d) / 'ds')
            self.assertEqual(sorted(p.name for p in (Path(d) / 'ds').iterdir()),
                             ['manifest.json', 'paper__x__a.py', 'paper__x__b.json'])
            self.assertEqual(man['paper/x/b.json.gz'], hashlib.sha256(b'{"k": 1}').hexdigest())
            art = Path(d) / 'art'
            K.restore(Path(d) / 'ds', art)
            self.assertEqual(gzip.decompress((art / 'paper/x/b.json.gz').read_bytes()), b'{"k": 1}')
            self.assertEqual((art / 'paper/x/a.py').read_text(), 'print(1)\n')
            (Path(d) / 'ds/paper__x__a.py').write_text('tampered\n')
            with self.assertRaises(SystemExit):
                K.restore(Path(d) / 'ds', Path(d) / 'art2')

    def test_build_refuses_secrets_and_missing_files(self):
        import v7_kaggle_package as K
        with tempfile.TemporaryDirectory() as d:
            for bad in ('kaggle.json', 'x/.env', 'my_token.txt', 'k.pem'):
                with self.assertRaises(SystemExit):
                    K.build(Path(d), [bad], Path(d) / 'ds')
            with self.assertRaises(FileNotFoundError):
                K.build(Path(d), ['paper/missing.py'], Path(d) / 'ds')

    @unittest.skipUnless((ROOT / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz').exists(), 'labels not present')
    def test_registered_file_list_exists(self):
        import v7_kaggle_package as K
        for p in K.files(ROOT):
            self.assertTrue((ROOT / p).is_file(), p)
        self.assertIn('paper/scripts/v7_g6_readouts.py', K.files(ROOT))

    @unittest.skipUnless((ROOT / 'paper/notebooks/v7_g6.ipynb').exists(), 'notebooks not present')
    def test_notebook_parts_dispatch_to_existing_functions(self):
        import v7_g6_readouts as G
        nb = json.loads((ROOT / 'paper/notebooks/v7_g6.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        self.assertIn("PART = ", code)
        for name in ('run_patch_comp', 'vllm_generate', 'run_settings'):
            self.assertIn(f'G.{name}(', code)
            self.assertTrue(callable(getattr(G, name)))
        self.assertIn("K.restore(", code)


if __name__ == '__main__':
    unittest.main()
