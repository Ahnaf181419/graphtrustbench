"""The private Kaggle input dataset for the G6 notebooks. Kaggle decompresses .gz uploads, so every file is
stored uncompressed under a flat name (paper/x/y.json.gz -> paper__x__y.json); manifest.json maps each repo
path to the sha256 of its uncompressed content, and restore() rebuilds and checks the tree on Kaggle.

    python3 paper/scripts/v7_kaggle_package.py <out_dir>      # then: kaggle datasets create/version -p <out_dir>
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

SECRET = re.compile(r'(^|/)(kaggle\.json|\.env)$|token|\.pem$', re.I)
DATA = ['paper/config/pilot_v3.json',
        'paper/data/heldout_full_2026-11-07.json',
        'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz',
        'paper/results/2026-11-11_g4_trace/repaired_rankings.json.gz',
        'paper/results/2026-11-11_g4_trace/tests_competition.json'] + \
       [f'paper/results/2026-11-10_gemma_readout/raw_{s}.jsonl'
        for s in ('stage1', 'stage2', 'stage3', 'stage4_rep1', 'stage4_rep2', 'stage4_rep3')]


def files(root):
    root = Path(root)
    code = sorted(str(p.relative_to(root)) for p in (root / 'paper/scripts').glob('*.py'))
    code += sorted(str(p.relative_to(root)) for p in (root / 'paper/scripts/gtb6').glob('*.py'))
    return code + DATA


def flat(path):
    return path.removesuffix('.gz').replace('/', '__')


def build(root, paths, out):
    bad = [p for p in paths if SECRET.search(p)]
    if bad:
        raise SystemExit(f'refusing to package {bad}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for p in paths:
        data = (Path(root) / p).read_bytes()
        if p.endswith('.gz'):
            data = gzip.decompress(data)
        (out / flat(p)).write_bytes(data)
        manifest[p] = hashlib.sha256(data).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def restore(ds, art):
    ds, art = Path(ds), Path(art)
    manifest = json.loads((ds / 'manifest.json').read_text())
    for p, want in manifest.items():
        data = (ds / flat(p)).read_bytes()
        if hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f'{p}: content differs from the manifest')
        dest = art / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(data, mtime=0) if p.endswith('.gz') else data)
    return manifest


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    m = build(root, files(root), sys.argv[1])
    print(len(m), 'files ->', sys.argv[1])
