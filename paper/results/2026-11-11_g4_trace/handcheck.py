import json, subprocess, sys, tarfile, shutil
from pathlib import Path
ROOT = Path('/home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent'); sys.path.insert(0, str(ROOT / 'paper/scripts'))
import v7_g4_trace as G
from gtb6 import repo_ops as O
S = Path(sys.argv[1]); T = {json.loads(l)['instance_id']: json.loads(l) for l in open(ROOT / 'tasks.jsonl')}
for tid in ('fastapi_11194', 'rich_2725', 'requests_6589'):
    t, w = T[tid], json.load(open(ROOT / f'paper/results/2026-11-11_g4_trace/work/{tid}.json'))
    d = S / tid; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    tarfile.open(ROOT / f'snapshots/{tid}.tgz').extractall(d, filter='data')
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d, text=True).strip()
    run = G.docker_cmd(G.IMAGE)
    ok, _ = G.prepare(d, base, t['repo'], t['created_at'], run, str(Path.home() / '.cache/gtb6/uvcache'))
    code, test, _ = O.split_reference(t['patch'] + '\n' + t['test_patch'])
    runner = lambda r, tg: G._outcomes(run, r, tg)
    f2p, p2p = w['tests']['f2p'], w['tests']['p2p']
    good = O.evaluate(d, base, f'```diff\n{code}```', test, f2p, p2p, runner)
    none = O.evaluate(d, base, 'no patch', test, f2p, p2p, runner)
    print(tid, 'install', ok, 'f2p', len(f2p), 'p2p', len(p2p), '| reference fix resolved:', good['resolved'],
          good['failed_tests'][:2], '| no patch resolved:', none['resolved'], flush=True)
    shutil.rmtree(d, ignore_errors=True)
