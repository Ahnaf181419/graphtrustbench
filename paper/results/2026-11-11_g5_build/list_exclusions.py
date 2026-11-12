"""Per-candidate exclusion reasons for the frozen post-cutoff manifest (registration §2.2 "exclusion reasons").
Reads details_cache.json and the build results in work/ (not committed); writes postcutoff_exclusions.json."""
import gzip, json
from pathlib import Path

D = Path(__file__).resolve().parent
cache = json.loads((D / 'details_cache.json').read_text())
out = []
for c in cache['candidates']:
    d = cache['details'][f"{c['repo']}#{c['pr']}"]
    tid = f"{c['repo'].split('/')[1]}-{c['pr']}"
    if d is None:
        why = 'the linked issue is a pull request'
    elif 'error' in d:
        why = 'issue fetch failed: ' + d['error'].strip()
    else:
        why = json.load(gzip.open(D / 'work' / f'{tid}.json.gz', 'rt'))['reason']
    if why != 'ok':
        out.append(dict(task_id=tid, repo=c['repo'], pr=c['pr'], issue=c['issue'], reason=why))
(D / 'postcutoff_exclusions.json').write_text(json.dumps(out, indent=1) + '\n')
print(len(out), 'excluded of', len(cache['candidates']))
