#!/usr/bin/env python3
"""Validate pilot source bindings and evaluation-only labels; no retrieval scores."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

from audit_competition_data import digest, read_tasks, write_new_json
from localization_data import extract_candidates, read_snapshot, snapshot_tree
from localization_evaluation import map_gold, parse_patch
from pilot_integrity import candidate_digest, index_unique, validate_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--upstream-trees', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status':'running',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'python':platform.python_version(), 'command':__import__('sys').argv,
        'retrieval_scores_computed':False}
    records=[]
    try:
        manifest.update(audit_sha256=digest(args.audit), pilot_sha256=digest(args.pilot),
                        upstream_trees_sha256=digest(args.upstream_trees))
        pilot=json.loads(args.pilot.read_text())
        audit=json.loads(args.audit.read_text())
        upstream=index_unique(json.loads(args.upstream_trees.read_text())['tasks'])
        selected_tasks=validate_selection(pilot, audit)
        if set(upstream) != set(selected_tasks):
            raise ValueError('Upstream tree records must cover exactly the pilot task set')
        if digest(args.data_root/'tasks.jsonl') != pilot['tasks_sha256']:
            raise ValueError('Tasks changed since audit')
        tasks={t['instance_id']:t for t in read_tasks(args.data_root/'tasks.jsonl')}
        with (args.output_dir/'mapping.jsonl').open('x') as output:
            for selected in pilot['tasks']:
                task=tasks[selected['instance_id']]
                row={k:task[k] for k in ('instance_id','repo','base_commit')}
                tick=time.perf_counter()
                try:
                    if any(task[k] != selected[k] for k in ('repo', 'base_commit')):
                        raise ValueError('Pilot task metadata differs from tasks.jsonl')
                    for field in ('graph_path','snapshot_path'):
                        relative=selected[field]
                        if digest(args.data_root/relative) != audit['assets'][relative]['sha256']:
                            raise ValueError(f'Asset changed: {relative}')
                    tree=snapshot_tree(args.data_root/selected['snapshot_path'])
                    expected=upstream[task['instance_id']]
                    if expected['base_commit'] != task['base_commit'] or expected['repo'] != task['repo'] or tree['tree_sha'] != expected['tree_sha']:
                        raise ValueError('Snapshot content differs from the upstream base-commit tree')
                    row['snapshot_tree']=tree
                    row['snapshot_tree']['verified_against']=expected['url']
                    graph=json.loads((args.data_root/selected['graph_path']).read_text())
                    sources=read_snapshot(args.data_root/selected['snapshot_path'])
                    candidates, coverage=extract_candidates(sources, graph['nodes'])
                    row['candidate_coverage']=coverage
                    row['candidate_sha256']=candidate_digest(candidates)
                    files=parse_patch(task['patch'])
                    old_paths={f['old_path'] or f['new_path'] for f in files}
                    extra=old_paths-set(sources)
                    if extra:
                        sources.update(read_snapshot(args.data_root/selected['snapshot_path'],extra))
                    row['mapping']=map_gold(task['patch'], sources, candidates)
                    row['status']='passed' if not row['mapping']['preimage_errors'] else 'blocked'
                except (OSError, ValueError) as exc:
                    row.update(status='blocked', error=str(exc))
                row['runtime_seconds']=time.perf_counter()-tick
                output.write(json.dumps(row,sort_keys=True)+'\n');output.flush()
                records.append(row)
                print(row['instance_id'],row['status'], row.get('mapping',{}).get('status'),flush=True)
        manifest['status']='passed' if all(r['status']=='passed' for r in records) else 'blocked'
        manifest['tasks']=len(records)
        manifest['preimage_passed']=sum(r['status']=='passed' for r in records)
        manifest['dataset_fingerprint']=pilot['dataset_fingerprint']
        manifest['mapping_sha256']=digest(args.output_dir/'mapping.jsonl')
        # Label-free bindings let retrieval verify its candidate pool before reading gold.
        manifest['candidate_sha256_by_task']={r['instance_id']:r['candidate_sha256']
                                             for r in records if 'candidate_sha256' in r}
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['runtime_seconds']=time.perf_counter()-started
        manifest['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_new_json(args.output_dir/'manifest.json',manifest)
    return 0 if manifest['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
