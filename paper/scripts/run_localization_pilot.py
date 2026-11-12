#!/usr/bin/env python3
"""Frozen CPU pilot: predict first, then load separately verified evaluation labels."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from audit_competition_data import digest, read_tasks, write_new_json
from localization_data import extract_candidates, read_snapshot, retrieval_task
from localization_retrieval import (bm25, delete_edges, delete_scoped, one_hop, rank_files,
                                    rewire_edges, rewire_scoped, seed_nodes)
from pilot_integrity import (candidate_digest, conditions, index_unique,
                             validate_mapping_sources, validate_selection)


def aggregate(rows: list[dict]) -> dict:
    """Average technical seeds within task before averaging independent tasks."""
    # A run that varies the corruption kind must never pool kinds into one mean;
    # single-kind legacy runs keep their exact historical summary shape.
    kinds = {row['corruption'] for row in rows if row['severity']}
    ident = ('method', 'corruption', 'severity') if len(kinds) > 1 else ('method', 'severity')
    task_groups=defaultdict(list)
    for row in rows:
        task_groups[tuple(row[k] for k in ident)+(row['instance_id'],row['repo'])].append(row)
    tasks=[]
    for key,observations in sorted(task_groups.items()):
        record=dict(zip(ident+('instance_id','repo'),key))
        record['metrics']={key:sum(r['metrics'][key] for r in observations)/len(observations) for key in observations[0]['metrics']}
        tasks.append(record)
    clean={(t['method'],t['instance_id']):t['metrics'] for t in tasks if t['severity']==0}
    groups=defaultdict(list)
    for row in tasks:
        row['delta_from_clean']={key:value-clean[(row['method'],row['instance_id'])][key] for key,value in row['metrics'].items()}
        for repo in ('all',row['repo']):
            groups[tuple(row[k] for k in ident)+(repo,)].append(row)
    summaries=[]
    for key,observations in sorted(groups.items()):
        record=dict(zip(ident+('repo',),key)); record['tasks']=len(observations)
        record['mean']={k:sum(t['metrics'][k] for t in observations)/len(observations) for k in observations[0]['metrics']}
        record['mean_delta_from_clean']={k:sum(t['delta_from_clean'][k] for t in observations)/len(observations) for k in observations[0]['metrics']}
        summaries.append(record)
    seed_means=[]
    for key in sorted({tuple(r[k] for k in ident)+(r['seed'],) for r in rows},key=str):
        group=[r for r in rows if tuple(r[k] for k in ident)+(r['seed'],)==key]
        record=dict(zip(ident+('seed',),key)); record['tasks']=len(group)
        record['recall@5']=sum(r['metrics']['recall@5'] for r in group)/len(group)
        seed_means.append(record)
    return {'aggregation':'seeds averaged within task; then macro mean across tasks; pilot, no significance claims',
            'summaries':summaries,'per_task':tasks,'seed_means':seed_means}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--mapping-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('paper/config/pilot.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Refusing to overwrite an existing run')
    args.output_dir.mkdir(parents=True)
    started=time.perf_counter()
    manifest={'experiment_id':args.output_dir.name,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
              'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
              'python':platform.python_version(),'platform':platform.platform(),'command':sys.argv if argv is None else argv}
    try:
        manifest.update(pilot_sha256=digest(args.pilot),audit_sha256=digest(args.audit),
                        config_sha256=digest(args.config),mapping_sha256=digest(args.mapping_dir/'mapping.jsonl'),
                        mapping_manifest_sha256=digest(args.mapping_dir/'manifest.json'))
        pilot=json.loads(args.pilot.read_text()); audit=json.loads(args.audit.read_text())
        mapping_manifest=json.loads((args.mapping_dir/'manifest.json').read_text())
        config=json.loads(args.config.read_text())
        manifest['configuration']=config
        manifest['dataset_fingerprint']=pilot['dataset_fingerprint']
        selected_tasks=validate_selection(pilot, audit)
        validate_mapping_sources(mapping_manifest, manifest['source_hashes'])
        if audit['status']!='passed_schema_preflight' or mapping_manifest['status']!='passed' or mapping_manifest['pilot_sha256']!=digest(args.pilot):
            raise ValueError('Passing dataset and mapping preflights required')
        if mapping_manifest['audit_sha256']!=digest(args.audit) or mapping_manifest.get('mapping_sha256')!=manifest['mapping_sha256']:
            raise ValueError('Mapping outputs or audited inputs changed since validation')
        if pilot['dataset_fingerprint']!=audit['dataset_fingerprint'] or digest(args.data_root/'tasks.jsonl')!=pilot['tasks_sha256']:
            raise ValueError('Dataset changed since preflight')
        if set(mapping_manifest.get('candidate_sha256_by_task', {})) != set(selected_tasks):
            raise ValueError('Rebuild mapping with candidate hashes for exactly the pilot task set')
        # Discard all non-allowlisted task fields before any retrieval call.
        tasks={t['instance_id']:retrieval_task(t) for t in read_tasks(args.data_root/'tasks.jsonl')}
        interventions=conditions(config)
        with (args.output_dir/'predictions.jsonl').open('x') as output:
            for item in pilot['tasks']:
                task=tasks[item['instance_id']]
                if any(task[k] != item[k] for k in ('repo', 'base_commit')):
                    raise ValueError('Pilot task metadata differs from tasks.jsonl')
                for field in ('graph_path','snapshot_path'):
                    if digest(args.data_root/item[field])!=audit['assets'][item[field]]['sha256']:
                        raise ValueError(f'Changed asset: {item[field]}')
                tick=time.perf_counter()
                graph=json.loads((args.data_root/item['graph_path']).read_text())
                candidates,coverage=extract_candidates(read_snapshot(args.data_root/item['snapshot_path']),graph['nodes'])
                candidate_sha=candidate_digest(candidates)
                if candidate_sha != mapping_manifest['candidate_sha256_by_task'][task['instance_id']]:
                    raise ValueError(f'Candidate pool differs from verified mapping: {task["instance_id"]}')
                index_seconds=time.perf_counter()-tick
                tick=time.perf_counter()
                lexical=bm25(task['problem_statement'],candidates,**config['bm25'])
                lexical_seconds=time.perf_counter()-tick
                clean_lexical=rank_files(lexical,lexical,candidates)
                # The seed set is clean-graph and BM25-derived, so it is identical across every
                # condition; targeted corruption may therefore be scoped by it deterministically.
                seed_scope=set(seed_nodes(lexical,config['one_hop']['seed_count']))
                for corruption,severity,seed in interventions:
                    tick=time.perf_counter()
                    # Clean is severity-zero deletion: a verified no-op that keeps the
                    # intervention record shape identical to protocol 0.1/0.2 clean rows.
                    if corruption=='clean' or corruption=='missing':
                        edges,intervention=delete_edges(graph['edges'],severity,seed)
                    elif corruption=='spurious':
                        edges,intervention=rewire_edges(graph['edges'],severity,seed)
                    elif corruption=='missing_targeted':
                        edges,intervention=delete_scoped(graph['edges'],severity,seed,seed_scope)
                    elif corruption=='spurious_targeted':
                        edges,intervention=rewire_scoped(graph['edges'],severity,seed,seed_scope)
                    else:
                        raise ValueError(f'Unsupported corruption kind: {corruption}')
                    corruption_seconds=time.perf_counter()-tick
                    for method in config['methods']:
                        tick=time.perf_counter()
                        if method=='bm25':
                            scores=lexical
                            cost={'candidate_expansions':0,'expanded_nodes':0,'edges_scanned':0}
                        else:
                            scores,cost=one_hop(lexical,edges,**config['one_hop'])
                        files,nodes=rank_files(scores,lexical,candidates)
                        if method=='bm25' and (files,nodes)!=clean_lexical:
                            raise ValueError('Lexical negative control changed under edge corruption')
                        row={'experiment_id':args.output_dir.name,'instance_id':task['instance_id'],'repo':task['repo'],
                             'base_commit':task['base_commit'],'method':method,'corruption':corruption,'severity':severity,'seed':seed,
                             'ranked_files':files,'top10_nodes':nodes[:10],'candidate_count':len(candidates),
                             'candidate_sha256':candidate_sha,'query_sha256':hashlib.sha256(task['problem_statement'].encode()).hexdigest(),
                             'intervention':intervention,'cost':cost,'index_seconds':index_seconds,
                             'shared_lexical_seconds':lexical_seconds,'ranking_seconds':time.perf_counter()-tick,
                             'corruption_seconds':corruption_seconds}
                        output.write(json.dumps(row,sort_keys=True)+'\n')
                    output.flush()
                print('Predicted',task['instance_id'],flush=True)
        # Labels are first loaded after all predictions have been saved.
        from localization_evaluation import metrics
        labels=index_unique([json.loads(l) for l in (args.mapping_dir/'mapping.jsonl').read_text().splitlines()])
        if set(labels) != set(selected_tasks):
            raise ValueError('Mapping labels differ from pilot task set')
        rows=[]
        with (args.output_dir/'per_task.jsonl').open('x') as output:
            for line in (args.output_dir/'predictions.jsonl').read_text().splitlines():
                row=json.loads(line)
                label=labels[row['instance_id']]
                if label['candidate_sha256'] != row['candidate_sha256']:
                    raise ValueError('Mapping label candidate pool differs from prediction')
                if label['status']!='passed' or not label.get('snapshot_tree',{}).get('verified_against'):
                    raise ValueError('Unverified source tree or gold preimage')
                gold=set(label['mapping']['gold_files'])
                row['metrics']=metrics(row['ranked_files'],gold)
                row['gold_files']=sorted(gold)
                row['mapping_status']=label['mapping']['status']
                row['unretrievable_gold_files']=label['mapping']['unretrievable_files']
                rows.append(row)
                output.write(json.dumps(row,sort_keys=True)+'\n')
        write_new_json(args.output_dir/'summary.json',aggregate(rows))
        manifest.update(status='completed',tasks=len(pilot['tasks']),observations=len(rows),
                        predictions_sha256=digest(args.output_dir/'predictions.jsonl'),
                        per_task_sha256=digest(args.output_dir/'per_task.jsonl'),
                        summary_sha256=digest(args.output_dir/'summary.json'),lexical_invariance_passed=True)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest.update(runtime_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        manifest['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_new_json(args.output_dir/'manifest.json',manifest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
