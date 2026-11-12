"""Shared checks for frozen pilot identities, conditions and source bindings."""
import hashlib
import json
import math
from pathlib import PurePosixPath


def candidate_digest(candidates):
    return hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()


def index_unique(rows, field='instance_id'):
    indexed = {row[field]: row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f'Empty or duplicate {field} records')
    return indexed


def validate_selection(pilot, audit):
    if (pilot['status'] != 'passed_schema_preflight' or
            audit['status'] != 'passed_schema_preflight' or not audit['scan_snapshots'] or
            pilot['dataset_fingerprint'] != audit['dataset_fingerprint'] or
            pilot['tasks_sha256'] != audit['tasks_sha256']):
        raise ValueError('Full passing audit required for this exact pilot dataset')
    selected = index_unique(pilot['tasks'])
    audited = index_unique(audit['tasks'])
    for task_id, task in selected.items():
        reference = audited.get(task_id)
        if reference is None or not reference['schema_usable']:
            raise ValueError(f'Unaudited task: {task_id}')
        for field in ('repo', 'base_commit', 'graph_path', 'snapshot_path', 'embedding_path'):
            if task[field] != reference[field]:
                raise ValueError(f'Pilot/audit identity mismatch: {task_id} {field}')
        for field in ('graph_path', 'snapshot_path', 'embedding_path'):
            path = PurePosixPath(task[field])
            if path.is_absolute() or '..' in path.parts or task[field] not in audit['assets']:
                raise ValueError(f'Unsafe or unaudited asset: {task[field]}')
    return selected


def validate_mapping_sources(manifest, source_hashes):
    for name in ('localization_data.py', 'localization_evaluation.py'):
        if manifest.get('source_hashes', {}).get(name) != source_hashes[name]:
            raise ValueError(f'Rebuild mapping after source change: {name}')


def conditions(config):
    if (config['methods'] != ['bm25', 'one_hop'] or config['cutoffs'] != [1, 5, 10]
            or config['query'] != 'problem_statement_only'
            or config['primary_unit'] != 'file'):
        raise ValueError('Unsupported pilot protocol')
    corruption = config['corruption']
    # The protocol version fixes which interventions exist, so a config cannot quietly gain or
    # lose an intervention while keeping its version label.
    allowed_by_protocol = {
        'graphtrust-pilot-0.1': ['missing'],
        'graphtrust-pilot-0.2': ['missing', 'spurious'],
        'graphtrust-pilot-0.3': ['missing', 'spurious'],
        'graphtrust-pilot-0.4': ['missing', 'spurious', 'missing_targeted', 'spurious_targeted'],
    }
    allowed = allowed_by_protocol.get(config.get('protocol_version'))
    if allowed is None:
        raise ValueError('Unsupported pilot protocol')
    if isinstance(corruption, str):
        # 'edge_deletion' is the historical name for the deletion intervention.
        if corruption == 'edge_deletion':
            kinds = ['missing'] if 'missing' in allowed else None
        else:
            kinds = [corruption] if corruption in allowed else None
    elif (isinstance(corruption, list) and corruption
            and all(kind in allowed for kind in corruption)
            and len(set(corruption)) == len(corruption)):
        kinds = corruption
    else:
        kinds = None
    if kinds is None:
        raise ValueError('Unsupported pilot protocol')
    seeds, severities = config['seeds'], config['severities']
    if (not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or not severities or len(set(severities)) != len(severities)
            or any(type(f) not in (int, float) or not math.isfinite(f) or not 0 < f <= 1
                   for f in severities)):
        raise ValueError('Nonempty, unique seeds and finite positive severities required')
    bm25, hop = config['bm25'], config['one_hop']
    values = (bm25['k1'], bm25['b'], hop['restart'])
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or bm25['k1'] <= 0 or not 0 <= bm25['b'] <= 1 or not 0 <= hop['restart'] <= 1
            or any(type(hop[k]) is not int or hop[k] < 1 for k in ('seed_count', 'edge_budget'))):
        raise ValueError('Invalid retrieval parameters or budgets')
    return [('clean', 0.0, 0)] + [(kind, f, s) for kind in kinds for f in severities for s in seeds]
