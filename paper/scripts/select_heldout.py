#!/usr/bin/env python3
"""Select a deterministic held-out split that never shares a repository/base-commit group with development."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from audit_competition_data import digest, write_new_json


def select(tasks: list[dict], excluded_ids: set[str], limit: int, seed: int) -> list[dict]:
    """Round-robin across repositories, refusing any base commit already used by dev or selection."""
    if limit <= 0:
        raise ValueError('limit must be positive')
    dev_commits = {f"{t['repo']}@{t['base_commit']}" for t in tasks if t['instance_id'] in excluded_ids}
    used = set(dev_commits)
    by_repo = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t['instance_id']):
        if t['instance_id'] in excluded_ids or f"{t['repo']}@{t['base_commit']}" in dev_commits:
            continue
        by_repo[t['repo']].append(t)
    rng = random.Random(seed)
    selected = []
    while len(selected) < limit:
        progressed = False
        for repo in sorted(by_repo):
            if len(selected) == limit:
                break
            pool = [t for t in by_repo[repo] if f"{t['repo']}@{t['base_commit']}" not in used]
            if not pool:
                continue
            pick = rng.choice(sorted(pool, key=lambda t: t['instance_id']))
            used.add(f"{pick['repo']}@{pick['base_commit']}")
            by_repo[repo].remove(pick)
            selected.append(pick)
            progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda t: t['instance_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing selection')
    audit = json.loads(args.audit.read_text())
    pilot = json.loads(args.pilot.read_text())
    if audit['status'] != 'passed_schema_preflight' or pilot['dataset_fingerprint'] != audit['dataset_fingerprint']:
        parser.error('A passing audit for this exact dataset is required')
    chosen = select([t for t in audit['tasks'] if t['schema_usable']],
                    {t['instance_id'] for t in pilot['tasks']}, args.limit, args.seed)
    fields = ('instance_id', 'repo', 'base_commit', 'graph_path', 'embedding_path', 'snapshot_path')
    write_new_json(args.output, {'status': audit['status'], 'seed': args.seed,
        'selection': 'repository round-robin, one base commit per draw; excludes every development task and every task sharing a development base commit; no gold used',
        'dataset_fingerprint': audit['dataset_fingerprint'], 'tasks_sha256': pilot['tasks_sha256'],
        'tasks': [{k: t[k] for k in fields if k in t} for t in chosen]})
    print(f'Selected {len(chosen)} held-out tasks across {len({t["repo"] for t in chosen})} repositories -> {args.output}')


if __name__ == '__main__':
    main()
