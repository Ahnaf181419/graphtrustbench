#!/usr/bin/env python3
"""Render tables only from committed, mutually consistent raw pilot artifacts."""
import argparse
import gzip
import hashlib
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from audit_competition_data import digest
from localization_evaluation import metrics
from pilot_integrity import conditions, index_unique, validate_selection
from run_localization_pilot import aggregate


def committed_json(path: Path, lines=False):
    root = Path(__file__).resolve().parents[2]
    source = Path(str(path)+'.gz') if path.suffix=='.jsonl' and Path(str(path)+'.gz').exists() else path
    relative = source.resolve().relative_to(root).as_posix()
    committed = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=root)
    if source.read_bytes() != committed:
        raise ValueError(f'Input differs from committed content: {path}')
    if source != path:
        committed = gzip.decompress(committed)
        if path.exists() and digest(path) != hashlib.sha256(committed).hexdigest():
            raise ValueError(f'Uncompressed input differs from committed bytes: {path}')
    return [json.loads(line) for line in committed.splitlines()] if lines else json.loads(committed)


def artifact_digest(path: Path) -> str:
    packed = Path(str(path)+'.gz')
    if path.suffix != '.jsonl' or not packed.exists():
        return digest(path)
    sha = hashlib.sha256()
    with gzip.open(packed,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            sha.update(chunk)
    return sha.hexdigest()


def observation_key(row):
    return tuple(row[k] for k in ('instance_id', 'method', 'corruption', 'severity', 'seed'))


def cell_listing(cells):
    """Render per-kind changed-cell counts with every driving cell named and rounded."""
    return '; '.join(
        f"{method}/{kind} changed {len(v['changed'])} of {v['cells']} corrupted task/seed/severity cells"
        + (f" ({', '.join(f'{t} seed {s} at {round(f*100)}% {kind}, {d:+.4f}' for t, s, f, d in v['changed'])})"
           if v['changed'] else '')
        for (method, kind), v in sorted(cells.items()))


def footprint_text(rows):
    """Explain why random corruption rarely reaches the traversal: report the reachable edge share."""
    clean = [r for r in rows if r['method'] == 'one_hop' and r['severity'] == 0
             and 'edges_incident_to_seeds' in r.get('cost', {})]
    if not clean:
        return None
    fractions = [r['cost']['edges_incident_to_seeds'] / r['cost']['edges_scanned'] for r in clean]
    middle = statistics.median(fractions)
    return (f"Traversal footprint: across {len(clean)} tasks, one-hop expansion can ever consult "
            f"{100*min(fractions):.1f}%–{100*max(fractions):.1f}% of graph edges "
            f"(median {100*middle:.1f}%); corruption of the remaining edges cannot reach the ranking by construction.")


def sign_test(deltas: list[float]) -> float | None:
    """Exact two-sided sign test on nonzero deltas; None when every delta is zero."""
    nonzero = [d for d in deltas if d]
    if not nonzero:
        return None
    negatives = sum(1 for d in nonzero if d < 0)
    tail = min(negatives, len(nonzero) - negatives)
    probability = sum(math.comb(len(nonzero), k) for k in range(tail + 1)) / 2 ** len(nonzero)
    return min(1.0, 2 * probability)


def bootstrap_deltas(rows: list[dict], resamples: int = 10000, seed: int = 2026) -> dict:
    """Paired bootstrap over tasks of the mean delta versus clean; seeds are averaged first."""
    clean = {(r['instance_id'], r['method']): r['metrics']['recall@5'] for r in rows if r['severity'] == 0}
    per_task = defaultdict(list)
    for r in rows:
        if r['severity'] == 0:
            continue
        delta = r['metrics']['recall@5'] - clean[(r['instance_id'], r['method'])]
        per_task[(r['method'], r['corruption'], r['severity'])].append((r['instance_id'], delta))
    task_means = {}
    for key, observations in per_task.items():
        by_task = defaultdict(list)
        for task_id, delta in observations:
            by_task[task_id].append(delta)
        task_means[key] = [sum(values) / len(values) for _, values in sorted(by_task.items())]
    rng = random.Random(seed)
    result = {}
    for key in sorted(task_means):
        values = task_means[key]
        count = len(values)
        draws = []
        # A per-group stream keeps one group's interval independent of the other groups present,
        # so adding a row type can never silently move an existing interval.
        group_rng = random.Random(f'{seed}:{key}')
        for _ in range(resamples):
            draws.append(sum(values[group_rng.randrange(count)] for _ in range(count)) / count)
        draws.sort()
        low = draws[int(0.025 * len(draws))]
        high = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
        result[key] = {'delta': sum(values) / count, 'low': low, 'high': high, 'tasks': count,
                       'degraded': sum(1 for v in values if v < 0), 'improved': sum(1 for v in values if v > 0),
                       'p_sign': sign_test(values)}
    return result


def interval_text(rows: list[dict]) -> str:
    """Uncertainty table: bootstrap interval and sign test beside every non-zero aggregate."""
    intervals = bootstrap_deltas(rows)
    lines = ['| Method | Intervention | Severity | Δ Recall@5 | 95% CI (paired bootstrap over tasks) | Tasks degraded/improved | Exact sign test p |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for key in sorted(intervals):
        info = intervals[key]
        if info['delta'] == 0 and info['low'] == 0 and info['high'] == 0:
            continue
        method, kind, severity = key
        p_value = 'n/a' if info['p_sign'] is None else f"{info['p_sign']:.4f}"
        lines.append(f"| {method} | {kind} | {severity:.0%} | {info['delta']:+.4f} | "
                     f"[{info['low']:+.4f}, {info['high']:+.4f}] | "
                     f"{info['degraded']}/{info['improved']} of {info['tasks']} | {p_value} |")
    if len(lines) == 2:
        return 'No aggregate differs from clean.'
    return ('Paired bootstrap over tasks (10,000 resamples; corruption seeds averaged within task first, '
            'so seeds are never treated as independent tasks). The exact sign test is two-sided and '
            'uncorrected for multiplicity; it is exploratory, not confirmatory.\n\n' + '\n'.join(lines))


def degradation_cells(rows):
    """Count Recall@5 changes per corrupted cell so aggregates cannot hide a single driver."""
    clean = {(r['instance_id'], r['method']): r['metrics']['recall@5'] for r in rows if r['severity'] == 0}
    result = {}
    for r in rows:
        if r['severity'] == 0:
            continue
        slot = result.setdefault((r['method'], r['corruption']), {'cells': 0, 'changed': []})
        slot['cells'] += 1
        delta = r['metrics']['recall@5'] - clean[(r['instance_id'], r['method'])]
        if delta:
            slot['changed'].append((r['instance_id'], r['seed'], r['severity'], delta))
    for slot in result.values():
        slot['changed'].sort()
    return result


def validate_observations(manifest, pilot, predictions, rows, mappings):
    """Check the full Cartesian design, label identity and prediction/evaluation boundary."""
    tasks = index_unique(pilot['tasks'])
    labels = index_unique(mappings)
    if set(labels) != set(tasks) or manifest['tasks'] != len(tasks):
        raise ValueError('Mapping task set differs from pilot')
    expected = {(t, m, c, f, s) for t in tasks for m in manifest['configuration']['methods']
                for c, f, s in conditions(manifest['configuration'])}
    predicted = {observation_key(r): r for r in predictions}
    evaluated = {observation_key(r): r for r in rows}
    if (len(predictions) != len(expected) or len(rows) != len(expected)
            or set(predicted) != expected or set(evaluated) != expected
            or manifest['observations'] != len(expected)):
        raise ValueError('Incomplete or duplicate observation matrix')
    for task_id, label in labels.items():
        if (label['status'] != 'passed' or label['mapping']['preimage_errors']
                or not label.get('snapshot_tree', {}).get('verified_against')
                or any(label[k] != tasks[task_id][k] for k in ('repo', 'base_commit'))):
            raise ValueError('Unverified or mismatched mapping labels')
    evaluation_fields = {'metrics', 'gold_files', 'mapping_status', 'unretrievable_gold_files'}
    for key, row in evaluated.items():
        task, label = tasks[row['instance_id']], labels[row['instance_id']]
        if {k: v for k, v in row.items() if k not in evaluation_fields} != predicted[key]:
            raise ValueError('Prediction/evaluation record mismatch')
        if (row['experiment_id'] != manifest['experiment_id']
                or any(row[k] != task[k] for k in ('repo', 'base_commit'))):
            raise ValueError('Observation identity mismatch')
        gold = sorted(set(label['mapping']['gold_files']))
        if (row['gold_files'] != gold or row['mapping_status'] != label['mapping']['status']
                or row['unretrievable_gold_files'] != label['mapping']['unretrievable_files']
                or row['metrics'] != metrics(row['ranked_files'], set(gold))):
            raise ValueError('Stored labels or metrics differ from verified mapping')
        if len(row['ranked_files']) != len(set(row['ranked_files'])):
            raise ValueError('Duplicate ranked files')
        if ('candidate_sha256' in label and
                (row['candidate_sha256'] != label['candidate_sha256'] or
                 row['candidate_count'] != label['candidate_coverage']['mapped_nodes'])):
            raise ValueError('Candidate pool differs from mapping')
    for task_id in tasks:
        group = [r for r in predictions if r['instance_id'] == task_id]
        for field in ('candidate_sha256', 'query_sha256', 'candidate_count'):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f'Unmatched retrieval input: {task_id} {field}')
        lexical = [r for r in group if r['method'] == 'bm25']
        if any((r['ranked_files'], r['top10_nodes']) !=
               (lexical[0]['ranked_files'], lexical[0]['top10_nodes']) for r in lexical):
            raise ValueError('Lexical negative control is not invariant')


def load_verified_run(audit_path, run_dir, mapping_dir, pilot_path, config_path):
    audit = committed_json(audit_path)
    pilot = committed_json(pilot_path)
    manifest = committed_json(run_dir/'manifest.json')
    mapping_manifest = committed_json(mapping_dir/'manifest.json')
    if (manifest['mapping_sha256'] != digest(mapping_dir/'mapping.jsonl')
            or mapping_manifest.get('mapping_sha256') != manifest['mapping_sha256']):
        raise ValueError('Mapping artifact does not belong to this run')
    if manifest['status'] != 'completed' or mapping_manifest['status'] != 'passed':
        raise ValueError('Passing mapping and completed run required')
    config = committed_json(config_path)
    if manifest['config_sha256'] != digest(config_path) or manifest['configuration'] != config:
        raise ValueError('Configuration differs from run manifest')
    validate_selection(pilot, audit)
    for name in ('localization_data.py', 'localization_evaluation.py'):
        if manifest['source_hashes'][name] != mapping_manifest['source_hashes'][name]:
            raise ValueError('Run/mapping source versions differ')
    if mapping_manifest['tasks'] != len(pilot['tasks']) or mapping_manifest['preimage_passed'] != len(pilot['tasks']):
        raise ValueError('Mapping preflight task count mismatch')
    for record in (manifest, mapping_manifest):
        if (record['audit_sha256'] != digest(audit_path) or record['pilot_sha256'] != digest(pilot_path)
                or record['dataset_fingerprint'] != audit['dataset_fingerprint']):
            raise ValueError('Run/mapping/audit/pilot provenance mismatch')
    for field, path in [('predictions_sha256', run_dir/'predictions.jsonl'),
                        ('per_task_sha256', run_dir/'per_task.jsonl'),
                        ('summary_sha256', run_dir/'summary.json'),
                        ('mapping_manifest_sha256', mapping_dir/'manifest.json')]:
        # Legacy 0.1 runs did not record the final two hashes; their committed
        # summary is still recomputed and their mapping inputs checked above.
        if field not in manifest and field in ('summary_sha256', 'mapping_manifest_sha256'):
            if manifest['configuration']['protocol_version'] == 'graphtrust-pilot-0.1':
                continue
        if manifest.get(field) != artifact_digest(path):
            raise ValueError(f'Artifact hash mismatch: {field}')
    predictions = committed_json(run_dir/'predictions.jsonl', True)
    rows = committed_json(run_dir/'per_task.jsonl', True)
    mappings = committed_json(mapping_dir/'mapping.jsonl', True)
    if config['protocol_version'] != 'graphtrust-pilot-0.1':
        bindings = {r['instance_id']: r['candidate_sha256'] for r in mappings}
        if bindings != mapping_manifest.get('candidate_sha256_by_task'):
            raise ValueError('Mapping candidate bindings differ from manifest')
    validate_observations(manifest, pilot, predictions, rows, mappings)
    summary = aggregate(rows)
    if summary != committed_json(run_dir/'summary.json'):
        raise ValueError('Stored summary differs from raw observations')
    return audit, pilot, manifest, mappings, rows, summary


def render_reports(audit, pilot, manifest, mappings, rows, summary, sha):
    n = len(mappings)
    mapped = sum(r['candidate_coverage']['mapped_nodes'] for r in mappings)
    nodes = sum(r['candidate_coverage']['graph_nodes'] for r in mappings)
    gold = sum(len(r['mapping']['gold_files']) for r in mappings)
    absent = sum(len(r['mapping']['unretrievable_files']) for r in mappings)
    aliases = sum(len(r['candidate_coverage'].get('alias_mapped_node_ids', [])) for r in mappings)
    ambiguous = sum(len(r['candidate_coverage'].get('ambiguous_alias_node_ids', [])) for r in mappings)
    conflicts = sum(len(r['candidate_coverage'].get('conflicting_exact_node_ids', [])) for r in mappings)
    statuses = dict(Counter(r['mapping']['status'] for r in mappings))
    distribution = dict(Counter(r['repo'] for r in mappings))
    coverage = (f'Candidate mapping: {mapped:,}/{nodes:,} nodes ({100*mapped/nodes:.2f}%). '
                f'Source-verified alias bindings: {aliases:,}; unresolved ambiguous aliases: {ambiguous:,}. '
                f'Exact-ID body conflicts encountered: {conflicts:,}. '
                f'Changed-file coverage: {gold-absent}/{gold}; all {absent} unavailable files remain in recall denominators.')
    lines = ['# Competition dataset audit', '', f'Raw-output checkpoint: `{sha}`.', '',
             f"Status: **{audit['status']}**. {audit['usable_tasks']}/{audit['total_tasks']} structurally usable tasks across {audit['total_repositories']} repositories.", '',
             '| Repository | Tasks |', '| --- | ---: |']
    lines += [f'| {repo} | {count} |' for repo, count in audit['repository_distribution'].items()]
    external = sum(t.get('snapshot_audit', {}).get('potentially_external_links', 0) for t in audit['tasks'])
    lines += ['', f"- {audit['unique_graphs']} distinct graphs; {audit['unique_base_commits']} distinct repository/base-commit pairs.",
              f"- {audit['unique_graph_nodes']:,} graph nodes; {audit['unique_graph_edges']:,} edges, counted once per graph. Edge types: `{audit['edge_types']}`.",
              f"- Missing required assets: `{audit['missing_files']}`; tasks with graph errors: {audit['tasks_with_graph_errors']}.",
              f"- Consumed asset bytes: {sum(a['bytes'] for a in audit['assets'].values()):,}; tasks.jsonl and auxiliary wheels are additional.",
              f"- Audit runtime: {audit['provenance']['runtime_seconds']:.2f} seconds; potentially external archive links: {external}. Links are not followed.", '',
              f'Upstream worktree-tree identity and patch preimages are verified for these {n} pilot tasks. This verification has not been extended to the other {audit["total_tasks"]-n} tasks.', '', coverage, '',
              f'Mapping statuses: `{statuses}`. Symbol labels use the smallest available enclosing definition; class stubs do not establish exact function relevance. File-level evaluation is primary.', '',
              f'Pilot repository distribution: `{distribution}`. Sampling seed: {pilot["seed"]}; selection: {pilot["selection"]}.', '',
              f"Shared repository/base-commit groups, which must stay within one future split: `{audit['shared_base_commits']}`.", '',
              f"Dataset fingerprint: `{audit['dataset_fingerprint']}`. Graphs have symbol IDs but no explicit source positions; supplied unperturbed edges are not independently established as semantically correct."]
    audit_text = '\n'.join(lines)
    peak = f"{manifest['peak_rss_kib']:,} KiB" if 'peak_rss_kib' in manifest else 'not recorded in this manifest'
    vary = any('corruption' in row for row in summary['summaries'])
    lines = ['# Localization development pilot', '',
             f"Raw-output checkpoint: `{sha}`. Run: `{manifest['experiment_id']}`. Protocol: `{manifest['configuration']['protocol_version']}`.", '',
             '**Exploratory development results; no confirmatory superiority or agent repair claim.**', '',
             f"{manifest['tasks']} tasks; {manifest['observations']} observations. {len(manifest['configuration']['seeds'])} corruption seeds are technical repeats, averaged within each task before task means. Runtime: {manifest['runtime_seconds']:.2f} seconds; peak process RSS: {peak}.", '',
             ('| Method | Intervention | ' if vary else '| Method | ') + 'Corrupted edges | Recall@1 | Recall@5 | Recall@10 | MRR | Success@5 | Δ Recall@5 vs clean |',
             ('| --- | --- | ' if vary else '| --- | ') + '---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['summaries']:
        if row['repo'] == 'all':
            m = row['mean']
            kind = f"{row['corruption']} | " if vary else ''
            lines.append(f"| {row['method']} | {kind}{100*row['severity']:.0f}% | {m['recall@1']:.4f} | {m['recall@5']:.4f} | {m['recall@10']:.4f} | {m['mrr']:.4f} | {m['success@5']:.4f} | {row['mean_delta_from_clean']['recall@5']:+.4f} |")
    clean = {r['method']: r['mean'] for r in summary['summaries'] if r['repo'] == 'all' and r['severity'] == 0}
    footprint = footprint_text(rows)
    per_cell = cell_listing(degradation_cells(rows))
    lines += ['', f"Clean one-hop minus BM25: Recall@5 {clean['one_hop']['recall@5']-clean['bm25']['recall@5']:+.4f}; MRR {clean['one_hop']['mrr']-clean['bm25']['mrr']:+.4f}.", '']
    if footprint:
        lines += [footprint, '']
    lines += [f"Per-cell check versus clean: {per_cell}. The Δ Recall@5 values in the table above are produced only by the listed cells; every unlisted cell has unchanged Recall@5 but may have a different ranking. Gains and losses can cancel in the aggregate.", '',
              'The small, uneven development sample cannot establish general superiority. The intervals below are exploratory and uncorrected for multiplicity; no significance claim is made. Candidate pools and returned-file cutoffs are shared; compute and token budgets are not matched.', '',
              '## Clean results by repository', '', '| Repository | Tasks | Method | Recall@5 | MRR |', '| --- | ---: | --- | ---: | ---: |']
    for row in summary['summaries']:
        if row['repo'] != 'all' and row['severity'] == 0:
            lines.append(f"| {row['repo']} | {row['tasks']} | {row['method']} | {row['mean']['recall@5']:.4f} | {row['mean']['mrr']:.4f} |")
    lines += ['', '## Uncertainty', '', interval_text(rows), '']
    seed_header = (('| Method | Intervention | Corrupted edges | Seed | Recall@5 |',
                    '| --- | --- | ---: | ---: | ---: |') if vary else
                   ('| Method | Missing edges | Seed | Recall@5 |',
                    '| --- | ---: | ---: | ---: |'))
    lines += ['', '## Variation across corruption seeds', '', *seed_header]
    for row in summary['seed_means']:
        if row['method'] == 'one_hop' and row['severity']:
            kind = f"{row['corruption']} | " if vary else ''
            lines.append(f"| one_hop | {kind}{row['severity']*100:.0f}% | {row['seed']} | {row['recall@5']:.4f} |")
    lines += ['', '## Coverage and failure cases', '', coverage, '', 'Unavailable changed files:', '']
    lines += [f"- `{r['instance_id']}`: `{path}`" for r in mappings for path in r['mapping']['unretrievable_files']]
    if not absent:
        lines.append('None.')
    failed = [t for t in sorted({r['instance_id'] for r in rows})
              if all(r['metrics']['success@10'] == 0 for r in rows if r['instance_id'] == t and r['severity'] == 0)]
    lines += ['', 'Tasks without a clean top-10 hit for either method: ' + (', '.join(f'`{t}`' for t in failed) or 'none') + '.', '',
              'Lexical rankings and query/candidate hashes are invariant across conditions, checked from raw predictions. Prediction/evaluation records, label identity, the complete condition matrix and recomputed metrics/summary all pass validation.', '',
              '## Provenance and limits', '',
              f"Run code checkpoint: `{manifest['git_sha']}`. Per-file source hashes pin the executed working state. Python: `{manifest['python']}`; platform: `{manifest['platform']}`.", '',
              'Predictions are written before the separate evaluation labels are loaded. Retrieval uses only the issue statement and pre-fix candidates. The manifests record commands, configuration, asset fingerprints and source hashes. No model inference is part of this pilot.', '',
              'A compatible issue-query embedding encoder remains unavailable, so no semantic baseline is reported.']
    return audit_text, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('paper/data/pilot_2026-11-06.json'))
    parser.add_argument('--config', type=Path, default=Path('paper/config/pilot.json'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mapping-dir', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--results-output', type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists() or args.results_output.exists() or args.audit_output.resolve() == args.results_output.resolve():
        parser.error('Refusing to overwrite reports or share an output path')
    data = load_verified_run(args.audit, args.run_dir, args.mapping_dir, args.pilot, args.config)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    reports = render_reports(*data, sha)
    for path, report in zip((args.audit_output, args.results_output), reports):
        links = [('raw audit', args.audit), ('pilot selection', args.pilot), ('configuration', args.config),
                 ('run manifest', args.run_dir/'manifest.json'), ('mapping manifest', args.mapping_dir/'manifest.json')]
        report += '\n\nSources: ' + ', '.join(f'[{name}]({os.path.relpath(target, path.parent)})' for name, target in links) + '.\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as output:
            output.write(report)


if __name__ == '__main__':
    main()
