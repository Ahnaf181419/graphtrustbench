"""Reachability and delivery analysis: why graph corruption usually cannot move a shallow ranking.

Answers three questions from committed artifacts only:
  1. What fraction of a graph can a one-hop traversal ever consult?
  2. How often does a uniform random corruption draw fail to touch that reachable set at all?
  3. Does the corruption that *does* land predict whether the ranking changes?
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from audit_competition_data import digest, read_tasks, write_new_json
from localization_data import extract_candidates, read_snapshot
from localization_retrieval import bm25, delete_edges, rewire_edges, seed_nodes
from render_pilot_report import committed_json, load_verified_run, artifact_digest


def outcome_changes(clean: dict, corrupted: dict) -> dict:
    return {'recall5_changed': clean['metrics']['recall@5'] != corrupted['metrics']['recall@5'],
            'top5_order_changed': clean['ranked_files'][:5] != corrupted['ranked_files'][:5],
            'top5_set_changed': set(clean['ranked_files'][:5]) != set(corrupted['ranked_files'][:5]),
            'full_ranking_changed': clean['ranked_files'] != corrupted['ranked_files']}


def point_biserial(values: list[float], outcomes: list[bool]) -> float | None:
    import math
    if len(values) != len(outcomes) or not values:
        raise ValueError('Matched nonempty vectors required')
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Finite values required')
    x = [float(v) for v in outcomes]
    mx, my = sum(x)/len(x), sum(values)/len(values)
    xx = sum((v-mx)**2 for v in x)
    yy = sum((v-my)**2 for v in values)
    if not xx or not yy:
        return None
    return sum((a-mx)*(b-my) for a,b in zip(x,values))/(xx*yy)**0.5


def zero_selection_probability(total: int, scope: int, selected: int) -> float:
    if any(type(v) is not int for v in (total,scope,selected)) or not 0 <= scope <= total or not 0 <= selected <= total:
        raise ValueError('Invalid finite population')
    if selected > total-scope:
        return 0.0
    result = 1.0
    for i in range(scope):
        result *= (total-selected-i)/(total-i)
    return result


def edge_identity(edge: dict) -> tuple:
    return (edge['source'], edge['target'], edge.get('type', ''), str(edge.get('key', 0)))


def scope_edge_count(edges: list[dict], seeds: set[str]) -> int:
    """Edges whose source is a seed: the only ones a one-hop traversal can consult."""
    return sum(1 for edge in edges if edge['source'] in seeds)


def delivered_count(scope: list[dict], mutated: list[dict]) -> int:
    """Net change in the reachable relation multiset: how much of the scope was corrupted.

    Measured per (source, target) relation rather than per edge identity, so a rewiring that is
    later undone by another rewiring counts as the net change it is. Zero means the traversal
    would read exactly the same neighbourhood.
    """
    before = Counter((edge['source'], edge['target']) for edge in scope)
    after = Counter((edge['source'], edge['target']) for edge in mutated)
    return sum(abs(before[key] - after[key]) for key in before)



def _intervention(kind: str, edges: list[dict], fraction: float, seed: int):
    """The intervention for one condition, with its record."""
    if kind == 'missing':
        return delete_edges(edges, fraction, seed)
    return rewire_edges(edges, fraction, seed)

def zero_delivery_probability(scope_size: int, fraction: float) -> float:
    """Legacy independent-selection approximation; not the fixed-size intervention probability."""
    return (1 - fraction) ** scope_size if scope_size else 1.0


def delivered_dose_bins(cells: list[dict], threshold: float = 0.3) -> dict:
    """Split cells that received corruption by delivered share and count how many changed."""
    landed = [cell for cell in cells if cell['delivered'] > 0]
    bins = {'at_or_above_threshold': {'cells': 0, 'changed': 0},
            'below_threshold': {'cells': 0, 'changed': 0}}
    for cell in landed:
        key = 'at_or_above_threshold' if cell['delivered_fraction'] >= threshold else 'below_threshold'
        bins[key]['cells'] += 1
        bins[key]['changed'] += cell['changed']
    bins['no_corruption_delivered'] = {'cells': len(cells) - len(landed),
                                        'changed': sum(cell['changed'] for cell in cells if cell['delivered'] == 0)}
    return bins


def reachability_contingency(cells: list[dict]) -> list[dict]:
    """Cross-tabulate whether corruption reached the reachable neighbourhood against each outcome.

    Row 0 is the set of cells in which no reachable edge was corrupted; row 1 the set in which
    at least one was. This is the mechanism statement: an undelivered corruption cannot change
    any outcome, and a delivered one mostly changes the ranking rather than the accuracy.
    """
    rows = []
    for label, predicate in (('not reached', lambda cell: cell['delivered'] == 0),
                             ('reached', lambda cell: cell['delivered'] > 0)):
        subset = [cell for cell in cells if predicate(cell)]
        row = {'group': label, 'cells': len(subset)}
        for outcome in ('recall5_changed', 'top5_set_changed', 'top5_order_changed',
                        'full_ranking_changed'):
            row[outcome] = sum(1 for cell in subset if cell['outcomes'].get(outcome))
        rows.append(row)
    return rows


def separators(tasks: list[dict], changed: dict[str, int]) -> dict:
    """Do any recorded graph features separate tasks whose ranking changed from those that did not?"""
    features = ('scope_edges', 'distinct_seed_neighbours', 'candidates')
    result = {}
    for feature in features:
        changed_values = [t[feature] for t in tasks if changed.get(t['instance_id'], 0) > 0 and feature in t]
        stable_values = [t[feature] for t in tasks if changed.get(t['instance_id'], 0) == 0 and feature in t]
        if not changed_values or not stable_values:
            continue
        mean_changed = sum(changed_values) / len(changed_values)
        mean_stable = sum(stable_values) / len(stable_values)
        all_values = changed_values + stable_values
        overall = sum(all_values) / len(all_values)
        spread = (sum((v - overall) ** 2 for v in all_values) / len(all_values)) ** 0.5
        result[feature] = {
            'changed_mean': mean_changed, 'unchanged_mean': mean_stable,
            'changed_tasks': len(changed_values), 'unchanged_tasks': len(stable_values),
            'point_biserial': point_biserial(changed_values + stable_values, [True]*len(changed_values)+[False]*len(stable_values))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--mapping-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing analysis')
    audit, pilot, manifest, mappings, rows, summary = load_verified_run(
        args.audit,args.run_dir,args.mapping_dir,args.pilot,args.config)
    if digest(args.data_root/'tasks.jsonl') != pilot['tasks_sha256']:
        raise ValueError('Task corpus differs from frozen selection')
    problems = {t['instance_id']: t['problem_statement'] for t in read_tasks(args.data_root/'tasks.jsonl')}
    config = manifest['configuration']
    bm25_params = config['bm25']
    seed_count = config['one_hop']['seed_count']
    recorded = {}
    for row in rows:
        if row['severity'] and row['method'] == 'one_hop':
            recorded[(row['instance_id'], row['corruption'], row['severity'], row['seed'])] = row['intervention']
    clean = {(r['instance_id'], r['method']): r for r in rows if r['severity'] == 0}
    kinds = sorted({r['corruption'] for r in rows if r['severity'] and not r['corruption'].endswith('targeted')})
    severities = sorted({r['severity'] for r in rows if r['severity']})
    seeds = sorted({r['seed'] for r in rows if r['severity']})

    tasks_out = []
    cells = []
    for item in pilot['tasks']:
        task_id = item['instance_id']
        for field in ('graph_path','snapshot_path'):
            if digest(args.data_root/item[field]) != audit['assets'][item[field]]['sha256']:
                raise ValueError(f'Changed asset: {item[field]}')
        graph = json.loads((args.data_root / item['graph_path']).read_text())
        candidates, _ = extract_candidates(read_snapshot(args.data_root / item['snapshot_path']), graph['nodes'])
        lexical = bm25(problems[task_id], candidates, **bm25_params)
        seeds_used = set(seed_nodes(lexical, seed_count))
        scope = [edge for edge in graph['edges'] if edge['source'] in seeds_used]
        size = len(scope)
        per_task = {'instance_id': task_id, 'repo': item['repo'], 'graph_edges': len(graph['edges']),
                    'scope_edges': size,
                    'scope_fraction': size / len(graph['edges']) if graph['edges'] else 0.0,
                    'distinct_seed_neighbours': len({edge['target'] for edge in scope}),
                    'candidates': len(candidates),
                    'zero_delivery': {}}
        for fraction in severities:
            expected = zero_selection_probability(len(graph['edges']), size, int(len(graph['edges'])*fraction))
            observed = 0
            for seed in seeds:
                for kind in kinds:
                    kept = delete_edges(graph['edges'], fraction, seed)[0] if kind == 'missing' \
                        else rewire_edges(graph['edges'], fraction, seed)[0]
                    if delivered_count(scope, kept) == 0:
                        observed += 1
            trials = len(seeds) * len(kinds)
            per_task['zero_delivery'][f'{fraction}'] = {
                'zero_selected_scope_probability': expected, 'observed_fraction': observed / trials if trials else 0.0,
                'trials': trials}
        for kind in kinds:
            for fraction in severities:
                for seed in seeds:
                    kept, info = _intervention(kind, graph['edges'], fraction, seed)
                    # The recomputed intervention must match the one the run actually recorded.
                    expected = recorded.get((task_id, kind, fraction, seed))
                    if expected is not None and info != expected:
                        raise ValueError(f'Recomputed {kind} intervention differs from the run for '
                                         f'{task_id} at severity {fraction} seed {seed}')
                    delivered = delivered_count(scope, kept)
                    row = next(r for r in rows if r['instance_id'] == task_id and r['method'] == 'one_hop'
                               and r['corruption'] == kind and r['severity'] == fraction and r['seed'] == seed)
                    cells.append({'instance_id': task_id, 'kind': kind, 'severity': fraction, 'seed': seed,
                                  'scope_edges': size, 'delivered': delivered,
                                  'delivered_fraction': delivered / size if size else 0.0,
                                  'changed': int(outcome_changes(clean[(task_id, 'one_hop')], row)['recall5_changed']),
                                  'outcomes': outcome_changes(clean[(task_id, 'one_hop')], row)})
        tasks_out.append(per_task)

    summary = {}
    for kind in kinds:
        subset = [cell for cell in cells if cell['kind'] == kind]
        summary[kind] = {'cells': len(subset), 'changed': sum(cell['changed'] for cell in subset),
                          'dose_bins': delivered_dose_bins(subset),
                          'outcome_counts': {key:sum(c['outcomes'][key] for c in subset) for key in subset[0]['outcomes']}}
    expected_cells = len(pilot['tasks']) * len(kinds) * len(severities) * len(seeds)
    if len(cells) != expected_cells:
        raise ValueError(f'Expected {expected_cells} analysed cells, produced {len(cells)}')
    changed_counts = defaultdict(int)
    for cell in cells:
        changed_counts[cell['instance_id']] += cell['changed']
    separator_report = separators(tasks_out, changed_counts)
    write_new_json(args.output, {
        'status': 'analysis_only', 'analysis_version': '0.5', 'changed_definition': 'recall5_changed',
        'run_id': args.run_dir.name,
        'input_hashes': {str(p):artifact_digest(p) for p in (args.audit,args.pilot,args.config,args.run_dir/'per_task.jsonl',args.run_dir/'manifest.json')},
        'analysis_source_sha256': digest(Path(__file__)),
        'per_task': tasks_out,
        'summary_by_intervention': summary,
        'task_separators': separator_report,
        'cells': cells,
        'interpretation_limits': ['Scope edges are those leaving the clean lexical seeds; a shallow one-hop traversal consults no other edge.',
                                   'Delivered corruption counts scope edges whose identity is absent from the mutated graph.',
                                   'Delivery is necessary for a cell to change; this analysis does not establish sufficiency.',
                                   'One method (one_hop) is analysed because BM25 never consults the graph.']})
    print(f"Wrote {args.output} ({len(cells)} cells, {len(tasks_out)} tasks)")


if __name__ == '__main__':
    main()
