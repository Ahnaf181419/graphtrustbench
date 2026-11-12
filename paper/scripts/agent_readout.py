"""Build paired read-out contexts for a downstream agent experiment.

The pairing is deliberately strict: a task contributes only when the retrieved file set actually
differs between the clean and the corrupted condition, so any difference in the model's answer is
attributable to the context change rather than to sampling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

PROMPT_TEMPLATE = """You are locating which single file in a repository must be edited to fix an issue.

Issue:
{issue}

Candidate files, retrieved for you. Exactly one of them may be the file to edit.

{files}

Answer with the path of the single file to edit, copied exactly from the list above, and nothing else."""

FILE_BLOCK = "--- {path} (truncated) ---\n{body}"


def paired_contexts(rows: list[dict], top_k: int = 5, corruption: str = 'spurious',
                     severity: float = 0.4, seed: int = 13) -> list[dict]:
    """One pair per task whose corrupted ranking differs from its clean ranking."""
    clean, corrupted = {}, {}
    for row in rows:
        if row['method'] != 'one_hop':
            continue
        key = row['instance_id']
        if row['severity'] == 0:
            clean[key] = row
        elif row['corruption'] == corruption and row['severity'] == severity and row['seed'] == seed:
            corrupted[key] = row
    pairs = []
    for task_id in sorted(set(clean) & set(corrupted)):
        clean_files = clean[task_id]['ranked_files'][:top_k]
        corrupted_files = corrupted[task_id]['ranked_files'][:top_k]
        if clean_files == corrupted_files:
            continue
        pairs.append({'instance_id': task_id,
                       'clean_files': clean_files,
                       'corrupted_files': corrupted_files,
                       'gold_files': clean[task_id].get('gold_files', []),
                       'clean_recall_at_5': clean[task_id]['metrics']['recall@5'],
                       'corrupted_recall_at_5': corrupted[task_id]['metrics']['recall@5']})
    return pairs


def build_prompt(issue: str, files: list[str], sources: dict[str, str],
                 per_file_chars: int = 1200, total_chars: int = 6000) -> str:
    """A fixed prompt: issue, then the retrieved files, each truncated to a fixed budget."""
    if per_file_chars <= 0 or total_chars <= 0 or not issue.strip() or not files or len(set(files)) != len(files):
        raise ValueError('Nonempty issue, unique files and positive budgets required')
    missing = set(files)-set(sources)
    if missing:
        raise ValueError(f'Missing candidate source: {sorted(missing)}')
    # Reserve every header so scoring candidates always appear in the prompt.
    headers = [FILE_BLOCK.format(path=path,body='') for path in files]
    budget = total_chars-sum(map(len,headers))-2*(len(files)-1)
    if budget < 0:
        raise ValueError('Candidate headers exceed source budget')
    blocks=[]
    marker='\n... [truncated]'
    for path,header in zip(files,headers):
        body=sources[path]
        cap=min(per_file_chars,budget)
        if len(body)>cap:
            body=(body[:cap-len(marker)]+marker) if cap>=len(marker) else body[:cap]
        budget-=len(body)
        blocks.append(header+body)
    return PROMPT_TEMPLATE.format(issue=issue.strip(),files='\n\n'.join(blocks))


def prepare_contexts(data_root: Path, selection: dict, audit: dict, pairs: list[dict]) -> list[dict]:
    """Resolve only audited task-specific sources; no missing-data fallback."""
    from audit_competition_data import digest, read_tasks
    from pilot_integrity import validate_selection, index_unique
    from localization_data import read_snapshot
    import hashlib
    selected=validate_selection(selection,audit)
    if digest(data_root/'tasks.jsonl') != selection['tasks_sha256']:
        raise ValueError('Task fingerprint mismatch')
    tasks=index_unique(read_tasks(data_root/'tasks.jsonl'))
    records=[]
    for pair in pairs:
        task_id=pair['instance_id']
        if task_id not in selected or task_id not in tasks:
            raise ValueError(f'Unknown selected task: {task_id}')
        item=selected[task_id];task=tasks[task_id]
        if any(item[k]!=task[k] for k in ('repo','base_commit')):
            raise ValueError('Task identity mismatch')
        path=item['snapshot_path']
        if digest(data_root/path)!=audit['assets'][path]['sha256']:
            raise ValueError(f'Snapshot fingerprint mismatch: {task_id}')
        wanted=set(pair['clean_files'])|set(pair['corrupted_files'])
        sources=read_snapshot(data_root/path,wanted)
        if wanted-set(sources):
            raise ValueError(f'Missing candidate source for {task_id}: {sorted(wanted-set(sources))}')
        for condition in ('clean','corrupted'):
            files=pair[condition+'_files']
            prompt=build_prompt(task['problem_statement'],files,sources)
            records.append({'instance_id':task_id,'condition':condition,'files':files,
                'gold_files':pair['gold_files'],'prompt':prompt,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'snapshot_sha256':audit['assets'][path]['sha256']})
    return records


def parse_answer(answer: str, candidates: list[str]) -> str | None:
    """Accept a bare path copied from the candidate list; reject free text."""
    text = (answer or '').strip()
    if text in candidates:
        return text
    return None


def score_answers(records: list[dict], answers: dict[tuple[str, str], str]) -> dict:
    """Paired scoring: accuracy per arm, discordant pairs, exact sign test, faithfulness.

    `records` carry the per-task candidate lists and gold files; `answers` maps
    (instance_id, condition) to the raw model text.
    """
    per_task = defaultdict(dict)
    for record in records:
        chosen = parse_answer(answers.get((record['instance_id'], record['condition']), ''), record['files'])
        per_task[record['instance_id']][record['condition']] = {
            'chosen': chosen,
            'correct': bool(chosen and chosen in record['gold_files']),
            'faithful': chosen == record['files'][0] if chosen else None,
            'answerable': chosen is not None,
        }
    summary = {'tasks': len(per_task), 'arms': {}}
    for condition in ('clean', 'corrupted'):
        answered = [t[condition] for t in per_task.values() if condition in t]
        answerable = [a for a in answered if a['answerable']]
        faithful = [a['faithful'] for a in answerable if a['faithful']]
        summary['arms'][condition] = {
            'answerable': len(answerable),
            'unanswerable': len(answered) - len(answerable),
            'accuracy': (sum(a['correct'] for a in answerable) / len(answerable)) if answerable else None,
            'accuracy_all_attempts': sum(a['correct'] for a in answered)/len(answered) if answered else None,
            'faithfulness': (sum(faithful) / len(answerable)) if answerable else None,
        }
    clean_better = sum(1 for t in per_task.values()
                       if t.get('clean') and t.get('corrupted')
                       and t['clean']['answerable'] and t['corrupted']['answerable']
                       and t['clean']['correct'] and not t['corrupted']['correct'])
    corrupted_better = sum(1 for t in per_task.values()
                           if t.get('clean') and t.get('corrupted')
                           and t['clean']['answerable'] and t['corrupted']['answerable']
                           and t['corrupted']['correct'] and not t['clean']['correct'])
    total = clean_better + corrupted_better
    p_value = None
    if total:
        tail = min(clean_better, corrupted_better)
        p_value = min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / 2 ** total)
    summary['discordant'] = {'clean_only': clean_better, 'corrupted_only': corrupted_better,
                              'total': total, 'sign_test_p': p_value}
    summary['per_task'] = {task_id: arms for task_id, arms in sorted(per_task.items())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True, help='tasks.jsonl for issue statements')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--mapping-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--corruption', default='spurious')
    parser.add_argument('--severity', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing context file')
    from render_pilot_report import load_verified_run
    audit,selection,manifest,mappings,rows,summary=load_verified_run(
        args.audit,args.run_dir,args.mapping_dir,args.pilot,args.config)
    pairs=paired_contexts(rows,args.top_k,args.corruption,args.severity,args.seed)
    records=prepare_contexts(args.tasks.parent,selection,audit,pairs)
    # Prompts remain local; the exported diagnostic contains hashes, not corpus text.
    safe=[{k:v for k,v in r.items() if k not in ('prompt','gold_files')} for r in records]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps({'status':'cpu_preparation_only','pairs':len(pairs),'prompts':safe,
                                'run_id':args.run_dir.name},indent=2)+'\n')
    print(f'Prepared {len(pairs)} paired read-out contexts -> {args.output}')


if __name__ == '__main__':
    main()
