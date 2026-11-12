"""Load one competition task the way the registered runner does (run_localization_pilot.py)."""
import json
from functools import lru_cache
from pathlib import Path

from audit_competition_data import read_tasks
from localization_data import extract_candidates, read_snapshot, retrieval_task
from localization_retrieval import bm25


@lru_cache(maxsize=4)
def _tasks(path):
    return {t['instance_id']: retrieval_task(t) for t in read_tasks(Path(path))}


def load_task(data_root, item, config):
    """(candidates, lexical, edges) for a selection item; only allowlisted task fields are read."""
    task = _tasks(str(Path(data_root) / 'tasks.jsonl'))[item['instance_id']]
    graph = json.loads((Path(data_root) / item['graph_path']).read_text())
    candidates, _ = extract_candidates(read_snapshot(Path(data_root) / item['snapshot_path']), graph['nodes'])
    return candidates, bm25(task['problem_statement'], candidates, **config['bm25']), graph['edges']
