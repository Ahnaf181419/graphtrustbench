"""Evaluation-only patch parser and labels. Never import this from a retriever."""
from __future__ import annotations

import hashlib
import re
import shlex

from audit_competition_data import safe_member_name

HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def patch_path(header: str) -> str | None:
    raw = header[4:].split('\t', 1)[0]
    if raw.startswith('"'):
        raw = shlex.split(raw)[0]
    if raw == '/dev/null':
        return None
    if raw.startswith(('a/', 'b/')):
        raw = raw[2:]
    path = safe_member_name(raw)
    if path == '.' or path.startswith('.git/'):
        raise ValueError(f'Invalid patch path: {path}')
    return path


def parse_patch(patch: str) -> list[dict]:
    lines = patch.splitlines()
    files = []
    current = None
    git_section = False
    section_has_text = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            if git_section and not section_has_text:
                raise ValueError('Unsupported patch section without text hunks')
            git_section, section_has_text = True, False
            current = None
            i += 1
            continue
        if line.startswith(('Binary files ', 'GIT binary patch')):
            raise ValueError('Unsupported binary patch section')
        if line.startswith('--- '):
            if i + 1 >= len(lines) or not lines[i+1].startswith('+++ '):
                raise ValueError('Missing unified-diff new-file header')
            current = {'old_path': patch_path(line), 'new_path': patch_path(lines[i+1]), 'hunks': []}
            if current['old_path'] is None and current['new_path'] is None:
                raise ValueError('Both patch paths are null')
            files.append(current)
            i += 2
            continue
        match = HUNK.match(line)
        if match:
            if current is None:
                raise ValueError('Hunk without file headers')
            old_start, old_n, new_start, new_n = match.groups()
            old_start, old_n = int(old_start), int(old_n or 1)
            new_start, new_n = int(new_start), int(new_n or 1)
            if ((current['old_path'] is None and (old_start != 0 or old_n != 0)) or
                    (current['new_path'] is None and (new_start != 0 or new_n != 0))):
                raise ValueError('Null patch side must have zero start and count')
            before = []
            after_count = 0
            changed = []
            inserted = []
            pure_insertions = set()
            block_insertions = set()
            block_has_deletion = False
            i += 1
            while len(before) < old_n or after_count < new_n:
                if i >= len(lines) or not lines[i] or lines[i][0] not in (' ', '+', '-', '\\'):
                    raise ValueError('Truncated or malformed hunk')
                part = lines[i]
                if part.startswith('\\ No newline at end of file'):
                    i += 1
                    continue
                if part[0] == ' ':
                    if not block_has_deletion:
                        pure_insertions.update(block_insertions)
                    block_insertions.clear()
                    block_has_deletion = False
                if part[0] in (' ', '-'):
                    if part[0] == '-':
                        block_has_deletion = True
                        changed.append(old_start + len(before))
                    before.append(part[1:])
                if part[0] in (' ', '+'):
                    after_count += 1
                    if part[0] == '+':
                        # Insertion after the previous old-side line; 0 means before file start.
                        anchor = max(0, old_start + len(before) - (1 if old_n else 0))
                        inserted.append(anchor)
                        block_insertions.add(anchor)
                if len(before) > old_n or after_count > new_n:
                    raise ValueError('Hunk line counts do not match header')
                i += 1
            if not block_has_deletion:
                pure_insertions.update(block_insertions)
            section_has_text = True
            current['hunks'].append({'old_start': old_start, 'old_count': old_n,
                'new_start': new_start, 'new_count': new_n, 'before': before,
                'changed_old_lines': changed, 'insertion_after_lines': sorted(set(inserted)),
                'pure_insertion_after_lines': sorted(pure_insertions)})
            continue
        if line.startswith(('@@', '+', '-', ' ')) and not line.startswith(('+++ ', '--- ')):
            raise ValueError('Unexpected content outside a hunk')
        i += 1
    if not files or any(not f['hunks'] for f in files) or (git_section and not section_has_text):
        raise ValueError('Patch has no supported text hunks (binary/rename-only unsupported)')
    return files


def map_gold(patch: str, sources: dict[str, str], candidates: dict) -> dict:
    files = parse_patch(patch)
    errors = []
    gold_files = set()
    gold_nodes = set()
    details = []
    uncovered = 0
    ambiguous = False
    for file in files:
        path = file['old_path'] or file['new_path']
        gold_files.add(path)
        text = sources.get(path)
        lines = text.splitlines() if text is not None else []
        # The competition export uses a/path (not /dev/null) for some additions.
        inferred_new = text is None and all(h['old_start'] == 0 and h['old_count'] == 0 and h['new_start'] == 1 for h in file['hunks'])
        matched = 0
        file_nodes = set()
        for hunk in file['hunks']:
            start, count = hunk['old_start'], hunk['old_count']
            if file['old_path'] is not None and not inferred_new:
                if text is None or (count and (start < 1 or lines[start-1:start-1+count] != hunk['before'])) or (not count and not 0 <= start <= len(lines)):
                    errors.append({'path': path, 'old_start': start, 'reason': 'reference preimage does not match snapshot'})
                    continue
            elif text is not None:
                errors.append({'path': path, 'old_start': start, 'reason': 'new file already exists in snapshot'})
                continue
            matched += 1
            # Distinguish edit blocks: an unrelated insertion may share a replacement's hunk.
            points = [(n, False) for n in hunk['changed_old_lines']]
            points += [(n, True) for n in hunk['pure_insertion_after_lines']]
            for line, insertion in points:
                spans = [(end-start+1, node_id) for node_id, candidate in candidates.items()
                         if candidate['path'] == path for start, end in candidate['spans']
                         if (start <= line < end if insertion else start <= line <= end)]
                if not spans:
                    uncovered += 1
                    continue
                smallest = min(size for size, _ in spans)
                choices = {node_id for size, node_id in spans if size == smallest}
                ambiguous |= len(choices) != 1 or any(len(candidates[n]['spans']) > 1 for n in choices)
                file_nodes.update(choices)
        gold_nodes.update(file_nodes)
        details.append({'path': path, 'new_file': file['old_path'] is None or inferred_new,
                        'inferred_new_file': inferred_new and file['old_path'] is not None,
                        'hunks': len(file['hunks']), 'preimage_matched_hunks': matched,
                        'gold_nodes': sorted(file_nodes),
                        'changed_old_lines': sorted({n for h in file['hunks'] for n in h['changed_old_lines']}),
                        'insertion_after_lines': sorted({n for h in file['hunks'] for n in h['insertion_after_lines']}),
                        'source_sha256': hashlib.sha256(text.encode()).hexdigest() if text is not None else None})
    candidate_files = {c['path'] for c in candidates.values()}
    return {'status': 'failed' if errors else 'ambiguous' if ambiguous else 'file_only' if uncovered or not gold_nodes else 'precise',
            'gold_files': sorted(gold_files), 'gold_nodes': sorted(gold_nodes),
            'unretrievable_files': sorted(gold_files - candidate_files),
            'unmapped_changed_points': uncovered, 'preimage_errors': errors, 'files': details}


def metrics(ranking: list[str], gold: set[str]) -> dict:
    if not gold or len(set(ranking)) != len(ranking):
        raise ValueError('Metrics require nonempty gold and unique ranked locations')
    result = {}
    for k in (1, 5, 10):
        hits = len(set(ranking[:k]) & gold)
        result[f'recall@{k}'] = hits / len(gold)
        result[f'success@{k}'] = float(hits > 0)
    result['mrr'] = next((1 / i for i, item in enumerate(ranking, 1) if item in gold), 0.0)
    return result
