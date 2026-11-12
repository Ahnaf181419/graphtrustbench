"""Batched multi-turn tool loops for the agentic retriever (Q1) and the agent readout (R1).
`generate(list_of_conversations) -> list_of_texts` is pluggable: vLLM's llm.chat on Kaggle,
a fake for tests. All runs are greedy, so a loop is deterministic given the graph."""
import json
import re

from .repo_ops import extract_diff
from .retrievers import bm25_order

AG_STEPS = 8          # agentic retriever: neighbour calls before it must answer
AG_SHOW = 20          # neighbours shown per call
AGENT_TURNS = 20      # agent readout: tool calls before it must submit
READ_LINES = 200      # agent readout: lines per read_file call


def json_objects(text):
    """All top-level {...} spans that parse as JSON, scanning braces outside strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def last_tool_call(text):
    calls = [o for o in json_objects(text) if isinstance(o, dict) and 'tool' in o]
    return calls[-1] if calls else None


def signature(source):
    first = (source or '').strip().splitlines()
    return first[0][:160] if first else ''


# ---------- agentic retriever ----------

AG_SYSTEM = ('You explore a repository call graph to find the files that must be modified to resolve '
             'a GitHub issue. Call one tool per reply, as a JSON object and nothing else.')
AG_TOOLS = ('Tools:\n'
            '{"tool": "neighbours", "node": "<node id>"}  lists functions this node calls.\n'
            '{"tool": "done", "files": ["<path>", ...]}  ends the search: up to 10 files, most likely first.\n'
            f'You may call "neighbours" at most {AG_STEPS} times.')


def ag_start(task, issue, seeds, node2file, defs_src):
    lines = [f'- {s} ({node2file[s]}): {signature(defs_src.get(s))}' for s in seeds]
    user = f'Issue:\n{issue}\n\nStarting points:\n' + '\n'.join(lines) + '\n\n' + AG_TOOLS
    return dict(task=task, conv=[{'role': 'system', 'content': AG_SYSTEM}, {'role': 'user', 'content': user}],
                steps=0, done=False, seen=[], files=None, invalid=0)


def ag_step(state, call, succ, bm25, node2file, defs_src):
    """Execute one call; returns the observation text or None when the state is finished."""
    if call is None:
        state['invalid'] += 1
        if state['invalid'] >= 2:
            state.update(done=True, files=[])
            return None
        return 'Reply with one JSON tool call only.'
    if call.get('tool') == 'done' or state['steps'] >= AG_STEPS:
        files = call.get('files') if call.get('tool') == 'done' else None
        if files is None:
            return f'You have used all {AG_STEPS} calls. Reply with {{"tool": "done", "files": [...]}}.'
        state.update(done=True, files=[f for f in files if isinstance(f, str)])
        return None
    if call.get('tool') == 'neighbours':
        state['steps'] += 1
        n = str(call.get('node', ''))
        nb = [v for v in sorted(set(succ.get(n, []))) if v in bm25][:AG_SHOW]
        state['seen'] += [v for v in nb if v not in state['seen']]
        if not nb:
            return f'{n}: no known callees.'
        return f'{n} calls:\n' + '\n'.join(f'- {v} ({node2file[v]}): {signature(defs_src.get(v))}' for v in nb)
    return 'Unknown tool. Use "neighbours" or "done".'


def ag_final_ranking(state, bm25, node2file):
    """Registered rule: the model's listed files (deduplicated, candidates only) first,
    then every other candidate file in BM25 order."""
    cand_files = {node2file[n] for n in bm25}
    out = []
    for f in state['files'] or []:
        f = re.sub(r'^\./', '', f.strip())
        if f in cand_files and f not in out:
            out.append(f)
    for n in bm25_order(bm25):
        f = node2file[n]
        if f not in out:
            out.append(f)
    return out


def run_loop(states, generate, step, max_rounds):
    """Advance all unfinished states one model reply per round."""
    for _ in range(max_rounds):
        active = [s for s in states if not s['done']]
        if not active:
            break
        texts = generate([s['conv'] for s in active])
        for s, text in zip(active, texts):
            call = last_tool_call(text)
            s.setdefault('log', []).append(call)
            obs = step(s, call, text)
            s['conv'] = s['conv'] + [{'role': 'assistant', 'content': json.dumps(call) if call else text[-2000:]}]
            if obs is not None:
                s['conv'] = s['conv'] + [{'role': 'user', 'content': obs}]
    for s in states:
        if not s['done']:
            s.update(done=True, timed_out=True)
    return states


# ---------- agent readout ----------

AGENT_SYSTEM = ('You are a software engineer fixing a GitHub issue in a repository. Use the tools to read '
                'code, then submit a patch. Call one tool per reply as a JSON object. To submit, write the '
                'patch as a unified diff in a ```diff block, then {"tool": "submit"}.')
AGENT_TOOLS = ('Tools:\n'
               '{"tool": "read_file", "path": "<path>", "start": 1}  shows up to '
               f'{READ_LINES} lines from line "start".\n'
               '{"tool": "search", "query": "<words>"}  lists the 10 best-matching definitions.\n'
               '{"tool": "neighbours", "node": "<node id>"}  lists functions this node calls.\n'
               '{"tool": "submit"}  after a ```diff block with your patch.\n'
               f'You may make at most {AGENT_TURNS} calls.')


def agent_start(task, issue, top5, cond):
    user = (f'Issue:\n{issue}\n\nFiles a retriever suggests, most likely first:\n'
            + '\n'.join(f'{i}. {p}' for i, p in enumerate(top5, 1)) + '\n\n' + AGENT_TOOLS)
    return dict(task=task, cond=cond, top5=list(top5), steps=0, done=False, reads=[], diff=None,
                conv=[{'role': 'system', 'content': AGENT_SYSTEM}, {'role': 'user', 'content': user}])


def agent_step(state, call, text, read_file, search, succ, node2file):
    """read_file(task, path) -> full text or None; search(task, query) -> [(node, path, signature)]."""
    if call is not None and call.get('tool') == 'submit':
        state.update(done=True, diff=extract_diff(text))
        return None
    if state['steps'] >= AGENT_TURNS:
        state.update(done=True, diff=extract_diff(text))
        return None
    state['steps'] += 1
    if call is None:
        return 'Reply with one JSON tool call.'
    t = call.get('tool')
    if t == 'read_file':
        p = re.sub(r'^\./', '', str(call.get('path', '')).strip())
        src = read_file(state['task'], p)
        if src is None:
            return f'{p}: no such file.'
        state['reads'].append(p)
        lines = src.splitlines()
        start = max(int(call.get('start', 1) or 1), 1)
        chunk = lines[start - 1:start - 1 + READ_LINES]
        return f'{p} lines {start}-{start + len(chunk) - 1} of {len(lines)}:\n' + '\n'.join(
            f'{start + i}: {ln}' for i, ln in enumerate(chunk))
    if t == 'search':
        hits = search(state['task'], str(call.get('query', '')))[:10]
        return '\n'.join(f'- {n} ({p}): {sig}' for n, p, sig in hits) or 'No matches.'
    if t == 'neighbours':
        n = str(call.get('node', ''))
        nb = sorted(set(succ.get(n, [])))[:AG_SHOW]
        return '\n'.join(f'- {v} ({node2file.get(v, "?")})' for v in nb) or f'{n}: no known callees.'
    return 'Unknown tool.'


def recovered(state):
    """Registered recovery signal: the agent read at least one file outside its initial top five."""
    return any(p not in state['top5'] for p in state['reads'])
