#!/usr/bin/env python3
"""Gemma 4 first-choice readout (RQ3): the decision readout behind Table 2 and hypotheses H1–H3.

Two halves, one file, so the whole protocol is committed before the main run:
  * `run(...)`     — GPU side (Kaggle, 4x L4, vLLM). Builds every prompt from the committed rankings and
                     the competition corpus, runs stages 1-4 and writes raw outputs. Labels are used only to
                     draw non-gold placebo files and to skip Recall@5-changed cells; they never enter a prompt.
  * `analyze(...)` — CPU side (local). Codes flips and computes H1-H3, R20, Table 3 and Figure 2.
"""
import gzip
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

TOKEN_BUDGET = 2500          # per-file token budget (paper M9)
MAX_TOKENS = 256
SYSTEM = 'You are a software engineer. Your job is to decide which files must be modified to resolve a GitHub issue.'
ASK = ('Which of these candidate files would you modify to resolve the issue?\n'
       'Reply with a JSON object only, in this form:\n'
       '{"files": ["<path>", ...], "none_relevant": false}\n'
       'List up to 3 paths taken from the candidate list, most likely first.\n'
       'If none of the candidates is relevant, reply {"files": [], "none_relevant": true}.')
RETRY_SUFFIX = '\n\nReply with the JSON object only, with no other text.'
SPECIAL = ('NONE', 'OFF-LIST', 'INVALID')


# ----------------------------------------------------------------------------- shared helpers
def load_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as fh:
        return [json.loads(line) for line in fh]


def rankings(rows):
    """clean[t] and corr[(t, kind, fraction, seed)] one-hop rankings, plus gold sets."""
    clean, corr, gold = {}, {}, {}
    for r in rows:
        if r['method'] != 'one_hop':
            continue
        if r['severity'] == 0:
            clean[r['instance_id']] = r['ranked_files']
        else:
            corr[(r['instance_id'], r['corruption'], r['severity'], r['seed'])] = r['ranked_files']
        if 'gold_files' in r:
            gold[r['instance_id']] = set(r['gold_files'])
    return clean, corr, gold


def recall5(rank, g):
    return len(set(rank[:5]) & g) / len(g) if g else 0.0


NORM = lambda p: re.sub(r'^\./', '', p.strip())


def parse(text, shown):
    """(first_choice, files). shown=None for issue-only prompts; the last JSON object wins."""
    objs = re.findall(r'\{[^{}]*"files"[^{}]*\}', text or '', re.S)
    if not objs:
        return 'INVALID', None
    try:
        obj = json.loads(objs[-1])
    except json.JSONDecodeError:
        return 'INVALID', None
    files = obj.get('files')
    if not isinstance(files, list) or not all(isinstance(x, str) for x in files):
        return 'INVALID', None
    files = [NORM(x) for x in files]
    if shown is None:
        return (files[0] if files else 'NONE'), files
    if not files:
        return 'NONE', files
    for f in files:
        if f in shown:
            return f, files
    return 'OFF-LIST', files


def is_forced(c0, shown):
    return c0 not in SPECIAL and c0 not in shown


def placebo(task, kind, seed, clean_rank, shown, gold):
    """Replace each incoming file in place with a seeded non-gold draw from clean ranks 6-20; None if pool too small."""
    incoming = [p for p in shown if p not in clean_rank[:5]]
    pool = [p for p in clean_rank[5:20] if p not in gold and p not in shown]
    if len(pool) < len(incoming):
        return None
    draw = random.Random(f'{task}|{kind}|{seed}').sample(pool, len(incoming))
    return [draw[incoming.index(p)] if p in incoming else p for p in shown]


def render_definitions(defs, encode, decode, budget=TOKEN_BUDGET):
    """defs: [{id, path, spans, text, bm25}] for one file. Clean-BM25 order (then node ID); a definition whose
    spans lie inside an already rendered span is skipped; the text is cut to `budget` tokens at a line boundary.
    Returns (text, rendered line spans)."""
    blocks, spans = [], []
    for d in sorted(defs, key=lambda x: (-x['bm25'], x['id'])):
        if all(any(a <= s and e <= b for a, b in spans) for s, e in d['spans']):
            continue
        blocks.append((f"# {d['id']}\n{d['text'].rstrip()}\n", d['spans']))
        spans += [tuple(s) for s in d['spans']]
    text, kept = '', []
    for block, block_spans in blocks:
        trial = text + block
        if len(encode(trial)) <= budget:
            text = trial; kept += [list(s) for s in block_spans]
            continue
        cut = decode(encode(trial)[:budget])
        cut = cut[:cut.rfind('\n')] if '\n' in cut else cut
        body_lines = cut[len(text):].count('\n')             # complete source lines after the '# id' header
        if body_lines >= 1:
            start = block_spans[0][0]
            kept.append([start, start + body_lines - 1])
        text = cut
        break
    return text, kept


# ----------------------------------------------------------------------------- GPU side
def conversation(repo, issue, files, rendered, ask=ASK):
    body = [f'Repository: {repo}', f'Issue:\n{issue}', 'Candidate files (in the order given):']
    body += [f'[{i}] {p}\n{rendered[p]}' for i, p in enumerate(files, 1)]
    body.append(ask)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(body)}]


def issue_only(repo, issue):
    user = (f'Repository: {repo}\n\nIssue:\n{issue}\n\n'
            'Which file(s) in this repository would you modify to resolve the issue?\n'
            'Reply with a JSON object only: {"files": ["<repository-relative path>", ...]}\n'
            'List up to 3 paths, most likely first.')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def build_contexts(task_ids, needed, comp, selection, encode, decode, budget=TOKEN_BUDGET):
    """Render every needed (task, file) from the competition corpus; returns rendered text, spans, issues, repos."""
    from audit_competition_data import read_tasks
    from localization_data import extract_candidates, read_snapshot
    from localization_retrieval import bm25
    tasks = {t['instance_id']: t for t in read_tasks(comp / 'tasks.jsonl')}
    items = {t['instance_id']: t for t in selection}
    rendered, spans = {}, {}
    for t in task_ids:
        graph = json.loads((comp / items[t]['graph_path']).read_text())
        cand, _ = extract_candidates(read_snapshot(comp / items[t]['snapshot_path']), graph['nodes'])
        lexical = bm25(tasks[t]['problem_statement'], cand, k1=1.2, b=0.75)
        by_file = defaultdict(list)
        for key, c in cand.items():
            by_file[c['path']].append(dict(c, bm25=lexical[key]))
        for p in needed[t]:
            rendered[(t, p)], spans[(t, p)] = render_definitions(by_file[p], encode, decode, budget)
        print('rendered', t, len(needed[t]), flush=True)
    issues = {t: tasks[t]['problem_statement'] for t in task_ids}
    repos = {t: tasks[t]['repo'] for t in task_ids}
    return rendered, spans, issues, repos


def make_stage(generate, out, stamp):
    """stage(items, tag): generate, parse, one re-ask for INVALID, write raw_<tag>.jsonl. The file is written to
    .tmp and renamed, so a dead session leaves no partial stage; a tag whose file exists is read back, not rerun."""
    def stage(items, tag):
        path = out / f'raw_{tag}.jsonl'
        if path.exists():
            stamp(f'{tag}: exists, reused')
            return load_rows(path)
        tick = time.time()
        res = []
        for it, (text, n) in zip(items, generate([it['conv'] for it in items])):
            choice, files = parse(text, it['shown'])
            res.append({**{k: v for k, v in it.items() if k != 'conv'}, 'raw': text, 'choice': choice,
                        'files': files, 'n_prompt_tokens': n})
        bad = [i for i, r in enumerate(res) if r['choice'] == 'INVALID']
        if bad:
            retry = [items[i]['conv'][:-1] + [{'role': 'user', 'content': items[i]['conv'][-1]['content'] + RETRY_SUFFIX}]
                     for i in bad]
            for i, (text, _) in zip(bad, generate(retry)):
                c, f = parse(text, items[i]['shown'])
                res[i].update(raw_retry=text, choice=c, files=f, retried=True)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            for r in res:
                fh.write(json.dumps(r) + '\n')
        tmp.rename(path)
        stamp(f'{tag}: {len(items)} prompts, {len(bad)} retried, '
              f'{sum(r["choice"] == "INVALID" for r in res)} invalid, {time.time() - tick:.0f}s')
        return res
    return stage


def run(mode, art, comp, model_path, out, generate=None, encode=None, decode=None):
    """mode 'pilot' (development tasks, clean only) or 'main'. `generate(convs) -> texts` defaults to vLLM."""
    out.mkdir(parents=True, exist_ok=True)
    if generate is None:
        import os, threading
        from vllm import LLM, SamplingParams
        # A failed engine core can leave this parent waiting forever on 4 GPUs; give startup 30 minutes.
        watchdog = threading.Timer(1800, lambda: (print('engine startup watchdog fired', flush=True), os._exit(3)))
        watchdog.start()
        # max_num_seqs=32: the default 256 dummy sampler requests run out of memory with Gemma 4's vocabulary.
        llm = LLM(model=str(model_path), tensor_parallel_size=4, max_model_len=32768,
                  gpu_memory_utilization=0.85, max_num_seqs=32, seed=0)
        watchdog.cancel()
        tok = llm.get_tokenizer()
        encode = lambda s: tok.encode(s, add_special_tokens=False)
        decode = tok.decode
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

        def generate(convs):
            outs = llm.chat(convs, params, chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            return [(o.outputs[0].text, len(o.prompt_token_ids)) for o in outs]
        import vllm
        (out / 'environment.json').write_text(json.dumps({'vllm': vllm.__version__, 'model_path': str(model_path)}))
    log = open(out / 'run_log.txt', 'a')
    stamp = lambda msg: (print(msg, flush=True), log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}\n'), log.flush())

    if mode == 'pilot':
        rows = load_rows(art / 'paper/results/2026-11-07_localization_v3/per_task.jsonl')
        selection = json.loads((art / 'paper/data/pilot_2026-11-06.json').read_text())['tasks']
        clean, corr, gold = rankings(rows)
        tasks = sorted(clean)[:10]
        corr = {}
    else:
        rows = load_rows(art / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz')
        selection = json.loads((art / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']
        clean, corr, gold = rankings(rows)
        tasks = sorted(clean)
        corr = {k: v for k, v in corr.items() if k[2] == 0.4}
    needed = defaultdict(set)
    for t in tasks:
        needed[t] |= set(clean[t][:20])
    for (t, *_), r in corr.items():
        needed[t] |= set(r[:5])
    rendered, spans, issues, repos = build_contexts(tasks, needed, comp, selection, encode, decode)
    (out / 'rendered_spans.json').write_text(json.dumps({f'{t}|{p}': s for (t, p), s in spans.items()}))
    conv = lambda t, files: conversation(repos[t], issues[t], files, {p: rendered[(t, p)] for p in files})

    stage = make_stage(generate, out, stamp)

    decision_stages(tasks, clean, corr, gold, conv, lambda t: issue_only(repos[t], issues[t]), stage, stamp,
                    only_stage1=mode == 'pilot')
    if mode != 'pilot':
        stamp('done')


def decision_stages(tasks, clean, corr, gold, conv, cn_conv, stage, stamp, only_stage1=False):
    """Registration 1's four stages (C0/CR/CN, corrupted cells, matched placebos, C0r x3) for any task set.
    conv(task, files) and cn_conv(task) build conversations; stage(items, tag) runs them and returns rows with
    'choice'. Returns (r1, r2, r3, r4)."""
    s1 = []
    for t in tasks:
        top = clean[t][:5]
        s1 += [dict(id=f'C0|{t}', cond='C0', task=t, conv=conv(t, top), shown=top),
               dict(id=f'CR|{t}', cond='CR', task=t, conv=conv(t, top[::-1]), shown=top[::-1]),
               dict(id=f'CN|{t}', cond='CN', task=t, conv=cn_conv(t), shown=None)]
    r1 = stage(s1, 'stage1')
    if only_stage1:                      # pilot: clean conditions only
        return r1, [], [], []
    c0 = {r['task']: r['choice'] for r in r1 if r['cond'] == 'C0'}
    s2 = [dict(id=f'X|{t}|{k}|{sd}', cond='X', task=t, intervention=k, seed=sd, conv=conv(t, r[:5]), shown=r[:5])
          for (t, k, f, sd), r in sorted(corr.items()) if r[:5] != clean[t][:5]]
    r2 = stage(s2, 'stage2')
    s3, skipped = [], 0
    for row in r2:
        t, shown = row['task'], row['shown']
        top_c = clean[t][:5]
        if set(top_c) == set(shown) or recall5(shown, gold[t]) != recall5(top_c, gold[t]) or is_forced(c0[t], shown):
            continue
        plac = placebo(t, row['intervention'], row['seed'], clean[t], shown, gold[t])
        if plac is None:
            skipped += 1
            continue
        s3.append(dict(id=f'P|{t}|{row["intervention"]}|{row["seed"]}', cond='P', task=t,
                       intervention=row['intervention'], seed=row['seed'], conv=conv(t, plac), shown=plac))
    stamp(f'placebo pool too small: {skipped}')
    r3 = stage(s3, 'stage3')
    fill, r4 = s2 + s3, []
    for rep in (1, 2, 3):
        rr = random.Random(100 + rep)
        batch = [dict(it, id=f'C0r{rep}|{it["task"]}', cond='C0r', rep=rep) for it in s1 if it['cond'] == 'C0']
        batch += [dict(it, filler=True) for it in rr.sample(fill, min(len(fill), len(tasks)))]
        rr.shuffle(batch)
        r4 += [r for r in stage(batch, f'stage4_rep{rep}') if r['cond'] == 'C0r']
    return r1, r2, r3, r4


# ----------------------------------------------------------------------------- CPU side (analysis)
def load_stages(raw_dir, prefix=''):
    """(r1, r2, r3, r4) raw replies of one readout; r4 keeps only the C0r repeats, not the fillers."""
    raw = lambda tag: [json.loads(l) for l in open(raw_dir / f'raw_{prefix}{tag}.jsonl')]
    r4 = [r for rep in (1, 2, 3) for r in raw(f'stage4_rep{rep}') if r['cond'] == 'C0r']
    return raw('stage1'), raw('stage2'), raw('stage3'), r4


def analyze(raw_dir, per_task, out_dir, figure, tasks_file=None, seed=2026):
    rows = load_rows(per_task)
    clean, corr, gold = rankings(rows)
    retrievable = {r['instance_id']: set(r['gold_files']) - set(r['unretrievable_gold_files'])
                   for r in rows if r['method'] == 'one_hop' and r['severity'] == 0}
    result = analyze_core(*load_stages(raw_dir), clean, corr, gold, retrievable, seed)
    tasks = sorted(clean)
    if tasks_file:
        from audit_competition_data import read_tasks
        patches = {t['instance_id']: t['patch'] for t in read_tasks(tasks_file)}
        spans = json.loads((raw_dir / 'rendered_spans.json').read_text())
        pairs = [(t, p) for t in tasks for p in clean[t][:5] if p in gold[t]]
        vis = [visibility(edited_lines(patches[t]).get(p, []), spans.get(f'{t}|{p}', [])) for t, p in pairs]
        result['gold_visibility_C0'] = dict(shown_gold_files=len(pairs), all_edited_lines_visible=vis.count('all'),
                                            some_visible=vis.count('some'), none_visible=vis.count('none'))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'readout_values.json').write_text(json.dumps(result, indent=2) + '\n')
    figure2(result['table3'], figure)
    return result


def analyze_core(r1, r2, r3, r4, clean, corr, gold, retrievable, seed=2026):
    """The registration-1 analysis (H1-H3, Table 3, R20) for any task set: clean/corr are the one-hop
    rankings, gold the labels; only tasks in `clean` are analysed. Second registration H4 reuses it."""
    import numpy as np
    tasks = sorted(clean)
    c0 = {r['task']: r['choice'] for r in r1 if r['cond'] == 'C0'}
    hit = lambda ch, t: float(ch in gold[t])

    def boot(v, n=10_000):
        v = np.asarray(v, float)
        return v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(axis=1)

    def ci(v, level=0.95):
        v = np.asarray(v, float)
        if not len(v):
            return None
        b = boot(v); a = (1 - level) / 2
        return float(v.mean()), float(np.quantile(b, a)), float(np.quantile(b, 1 - a))

    one_sided_p = lambda v: float((boot(v) <= 0).mean())

    def holm(p, alpha=0.05):
        order = sorted(range(len(p)), key=lambda i: p[i]); rej = [False] * len(p)
        for rank, i in enumerate(order):
            if p[i] <= alpha / (len(p) - rank):
                rej[i] = True
            else:
                break
        return rej

    def per_task_mean(cells, field='flip'):
        acc = defaultdict(list)
        for c in cells:
            acc[c['task']].append(c[field])
        return {t: sum(v) / len(v) for t, v in sorted(acc.items())}

    valid = lambda r: r['choice'] != 'INVALID' and c0[r['task']] != 'INVALID'
    for r in r2 + r3 + r4 + r1:
        r['flip'] = float(r['choice'] != c0[r['task']])
    X = [r for r in r2 if valid(r)]
    for r in X:
        top_c = clean[r['task']][:5]
        r['order_only'] = set(top_c) == set(r['shown'])
        r['forced'] = is_forced(c0[r['task']], r['shown'])
        r['r5_same'] = recall5(r['shown'], gold[r['task']]) == recall5(top_c, gold[r['task']])
    floor = per_task_mean([r for r in r4 if valid(r)])
    crf = per_task_mean([r for r in r1 if r['cond'] == 'CR' and valid(r)])

    groups = {'order': [r for r in X if r['order_only']],
              'forced': [r for r in X if not r['order_only'] and r['forced']],
              'r5_changed': [r for r in X if not r['order_only'] and not r['forced'] and not r['r5_same']],
              'H2': [r for r in X if not r['order_only'] and not r['forced'] and r['r5_same']]}
    O = per_task_mean(groups['order'])
    d1 = [O[t] - crf[t] for t in O if t in crf]
    h1 = ci(d1, 0.90) if d1 else None
    f_m0 = per_task_mean(groups['H2'])
    d2 = [f_m0[t] - floor[t] for t in f_m0 if t in floor]
    h2 = ci(d2); p2 = one_sided_p(d2) if d2 else 1.0
    P = {(r['task'], r['intervention'], r['seed']): r['flip'] for r in r3 if valid(r)}
    pair = [dict(task=r['task'], intervention=r['intervention'], diff=r['flip'] - P[k], pflip=P[k])
            for r in groups['H2'] if (k := (r['task'], r['intervention'], r['seed'])) in P]
    d3m = per_task_mean(pair, 'diff'); d3 = list(d3m.values())
    h3 = ci(d3); p3 = one_sided_p(d3) if d3 else 1.0
    h3_90 = ci(d3, 0.90)
    rej2, rej3 = holm([p2, p3])
    if rej3:
        verdict = 'corruption-specific'
    elif h3 and h3[2] < 0:
        verdict = 'placebo-heavier'
    elif h3_90 and h3_90[1] > -0.10 and h3_90[2] < 0.10:
        verdict = 'generic'
    else:
        verdict = 'inconclusive'
    by_iv = {iv: ci(list(per_task_mean([p for p in pair if p['intervention'] == iv], 'diff').values()))
             for iv in ('missing', 'spurious')}

    # Sensitivity: drop tasks whose issue-only first choice is gold.
    cn = {r['task']: r['choice'] for r in r1 if r['cond'] == 'CN'}
    mem = {t for t in tasks if cn.get(t) in gold[t]}
    d3_sens = [v for t, v in d3m.items() if t not in mem]
    d2_sens = [f_m0[t] - floor[t] for t in f_m0 if t in floor and t not in mem]

    # R20: net change in gold first choices over all 40% cells; unchanged ordered top five = C0's prompt.
    xr = {(r['task'], r['intervention'], r['seed']): r['choice'] for r in r2}
    elig = [t for t in tasks if retrievable[t]]
    dv = []
    for t in elig:
        diffs = [hit(xr.get((t, k, sd), c0[t]), t) - hit(c0[t], t)
                 for (tt, k, f, sd) in sorted(corr) if tt == t and f == 0.4]
        dv.append(sum(diffs) / len(diffs))
    dv = np.array(dv)
    r20 = ci(dv)
    signs = np.random.default_rng(seed).choice([-1, 1], size=(100_000, len(dv)))
    p20 = float((np.abs((signs * dv).mean(1)) >= abs(dv.mean()) - 1e-15).mean())

    def row(name, cells, comparisons=None, rate=True):
        rates = per_task_mean(cells)
        lost = sum(c0[c['task']] in gold[c['task']] and c['choice'] not in gold[c['task']] for c in cells)
        gained = sum(c0[c['task']] not in gold[c['task']] and c['choice'] in gold[c['task']] for c in cells)
        return dict(condition=name, comparisons=comparisons or len(cells), tasks=len(rates),
                    flip=ci(list(rates.values())) if rate and rates else None, lost=lost, gained=gained)

    table3 = [row('C0r', [r for r in r4 if valid(r)]),
              row('CR', [r for r in r1 if r['cond'] == 'CR' and valid(r)]),
              row('order-only', groups['order']), row('forced', groups['forced'], rate=False),
              row('r5-changed', groups['r5_changed']), row('H2', groups['H2']),
              row('placebo', [r for r in r3 if valid(r)])]
    invalid = {tag: sum(r['choice'] == 'INVALID' for r in rs) for tag, rs in
               (('stage1', r1), ('stage2', r2), ('stage3', r3), ('C0r', r4))}
    result = dict(H1=dict(ci90=h1, equivalent=bool(h1 and h1[1] > -0.10 and h1[2] < 0.10), tasks=len(d1)),
                  H2=dict(ci=h2, p=p2, holm_reject=rej2, tasks=len(d2), sensitivity=ci(d2_sens)),
                  H3=dict(ci=h3, ci90=h3_90, p=p3, holm_reject=rej3, verdict=verdict, tasks=len(d3),
                          by_intervention=by_iv, sensitivity=ci(d3_sens)),
                  R2=float(np.mean(list(f_m0.values()))) if f_m0 else None,
                  R3=float(np.mean(list(per_task_mean([dict(task=p['task'], flip=p['pflip']) for p in pair]).values()))) if pair else None,
                  R20=dict(ci=r20, p_sign_flip=p20, tasks=len(elig)),
                  CN_hits=len(mem), C0_hits=int(sum(hit(c0[t], t) for t in elig)), eligible_tasks=len(elig),
                  invalid=invalid, table3=table3)   # the issue-only gold list stays out: it is a label
    return result


def edited_lines(patch):
    """Pre-image line numbers touched by a unified diff, per file: removed lines, and for a pure insertion the
    pre-image line it follows (line 1 for an insertion at the top). Evaluation only; never shown to the model."""
    out, path, old = defaultdict(set), None, 0
    for line in patch.splitlines():
        if line.startswith('--- '):
            path = line[4:].split('\t')[0]
            path = path[2:] if path.startswith('a/') else None if path == '/dev/null' else path
        elif line.startswith('+++ '):
            continue
        elif line.startswith('@@'):
            old = int(re.match(r'@@ -(\d+)', line).group(1))
        elif path is None:
            continue
        elif line.startswith('-'):
            out[path].add(old); old += 1
        elif line.startswith('+'):
            out[path].add(max(old - 1, 1))
        elif line.startswith(' '):
            old += 1
    return {p: sorted(v) for p, v in out.items()}


def visibility(lines, spans):
    inside = [any(a <= n <= b for a, b in spans) for n in lines]
    return 'all' if inside and all(inside) else 'some' if any(inside) else 'none'


def figure2(table3, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in table3 if r['flip'] and r['condition'] != 'C0r']
    floor = next(r for r in table3 if r['condition'] == 'C0r')['flip']
    labels = {'CR': 'reversed\nclean', 'order-only': 'order\nonly', 'r5-changed': 'membership,\nRecall@5 changed',
              'H2': 'membership,\nRecall@5 same', 'placebo': 'matched\nplacebo'}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    m = [r['flip'][0] * 100 for r in rows]
    err = [[(r['flip'][0] - r['flip'][1]) * 100 for r in rows], [(r['flip'][2] - r['flip'][0]) * 100 for r in rows]]
    ax.bar(range(len(rows)), m, yerr=err, capsize=4, color='#6b8fbf')
    if floor:
        ax.axhline(floor[0] * 100, color='#c0392b', ls='--', lw=1, label=f'engine floor (C0r) {floor[0] * 100:.1f}%')
        ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(range(len(rows)), [labels[r['condition']] for r in rows], fontsize=8)
    ax.set_ylabel('first-choice flip rate (%)')
    plt.tight_layout(); plt.savefig(path, dpi=200)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(analyze(Path(sys.argv[1]), root / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz',
                             root / 'paper/reports/gemma_readout', root / 'paper/archive/v5/figures/f2_flip_rates.png',
                             root / 'tasks.jsonl'), indent=2))
