"""Plan 3 (G6): the second registration's GPU readouts. Model calls go through an injected
generate(convs) -> [(text, n_prompt_tokens)], so everything here runs on CPU in tests."""
import gzip
import json
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import gemma_readout as g
from gemma_readout import make_stage                       # noqa: F401  (re-exported for the notebook)
from localization_data import extract_candidates

REPO = {'rich': 'Textualize/rich', 'fastapi': 'fastapi/fastapi', 'requests': 'psf/requests'}


def pc_inputs(full, rankings):
    """(tasks, clean, corr, gold) for the post-cutoff tasks, keyed like gemma_readout.rankings."""
    clean, corr = {}, {}
    for r in rankings:
        iv, frac, seed = r['key']
        if iv == 'clean':
            clean[r['task']] = r['ranking']
        else:
            corr[(r['task'], iv, frac, seed)] = r['ranking']
    gold = {T['task_id']: set(T['gold_files']) for T in full}
    return sorted(clean), clean, corr, gold


def pc_contexts(full, needed, encode, decode):
    """The competition rendering for the post-cutoff tasks: candidates bound by extract_candidates from the
    pre-fix files, ordered by the frozen BM25, rendered by render_definitions (B = 2,500)."""
    rendered, spans, issues, repos = {}, {}, {}, {}
    for T in full:
        t = T['task_id']
        if t not in needed:
            continue
        nodes = [dict(id=n, text=T['defs'][n]['source']) for n in T['node2file']]
        cand, _ = extract_candidates(T['files'], nodes)
        by_file = defaultdict(list)
        for key, c in cand.items():
            by_file[c['path']].append(dict(c, bm25=T['bm25'][key]))
        for p in needed[t]:
            rendered[(t, p)], spans[(t, p)] = g.render_definitions(by_file[p], encode, decode)
        issues[t], repos[t] = T['problem_statement'], REPO[T['repo']]
    return rendered, spans, issues, repos


def stamper(out):
    log = open(out / 'run_log.txt', 'a')
    return lambda msg: (print(msg, flush=True),
                        log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}\n'), log.flush())


def run_decision_pc(full_path, rankings_path, generate, encode, decode, out):
    """G6.1: registration 1's four decision stages on the post-cutoff tasks; raw_pc_stage*.jsonl."""
    out.mkdir(parents=True, exist_ok=True)
    full = json.load(gzip.open(full_path, 'rt'))
    tasks, clean, corr, gold = pc_inputs(full, json.load(gzip.open(rankings_path, 'rt')))
    needed = defaultdict(set)
    for t in tasks:
        needed[t] |= set(clean[t][:20])
    for (t, *_), r in corr.items():
        needed[t] |= set(r[:5])
    rendered, spans, issues, repos = pc_contexts(full, needed, encode, decode)
    (out / 'rendered_spans_pc.json').write_text(json.dumps({f'{t}|{p}': s for (t, p), s in spans.items()}))
    conv = lambda t, files: g.conversation(repos[t], issues[t], files, {p: rendered[(t, p)] for p in files})
    stamp = stamper(out)
    base = make_stage(generate, out, stamp)
    stage = lambda items, tag: base(items, f'pc_{tag}')
    res = g.decision_stages(tasks, clean, corr, gold, conv, lambda t: g.issue_only(repos[t], issues[t]), stage, stamp)
    stamp('pc decision readout done')
    return res


# ---------------------------------------------------------------- G6.2 patch readout (§5.2, Appendix A)
PB = 4800                                   # tokens per file
MAX_PROMPT = 32768 - 4096                   # max_model_len minus the output budget
SYSTEM_PATCH = 'You are a software engineer. Write a patch that resolves the GitHub issue.'
ASK_PATCH = ('Write a patch that resolves the issue. Reply with a single unified diff in a ```diff block, with paths '
             'relative to the repository root and a/ and b/ prefixes. Change only what is needed.')


def render_full(text, encode, decode, budget=PB):
    """The whole file, cut at a line boundary to at most `budget` tokens."""
    ids = encode(text)
    if len(ids) <= budget:
        return text
    cut = decode(ids[:budget])
    return cut[:cut.rfind('\n')] if '\n' in cut else cut


def patch_conv(repo, issue, files, texts):
    body = [f'Repository: {repo}', f'Issue:\n{issue}', 'Relevant files (in the order given):']
    body += [f'[{i}] {p}\n```python\n{texts[p]}\n```' for i, p in enumerate(files, 1)]
    body.append(ASK_PATCH)
    return [{'role': 'system', 'content': SYSTEM_PATCH}, {'role': 'user', 'content': '\n\n'.join(body)}]


def patch_items(tasks, clean, r2, r3, has_tests, rep_rank=None):
    """C0, the decision readout's corrupted cells (X) and placebo lists (P), and the repaired top five (REP,
    oracle repair) where it differs from clean."""
    key = lambda c, r: f"{c}|{r['task']}|{r['intervention']}|{r['seed']}"
    it = [dict(id=f'C0|{t}', cond='C0', task=t, shown=clean[t][:5]) for t in tasks if has_tests(t)]
    for cond, rows in (('X', r2), ('P', r3)):
        it += [dict(id=key(cond, r), cond=cond, task=r['task'], intervention=r['intervention'], seed=r['seed'],
                    shown=list(r['shown'])) for r in rows if has_tests(r['task'])]
    for t, rr in sorted((rep_rank or {}).items()):
        if has_tests(t) and rr[:5] != clean[t][:5]:
            it.append(dict(id=f'REP|{t}', cond='REP', task=t, shown=rr[:5]))
    return it


def floor_batches(items, rep):
    """Patch floor: the C0 items again, among min(200, n) seeded fillers at different batch positions."""
    rr = random.Random(200 + rep)
    batch = [dict(x, id=f'C0r{rep}|{x["task"]}', cond='C0r', rep=rep) for x in items if x['cond'] == 'C0']
    batch += [dict(x, filler=True) for x in rr.sample(items, min(200, len(items)))]
    rr.shuffle(batch)
    return batch


def comp_patch_texts(root, selection, items, encode, decode):
    """{(task, path): rendered text} from the competition snapshots (pre-fix trees)."""
    from localization_data import read_snapshot
    wanted = defaultdict(set)
    for x in items:
        wanted[x['task']] |= set(x['shown'])
    texts = {}
    for t, paths in sorted(wanted.items()):
        src = read_snapshot(Path(root) / selection[t]['snapshot_path'], paths)
        for p in paths:
            texts[(t, p)] = render_full(src.get(p, ''), encode, decode)
    return texts


def attach_patch_convs(items, texts, repos, issues):
    return [dict(x, conv=patch_conv(repos[x['task']], issues[x['task']], x['shown'],
                                    {p: texts[(x['task'], p)] for p in x['shown']})) for x in items]


def run_raw(items, tag, generate, encode, out, stamp, limit=MAX_PROMPT, chunk=100):
    """Generate and keep the raw text (patches are parsed and tested in G7). Refuses before generating if any
    prompt is over `limit` tokens (content tokens + 64 for the chat template). Atomic and resumable."""
    path = out / f'raw_{tag}.jsonl'
    if path.exists():
        stamp(f'{tag}: exists, reused')
        return g.load_rows(path)
    longest = max((sum(len(encode(m['content'])) for m in x['conv']) + 64 for x in items), default=0)
    if longest > limit:
        raise SystemExit(f'{tag}: a prompt has ~{longest} tokens, over {limit}; raising max_model_len needs a deviation')
    tick, res = time.time(), []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        for x, (text, n) in zip(part, generate([x['conv'] for x in part])):
            res.append({**{k: v for k, v in x.items() if k != 'conv'}, 'raw': text, 'n_prompt_tokens': n})
        stamp(f'{tag}: {len(res)}/{len(items)} after {time.time() - tick:.0f}s')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in res))
    tmp.rename(path)
    stamp(f'{tag}: {len(items)} prompts, max prompt {max((r["n_prompt_tokens"] for r in res), default=0)} tokens, '
          f'{time.time() - tick:.0f}s')
    return res


def run_patch_comp(art, comp, generate, encode, decode, out, which=('main', 'floor')):
    """G6.2 on the competition tasks: C0, X, P (registration-1 decision readout lists), REP; then the floor.
    Patches are generated for every task; tasks without F2P tests or with a failed environment are excluded at
    evaluation (G7), independent of model output."""
    out.mkdir(parents=True, exist_ok=True)
    stamp = stamper(out)
    clean, _, _ = g.rankings(g.load_rows(art / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz'))
    v5 = art / 'paper/results/2026-11-10_gemma_readout'
    r2, r3 = (g.load_rows(v5 / f'raw_stage{i}.jsonl') for i in (2, 3))
    rep = json.load(gzip.open(art / 'paper/results/2026-11-11_g4_trace/repaired_rankings.json.gz', 'rt'))
    assert all(v['clean'][:5] == clean[t][:5] for t, v in rep.items()), 'G4 clean rankings differ from registration 1'
    items = patch_items(sorted(clean), clean, r2, r3, lambda t: True, {t: v['repaired'] for t, v in rep.items()})
    selection = {t['instance_id']: t for t in json.loads((art / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']}
    tasks = {t['instance_id']: t for t in map(json.loads, open(comp / 'tasks.jsonl')) if t['instance_id'] in clean}
    texts = comp_patch_texts(comp, selection, items, encode, decode)
    attach = lambda its: attach_patch_convs(its, texts, {t: v['repo'] for t, v in tasks.items()},
                                            {t: v['problem_statement'] for t, v in tasks.items()})
    stamp(f'patch items: {len(items)} ' + str(dict(Counter(x["cond"] for x in items))))
    if 'main' in which:
        run_raw(attach(items), 'patch_comp', generate, encode, out, stamp)
    if 'floor' in which:
        for rep_i in (1, 2):
            run_raw(attach(floor_batches(items, rep_i)), f'patch_floor_rep{rep_i}', generate, encode, out, stamp)
    stamp(f'patch readout done: {which}')


# ---------------------------------------------------------------- GPU glue (Kaggle only)
def vllm_generate(model_path, out, max_tokens, max_model_len=32768, thinking=False, tp=4, sampling=None, tag=''):
    """The registered engine settings of gemma_readout.run (TP 4, max_num_seqs 32, seed 0, greedy), with the
    output budget and window of the readout. sampling: extra SamplingParams fields (the sampled setting), a
    'seed' among them. Returns (generate, encode, decode, set_sampling)."""
    import os
    import threading
    import vllm
    from vllm import LLM, SamplingParams
    watchdog = threading.Timer(1800, lambda: (print('engine startup watchdog fired', flush=True), os._exit(3)))
    watchdog.start()
    llm = LLM(model=str(model_path), tensor_parallel_size=tp, max_model_len=max_model_len,
              gpu_memory_utilization=0.85, max_num_seqs=32, seed=0)
    watchdog.cancel()
    tok = llm.get_tokenizer()
    state = dict(params=SamplingParams(temperature=0.0, max_tokens=max_tokens, **(sampling or {})))

    def set_sampling(**kw):
        state['params'] = SamplingParams(max_tokens=max_tokens, **kw)

    def generate(convs):
        outs = llm.chat(convs, state['params'], chat_template_kwargs={'enable_thinking': thinking}, use_tqdm=False)
        return [(o.outputs[0].text, len(o.prompt_token_ids)) for o in outs]
    out.mkdir(parents=True, exist_ok=True)
    (out / f'environment{tag}.json').write_text(json.dumps(dict(
        vllm=vllm.__version__, model_path=str(model_path), max_tokens=max_tokens, max_model_len=max_model_len,
        thinking=thinking, tp=tp, sampling=sampling)))
    return generate, (lambda s: tok.encode(s, add_special_tokens=False)), tok.decode, set_sampling


# ---------------------------------------------------------------- G6.3 settings (§8, exploratory)
PARA = ['Which of the files above would you edit to fix this issue? Give up to 3 paths from the list, most likely first.',
        'To resolve the issue, which listed files need changes? Name up to 3, best guess first.',
        'Pick the files from this list that you would change to address the issue, at most 3, in order of likelihood.']
SETTINGS = ('think', 'sampled', 'para1', 'para2', 'para3', 'b8000', 'm26', 'e4')


def ask_for(setting):
    """Paraphrase settings replace the question line of ASK (guide G6.3); the JSON format lines are unchanged."""
    if setting.startswith('para'):
        return PARA[int(setting[4:]) - 1] + '\n' + g.ASK.split('\n', 1)[1]
    return g.ASK


def modal_choice(choices):
    """The most frequent choice over the draws; ties go to the earliest draw."""
    counts = Counter(choices)
    return max(choices, key=lambda c: (counts[c], -choices.index(c)))


def settings_items(r1, r2, r3, clean, gold):
    """C0 for every task, the H2-type cells of the registration-1 readout (coded as analyze_core) and their
    matched placebo lists."""
    c0 = {r['task']: r['choice'] for r in r1 if r['cond'] == 'C0'}
    key = lambda r: (r['task'], r['intervention'], r['seed'])
    h2 = [r for r in r2 if r['choice'] != 'INVALID' and c0[r['task']] != 'INVALID'
          and set(clean[r['task']][:5]) != set(r['shown']) and not g.is_forced(c0[r['task']], r['shown'])
          and g.recall5(r['shown'], gold[r['task']]) == g.recall5(clean[r['task']][:5], gold[r['task']])]
    keys = {key(r) for r in h2}
    it = [dict(id=f'C0|{t}', cond='C0', task=t, shown=clean[t][:5]) for t in sorted(clean)]
    for cond, rows in (('X', h2), ('P', [r for r in r3 if key(r) in keys])):
        it += [dict(id=f'{cond}|{r["task"]}|{r["intervention"]}|{r["seed"]}', cond=cond, task=r['task'],
                    intervention=r['intervention'], seed=r['seed'], shown=list(r['shown'])) for r in rows]
    return it


def engine_for(setting):
    """(max_tokens, max_model_len, thinking, tp, file budget, model) of a §8 setting."""
    if setting == 'think':
        return 4096, 32768, True, 4, g.TOKEN_BUDGET, 'main'
    if setting == 'b8000':
        return g.MAX_TOKENS, 49152, False, 4, 8000, 'main'
    model = setting if setting in ('m26', 'e4') else 'main'
    return g.MAX_TOKENS, 32768, False, (1 if setting == 'e4' else 4), g.TOKEN_BUDGET, model


def run_settings(settings, art, comp, model_path, out, gen_config=None):
    """§8 settings on the competition tasks, on one engine (they must share its configuration, e.g.
    para1,para2,para3,sampled: the prompt prefixes are shared, so vLLM's prefix cache serves them).
    Writes raw_set_<setting>[_<seed>].jsonl."""
    assert all(x in SETTINGS for x in settings), settings
    assert len({engine_for(x) for x in settings}) == 1, f'{settings} need different engines'
    out.mkdir(parents=True, exist_ok=True)
    stamp = stamper(out)
    rows = g.load_rows(art / 'paper/results/2026-11-08_extension_v3/per_task.jsonl.gz')
    clean, _, gold = g.rankings(rows)
    r1, r2, r3, _ = g.load_stages(art / 'paper/results/2026-11-10_gemma_readout')
    items = settings_items(r1, r2, r3, clean, gold)
    max_tokens, window, think, tp, budget, _ = engine_for(settings[0])
    generate, encode, decode, set_sampling = vllm_generate(model_path, out, max_tokens=max_tokens, max_model_len=window,
                                                           thinking=think, tp=tp, tag='_' + '_'.join(settings))
    selection = json.loads((art / 'paper/data/heldout_full_2026-11-07.json').read_text())['tasks']
    needed = defaultdict(set)
    for x in items:
        needed[x['task']] |= set(x['shown'])
    rendered, _, issues, repos = g.build_contexts(sorted(needed), needed, comp, selection, encode, decode, budget)
    stage = make_stage(generate, out, stamp)
    stamp(f'{settings}: {len(items)} items ' + str(dict(Counter(x["cond"] for x in items))))
    for setting in settings:
        ask = ask_for(setting)
        its = [dict(x, conv=g.conversation(repos[x['task']], issues[x['task']], x['shown'],
                                           {p: rendered[(x['task'], p)] for p in x['shown']}, ask=ask)) for x in items]
        if setting == 'sampled':
            cfg = {k: gen_config[k] for k in ('temperature', 'top_p', 'top_k') if k in (gen_config or {})}
            stamp(f'sampled with the model generation_config: {cfg}')
            for s in range(5):
                set_sampling(seed=s, **cfg)
                stage(its, f'set_sampled_{s}')
            set_sampling(temperature=0.0)
        else:
            stage(its, f'set_{setting}')
        stamp(f'{setting} done')
