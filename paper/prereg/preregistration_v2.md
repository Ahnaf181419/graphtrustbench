# GraphTrustBench: second pre-registration (H4–H8)

**Status:** draft. Commit this file, the `gtb6` package and the frozen flip-risk score (`prereg/fliprisk_frozen.json`) **before any new Gemma run, test run or post-cutoff task build**. Record the commit hash and its UTC time as `{PREREG2_COMMIT}` in the paper.

**Relation to the first registration (commit b1044d0):** H1–H3, their cells, rules and verdicts stay exactly as registered and reported. This document adds hypotheses H4–H8 and fixes every design choice for the new experiments. Nothing here re-analyses H1–H3.

---

## 1 What is confirmatory

| Label | Claim tested | Data | Test |
| --- | --- | --- | --- |
| H4a | H2 rerun: cells with changed membership, unchanged Recall@5 and C0's choice still shown flip more often than the engine floor (C0r) | post-cutoff tasks, decision readout | one-sided bootstrap p, Holm |
| H4b | H3 rerun: the same cells flip more often than their matched placebos | post-cutoff tasks, decision readout | one-sided bootstrap p, Holm; registered reading rule |
| H5 | In the H2/H4a cell set, the patch resolve rate is equivalent to clean, within ±5 percentage points | competition + post-cutoff tasks, patch readout | two one-sided tests (90% interval inside ±0.05) |
| H6 | Cells where Recall@5 drops resolve less often than clean | competition + post-cutoff tasks, patch readout | one-sided bootstrap p, Holm |
| H7 | The repaired graph gives higher one-hop Recall@5 than the supplied graph | competition tasks with a successful trace | one-sided bootstrap p, Holm |
| H8 | The frozen flip-risk score predicts flips on new tasks: AUC above 0.5 | post-cutoff tasks, decision readout | task-cluster bootstrap, one-sided p, Holm |

**Holm family:** H4a, H4b, H6, H7 and H8, at α = 0.05. H5 is decided on its own by two one-sided tests at α = 0.05 each, i.e. by the 90% interval.

Everything else is **exploratory** and will be labelled so:
- the PPR and agentic retrievers' RQ1–RQ2 results;
- the model and setting variants;
- the mitigation;
- the agent readout;
- the low corruption rates;
- the fragility and exposure analyses;
- the flip-risk score's fit on competition data;
- the pre/post-cutoff split of the registration-1 cells: the H2 flip rate and the H3 placebo delta, each computed separately over cells whose task `created_at` is before 2025-03-01 versus on or after it, with the same estimands and intervals as H2/H3 (measured task counts: 28 before, 87 on or after). Registered here before anyone computes it.

---

## 2 Data

### 2.1 Competition tasks
The same 115 tasks as registration 1. For the patch readout, each task's repository is cloned at its base commit. Registration 1 verified that the tree hash matches the snapshot. The reference patch is the competition's.

### 2.2 Post-cutoff tasks (`gtb6/builder.py`)
This is a **fresh held-out set from a second extractor (PyCG)** that tests H4 and H8 out of sample; differences between task sets are attributable to the extractor or the tasks (§2.3). It is not a memorization test, and nothing here claims one. For context (measured 2026-11-11 from `tasks.jsonl`): 87 of the 115 competition tasks were created on or after 2025-03-01, and 28 before 2025-01-01.

- **Repositories:** fastapi/fastapi, Textualize/rich, psf/requests.
- **Cutoff:** **1 March 2025**. This is later than the January 2025 training cutoff given in the Gemma 4 31B model card.
- **A pull request qualifies if:**
  - it merged on or after the cutoff;
  - its body says it closes, fixes or resolves an issue (the first such issue is used);
  - that issue is a real issue, not itself a pull request, and was opened on or after the cutoff;
  - its diff changes at least one non-test `.py` file and at least one test file;
  - `derive_tests` finds at least one fail-to-pass test;
  - the environment builds;
  - PyCG produces a graph.
- **Fallback:** if fewer than 40 tasks pass with the issue-date rule, the issue-date rule is dropped (merge date only). Both counts are reported.
- **Order and cap:** by merge date; the first 100 tasks that pass every check.
- **Task contents:**
  - problem statement = issue title + body;
  - base commit = the pull request's base SHA;
  - gold files = the non-test `.py` files the pull request changes.
- **Freezing:** the task list and its manifest (repository, pull request, issue, base SHA, gold files, F2P and P2P tests, exclusion reasons) are committed as `prereg/postcutoff_tasks.json` **before** any Gemma run on these tasks. Building tasks uses no model output.

### 2.3 Graphs and candidates for the post-cutoff tasks
- **Candidates:** every function, method and class definition in the package (`extract_definitions`). Node ID = `module.qualname`. Closures are excluded.
- **Graph:** PyCG (pinned version recorded in the lock file) over all package modules. Only caller → callee pairs between candidates are kept (`to_succ`).
- **Lexical scoring:** the same BM25 code and tokenizer as the competition tasks (k1 = 1.2, b = 0.75). The issue is the query; each document is node ID + path + definition text.
- **Extractor confound:** these graphs come from a different extractor than the competition graphs. Any difference between task sets is reported as possibly due to either cause.

---

## 3 Retrievers

All retrievers use the five BM25 seeds and the same candidates and cutoffs. Each file takes its best node's rank, and unvisited candidates follow in BM25 order (`rank_files`).

- **One-hop:** the **registered registration-1 implementation** (`localization_retrieval.py`) is used for the one-hop retriever and the synthetic corruption; `gtb6` does not re-implement them. The committed anchor test (Task 2) pins `gtb6`'s data model against the committed extension rankings, and `gtb6.retrievers` supplies only PPR and the mitigation.
- **PPR (exploratory):**
  - restart probability 0.15 to the seeds' normalized BM25 mass;
  - run on the subgraph induced by mapped candidates;
  - dangling mass returns to the restart vector;
  - iterated until the L1 change is below 10⁻⁸, or for 100 iterations.
- **Agentic (exploratory):**
  - run on the clean graph and on every 40% cell (deletion and rewiring, three seeds);
  - Gemma 4 31B (same build as registration 1), thinking off, greedy decoding;
  - up to **8** `neighbours` calls, each showing up to 20 callees;
  - then `done` with up to 10 files;
  - final ranking: the files it lists (candidates only, deduplicated), then the rest in BM25 order;
  - a reply with no tool call is re-asked once; a second failure ends the run with an empty list;
  - prompts are given verbatim in `gtb6/agent.py` (`AG_SYSTEM`, `AG_TOOLS`).
- **Mitigation (exploratory):**
  - average node scores over **10** graphs with edge-hash dropout at rate **0.2**;
  - the same edge is dropped in every graph condition of a task;
  - applied to the one-hop retriever on clean and corrupted graphs.
  - Report:
    - top-five membership change (mitigated corrupted vs mitigated clean);
    - Recall@5 (mitigated clean vs unmitigated clean);
    - wall-clock cost ratio.

---

## 4 Graph conditions

- **Synthetic:** as in registration 1: deletion and rewiring at 10%, 20% and 40%, seeds 13, 37 and 101. The low rates of 1%, 2% and 5% are exploratory, with the same seeds.
- **Traced (real) errors (`gtb6/tracer.py`):**
  - For each task, check out the base commit and apply the reference patch's test part.
  - Run the task's F2P and P2P tests under the tracer. This records Python-level calls whose caller and callee both live in the package source.
  - Map the traced edges to graph IDs with the same alias rules as the candidates. Unmapped edges (closures, decorator wrappers, module-level code) are counted and reported.
- **Run-time miss rate:** the share of mapped traced edges, as unique node pairs, absent from the supplied graph's collapsed edges. It is reported pooled and as the median per task.
- **Kinds of miss (`gtb6/misskinds.py`):** each miss is classified by its callee, in priority order dunder > decorated > method > function, plus "unresolved".
- **Repaired graph:** the supplied graph plus every mapped traced edge it misses.
- **Exclusions:** tasks whose environment fails, or whose trace records no mapped edge, are excluded from H7 and counted.

---

## 5 Readouts

### 5.1 Decision readout
- The same prompt, file rendering (B = 2,500), model, engine and decoding as registration 1.
- The same conditions (C0, C0r ×3, CR, CN, corrupted 40% cells, matched placebos) and the same coding rules.
- Analysis by `gemma_readout.analyze_core` — registration 1's own analysis, refactored to take any task set; a characterization test pins it to the committed `readout_values.json` byte for byte.
- Run on the post-cutoff tasks for H4 and H8.

### 5.2 Patch readout
- **Model:** the same 31B build, thinking off, greedy decoding, at most 4,096 output tokens.
- **Prompt:** verbatim in Appendix A.
- **Files shown:** the context's five files, each as its full source text in original order, cut at a line boundary to at most 4,800 tokens (so at most 24,000 in total). A file's rendering does not depend on the other files, so it is identical across conditions.
- **Conditions:**
  - C0 (clean);
  - C0r: two repeats at different batch positions, the patch floor (exploratory reference);
  - every 40% cell whose ordered top five differs from clean (the decision readout's cells);
  - each matched placebo, using the same placebo list as the decision readout;
  - the repaired-graph top five, one per competition task with a successful trace (supplied vs repaired, exploratory).
- **Evaluation (`gtb6/repo_ops.evaluate`):**
  - extract the last ```diff block, or failing that the text from the first diff header;
  - drop edits to test files;
  - apply with `git apply` (-p1, then -p0), then with GNU `patch --fuzz=3`;
  - apply the reference test part on top;
  - run the F2P and P2P tests. A cell is resolved only if every one passes.
  - Fail-to-pass and pass-to-pass tests come from `derive_tests`, scoped to the test files the reference patch touches.
- **Exclusions:** tasks with no F2P test or a failed environment are excluded and counted. No exclusion depends on model output.

### 5.3 Agent readout (exploratory)
- **Tasks:** **50** competition tasks with at least one F2P test, drawn with `random.Random(2026).sample(sorted(eligible), 50)`.
- **Conditions:** the clean graph, 40% rewiring at seed 13, and the repaired graph.
- **Model setup:** Gemma 4 31B with thinking on.
- **Start:** the agent begins with the one-hop top five under that graph.
- **Tools** (prompts verbatim in `gtb6/agent.py`):
  - `read_file` (200 lines);
  - `search` (BM25 top 10);
  - `neighbours`;
  - `submit`.
- **Limit:** at most 20 calls.
- **Context limit:** when a conversation exceeds 28,000 tokens, tool observations older than the last eight are replaced by "[earlier output removed]".
- **Outcomes:**
  - resolve rate;
  - number of calls;
  - recovery, defined as reading at least one file outside the initial top five, among runs whose top five differs from clean.

---

## 6 Estimands and tests

**Common rules:**
- Rates are averaged within task, then across tasks.
- Intervals are percentile intervals from 10,000 task-level paired bootstrap resamples, seed 2026.
- A one-sided p-value is the share of resampled means at or below zero (or at or above zero, for "less than").
- Each test needs at least five eligible tasks; H5 needs at least 20. Below that, the result is reported as descriptive.

**H4a.** Per post-cutoff task: the mean flip rate over the H2-type cells minus that task's C0r flip rate. Mean greater than 0?

**H4b.** Per post-cutoff task: the mean over the H2-type cells of (flip in the cell − flip in its matched placebo). Mean greater than 0? If H4b is not rejected after Holm, the registration-1 reading rule applies:
- placebo-heavier if the 95% interval lies below 0;
- generic if the 90% interval lies within ±0.10;
- inconclusive otherwise.

**H5.** Per task (both task sets pooled; also reported per set): the mean over H2-type cells of (resolved in the cell − resolved under C0). Equivalent if the 90% interval lies within ±0.05. If the interval is wider than 0.10 in total, the result is reported as "not established (underpowered)".

**H6.** Per task: the mean over cells whose Recall@5 is below clean (forced cells included) of (resolved in the cell − resolved under C0). Mean less than 0?

**H7.** Per competition task with a successful trace: one-hop Recall@5 on the repaired graph minus Recall@5 on the supplied graph. Mean greater than 0? The change in top-five membership is reported alongside, descriptively.

**H8.**
- Cells: all valid, non-forced post-cutoff cells from the 40% decision readout (order-only and membership-changed).
- Score: the frozen score (Section 7).
- Outcome: flip.
- Test: AUC with a task-cluster bootstrap. One-sided p = the share of bootstrap AUCs at or below 0.5.

**Net gold change** (descriptive, as in registration 1): reported for the post-cutoff tasks with the sign-flip permutation p.

---

## 7 Flip-risk score (frozen in this commit)

- **Features** (`gtb6/fliprisk.FEATURES`), all computed before the model runs:
  - the number of incoming files;
  - log(1 + the smallest clean rank among incoming files, capped at 50);
  - the number of incoming files placed above C0's choice;
  - C0's choice's position in the corrupted list (6 if not shown);
  - the shift in that position;
  - 1 − RBO;
  - log(1 + k_eff).
- **Model:** logistic regression with L2 penalty (C = 1.0) on standardized features, solved by Newton's method in numpy (the same strictly convex objective as scikit-learn's `LogisticRegression`; scikit-learn is not installed here). The two agree to 3 × 10⁻⁹ in max |Δcoef| on the fit data.
- **Fit data:** all valid, non-forced 40% cells of the registration-1 readout (competition tasks) — realized as **530 cells** (421 H2, 77 order-only, 32 recall-changed; 78 forced cells excluded), flip rate 0.126.
- **Frozen output:** the feature means, scales, coefficients and intercept, committed as `prereg/fliprisk_frozen.json`.
- **Exploratory, on the fit data:** task-cluster bootstrap intervals for the coefficients (in-sample AUC 0.645; no coefficient interval excludes zero). Labelled as such.

---

## 8 Exploratory settings (decision readout, competition tasks, H2-type cells vs placebos)

Each setting reruns C0, the H2-type cells and their placebos under that setting, and flips are coded against that setting's own C0.

| Setting | Change from the main run |
| --- | --- |
| Thinking on | `chat_template_kwargs={"enable_thinking": True}`, max tokens 4,096; the parser takes the last JSON object |
| 26B-A4B, E4B | the Gemma 4 variations of those names, as available on Kaggle; exact IDs recorded |
| Sampled | the model card's recommended sampling, 5 draws (seeds 0–4); first choice = modal choice, ties to the earliest draw |
| 8,000-token files | B = 8,000, max_model_len 49,152 |
| Prompt paraphrases | the three wordings in Appendix A, pooled |

---

## 9 Deviations and stopping

- Any change after this commit goes into `prereg/deviations.md`, with:
  - the date;
  - the reason;
  - the affected analyses;
  - where possible, the result both ways.
- **Run order if GPU quota runs short:** decision readout on the post-cutoff tasks → patch readout → settings → agent readout. Whatever is not run is reported as not run.
- **Agent readout:** if not all 50 tasks complete, the completed tasks are reported, in the sampled order.

---

## 10 Substitutions, self-rerun and release

- **Registered code is reused, not re-implemented:** the one-hop retriever and the synthetic corruption are registration 1's `localization_retrieval.py`; the H4 analysis is registration 1's `gemma_readout` (via `analyze_core`); BM25 for the post-cutoff candidates is `localization_retrieval.bm25`, so both task sets share one tokenizer.
- **The logistic regression is numpy Newton–IRLS** (§7).
- **Upstream test suites and model-written patches run in Docker containers** (`gtb6.docker_runner`); `repo_ops.evaluate` refuses to run without an injected runner, so nothing executes on the host by default.
- **Self clean-room rerun:** there is no teammate to rerun the pipeline (solo participant). A self clean-room rerun — a fresh clone and fresh Kaggle notebooks rerunning the post-cutoff decision readout — replaces `RERUN_MATCH` and is reported as a self-rerun, with its agreement described.
- **Release scope:** the tracer, the scripts and the post-cutoff tasks' graphs are released publicly. The competition tasks' repaired graphs derive from competition graphs and **stay private** (competition rules §4b). Gold labels and per-task metrics never enter any public artifact.
- **Result phrases:** the guide's two-versions notes file does not exist, so every pre-written outcome sentence in the paper is written strictly from the measured verdicts under the "only measured claims" rule.

---

## Appendix A — prompts

**Patch readout.**

System:
> You are a software engineer. Write a patch that resolves the GitHub issue.

User:
> Repository: {repo}
>
> Issue:
> {issue}
>
> Relevant files (in the order given):
>
> [1] {path}
> ```python
> {file text, cut at a line boundary to 4,800 tokens}
> ```
> … (five files)
>
> Write a patch that resolves the issue. Reply with a single unified diff in a ```diff block, with paths relative to the repository root and a/ and b/ prefixes. Change only what is needed.

**Decision readout: paraphrases** (each replaces the final instruction; the JSON format line is unchanged):
1. "Which of the files above would you edit to fix this issue? Give up to 3 paths from the list, most likely first."
2. "To resolve the issue, which listed files need changes? Name up to 3, best guess first."
3. "Pick the files from this list that you would change to address the issue, at most 3, in order of likelihood."

## Appendix B — files in this commit
- `paper/prereg/preregistration_v2.md` (this file) and `paper/prereg/deviations.md`
- `paper/prereg/fliprisk_frozen.json` — the frozen score, fitted by `paper/scripts/v7_g1_freeze.py` on the 530 cells; the exploratory aggregates are in `paper/reports/v7/g1_values.json`
- `paper/scripts/gtb6/` — `stats.py`, `retrievers.py` (PPR, the mitigation, `rank`, `bm25_order`), `exposure.py`, `fliprisk.py`, `tracer.py`, `misskinds.py`, `outcomes.py`, `repo_ops.py`, `docker_runner.py`, `corpus.py`, `agent.py`, `builder.py`. There is no separate `analysis.py`: registration 1's analysis lives in `paper/scripts/gemma_readout.py` (`load_stages`, `analyze_core`), reused as §5.1 states.
- Tests: `paper/tests/test_gtb6_stats.py`, `test_gtb6_retrievers.py`, `test_gtb6_exposure.py`, `test_gtb6_fliprisk.py`, `test_gemma_readout.py`, `test_gtb6_repo_ops.py`, `test_v7_g1.py`, `test_gtb6_tracer.py`, `test_gtb6_agent.py`, `test_gtb6_builder.py` — **179 tests, all passing** (`python3 -m unittest discover -s paper/tests`), alongside the unchanged registration-1 suite.
- The GPU stages run as Kaggle notebooks (Plans 2–4); no notebook is part of this commit.
