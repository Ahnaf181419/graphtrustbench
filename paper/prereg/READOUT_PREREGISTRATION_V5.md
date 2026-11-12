# Gemma 4 readout — pre-registration (paper v5, RQ3)

Committed before the main run. The pilot used development tasks only (below); no extension-task prompt was sent to the model before this commit. The code is `paper/scripts/gemma_readout.py` at this commit (run on Kaggle by `paper/notebooks/gemma_readout.ipynb`), and the analysis is `gemma_readout.analyze` in the same file.

## Model, engine, decoding

- `gemma-4-31b-it-qat-w4a16-ct`, Kaggle model `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`.
- vLLM 0.19.1 from the competition wheelhouse (`metric/gemma-4-developer-agent-wheelhouse`), the official getting-started image, 4×L4, `tensor_parallel_size=4`, `max_model_len=32768`, `gpu_memory_utilization=0.85`, `max_num_seqs=32`, `seed=0` (the defaults, 0.90 and 256, ran out of memory while vLLM warmed up its sampler on the first pilot attempt).
- Chat template with `enable_thinking=False`; greedy (`temperature=0`), `max_tokens=256`. vLLM's default prefix caching stays on.

## Inputs

- 115 extension tasks; one-hop rankings from the committed `2026-11-08_extension_v3/per_task.jsonl.gz`; 40% corruption only.
- Prompt: system and user text exactly as `SYSTEM`, `ASK` and `conversation()` define them. The user text holds the repository name, the issue statement, and the shown files in order, each as `[i] path` followed by its rendering.
- Rendering (identical in every condition): the file's mapped definitions in clean-BM25 order (ties by node ID), each as `# <node id>` plus its source. A definition whose spans lie inside an already rendered definition is skipped. The text is cut to **2,500 Gemma tokens** at a line boundary. The rendered line spans are logged.
- Labels never enter a prompt. They are used only to skip Recall@5-changed cells and to draw non-gold placebo files.

## Conditions

As in paper §4.4: C0 (clean top five); CR (reversed); CN (issue only); X (every 40% cell whose ordered top five differs from clean); P (matched placebo for each X cell with changed membership, unchanged Recall@5 and C0's choice still shown: each incoming file replaced in place by a draw seeded with `task|intervention|seed` from clean ranks 6–20, non-gold, not already shown; cells with too small a pool are skipped and counted); C0r (C0 three more times, each batch shuffled with 115 filler prompts from X and P, seeds 101–103).

## Coding

First choice = first listed path that is shown; `NONE` for an empty list; `OFF-LIST` if no listed path is shown; `INVALID` if no parseable `{"files": [...]}` object (the last one wins). An INVALID reply is re-asked once with `RETRY_SUFFIX`; if it is still invalid, it is excluded and counted. A flip is a first choice different from C0's. A cell is forced when C0's choice is a path no longer shown; forced cells are reported but not tested.

## Hypotheses and decisions

Exactly as paper §4.4–4.5. In brief:

- Flip rates are averaged within task, then across tasks.
- Intervals come from 10,000 task-bootstrap resamples (numpy seed 2026).
- **H1:** order-only minus CR, with the 90% interval inside ±0.10.
- **H2:** H2 cells minus the C0r floor, one-sided.
- **H3:** H2 cells minus their paired placebos, one-sided.
- H2 and H3 use Holm at 0.05. If H3 fails, the verdict is placebo-heavier (95% upper bound < 0), generic (90% interval inside ±0.10) or inconclusive.
- Each hypothesis needs ≥ 5 tasks, otherwise it is reported as descriptive.
- Sensitivity analyses: drop tasks whose issue-only choice is gold; split H3 by intervention.
- **R20:** the per-task mean of corrupted minus clean Hit@1 over all 40% cells, with unchanged cells reusing C0's answer, on the 110 tasks with a retrievable gold file. It gets a bootstrap interval and an exact sign-flip permutation p (100,000 draws).
- **Gold visibility:** for gold files in C0's top five, whether all, some or none of the reference patch's pre-image edited lines fall within the rendered spans.

## Pilot (development tasks only, not analysed)

Ten development tasks (`pilot_2026-11-06.json`, first ten by ID) with stages C0, CR and CN only: 30 prompts on Kaggle kernel `mahnafsarker/graphtrustbench-v5-gemma-readout` v1 (2026-11-10, about 16:00 UTC).

- 30 of 30 replies parsed, with 0 retries and 0 INVALID.
- Replies come as fenced ```json blocks with no thought text.
- One reply kept writing after its object until it hit `max_tokens`, which the last-object rule covers.
- Prompts were 9,070–12,723 tokens (mean 11,181). Stage 1 took 170 s.
- First choices: 8 of 10 C0 were a shown path and 2 were NONE. For CR, 6 were a shown path, 3 NONE and 1 OFF-LIST.
- No prompt, budget or coding rule changed after the pilot.

Earlier attempts produced no model output:

- v1/v2 of the deleted kernel `graphtrustbench-v5-gemma` failed on a missing dataset input, then on vLLM's spawned workers re-running the driver's setup. The driver is now guarded by `__main__`.
- v3 ran out of memory in vLLM's sampler warm-up and left the session hung for about 5 h until it was deleted. That led to `max_num_seqs=32`, `gpu_memory_utilization=0.85` and a 30-minute engine-startup watchdog.

## Amendment before the main run (I/O only)

Main-run attempt v2 (16:10 UTC) failed before any prompt was built. Kaggle had decompressed the dataset's `ext_per_task.jsonl.gz`, so the expected path did not exist. The notebook driver now re-gzips the file into the committed path after asserting that its SHA-256 equals the run manifest's `per_task_sha256` (`4b6531…a67f`). `gemma_readout.py`, the prompts and the analysis are unchanged.
