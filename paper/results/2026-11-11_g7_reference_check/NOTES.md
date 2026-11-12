# G7 harness check: reference code patches as model replies

Input: each competition task's reference code patch (code part of `patch`; tests split off by
`repo_ops.split_reference`) wrapped as ```` ```diff ... ``` ```` in a reply with `cond` = `REF`.
The input file is rebuilt from `tasks.jsonl` and is not committed (competition data).

    python3 paper/scripts/v7_g7_evaluate.py --out paper/results/2026-11-11_g7_reference_check --raw raw_reference.jsonl --workers 3

Result (`summary_g7.json`): 115 tasks, 7 excluded (3 no F2P test, 4 base run produced no outcomes),
104 / 108 resolved, 0 crashed. The 4 failures (fastapi_13713, 14258, 15280, 15800) were the
harness: `extract_diff` closed the fenced block at a ```` ``` ```` inside a docstring the patch adds.
Fixed (fence only at line start; see `paper/prereg/deviations.md`). Of the 115 replies, the fix
changes the extracted diff of exactly these four; they were rerun into `fence_fix/`: 4 / 4 resolved.
Harness check after the fix: **108 / 108** evaluable reference patches resolve.
