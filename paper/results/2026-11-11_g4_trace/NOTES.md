# G4 run notes (2026-11-11)

- Driver: `paper/scripts/v7_g4_trace.py` (commits 22809e9, 72dc632, 97ea299); per-task records in `work/` (git-ignored), raw traces in `traces/`.
- The run was interrupted twice by session ends and resumed (the driver skips finished tasks). 112 tasks ran before the container memory cap (97ea299); rich_3480, rich_3772 and rich_4006 ran under it (6 GB, no swap).
- **Memory exclusions ("no trace output").** rich_3480 grew a test process to 19 GB before the cap (host OOM killer, 2026-11-11 17:02); under the cap, it and rich_3772 are killed at 6 GB. A check of rich_3772 without the tracer hits the same 6 GB at the base commit with the test part applied, so the unfixed bug exhausts memory; it is not the tracer. rich_4006 hit the 1,800 s timeouts. All three are exclusions under §4 ("tasks whose environment fails, or whose trace records no mapped edge").
- **"No mapped edges".** fastapi_15661: its tests exercise `scripts/prepare_release`, outside the package. requests_6757: its test part imports `requests.compat.is_urllib3_1`, which does not exist at the base commit, so no test runs under the tracer (and all 322 tests count as fail-to-pass, SWE-bench convention).
- **Miss check.** Spot check rich_2725: the traced edge `rich._wrap.divide_line -> rich.cells.cell_len` is absent from the supplied graph, whose listed callees of `divide_line` are `loop_last`, `words`, `chop_cells` and `Text.wrap`; `rich.cells.cell_len` is a candidate. Misses are extraction gaps, not mapping artefacts.
- Hand check (guide G4.3): `handcheck.txt`.
