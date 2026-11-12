# G4 re-derivation after the crashed-run fix (deviations.md, 2026-11-11)

`v7_g4_trace.py --tasks <35 ids>` with `repo_ops.read_outcomes` returning None for a missing outcomes file.
The 35 tasks are those whose first F2P or P2P list was empty (the only lists a crashed run can produce).
G7 uses `tests_competition.json` here in place of the first run's entries for these tasks. Only the test sets
are committed; the traces, repaired graphs and `fill_g4.json` of this subset are not used (H7 is reported from
the first run, `paper/results/2026-11-11_g4_trace/`).

| task | first run (F2P / P2P, reason) | re-derived |
| --- | --- | --- |
| requests_6592 | 0 / 0, no F2P test | base run produced no outcomes |
| rich_3480 | 109 / 0, no trace output | base run produced no outcomes |
| rich_3772 | 22 / 0, no trace output | base run produced no outcomes |
| rich_4006 | 67 / 0, no trace output | base run produced no outcomes |

The other 31 tasks are identical. Tasks evaluable in the patch readout: 108 of 115 (first run: 111).
