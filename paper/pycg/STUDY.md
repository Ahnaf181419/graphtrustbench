# PyCG on the post-cutoff repositories (2026-11-11)

PyCG is the registered second extractor (§2.3). Measured on the current upstream heads
(fastapi e1144f695, rich 3a109ccd, requests 3c7baaa0) before any post-cutoff task was built.

- **pycg 0.0.8** (PyPI wheel) ships its package as `PyCG/` but imports `pycg`, so it cannot run on a
  case-sensitive filesystem. **pycg 0.0.7** works; it needs `pkg_resources` (`setuptools==75.8.0`).
- **requests** crashes PyCG 0.0.7: `_get_fun_defaults` indexes only `args.args` while `args.defaults`
  also covers positional-only parameters (PEP 570). Fixed by `pycg-0.0.7-posonly-defaults.patch`
  (two lines), applied after a fresh, non-hardlinked install.
- **rich** does not reach PyCG's fixed point: the unbounded run did not finish in 30 min, and capped
  runs with 4, 5, 6 and 8 iterations did not finish in 25 min. With 3 iterations PyCG's output explodes
  (rich 131,025 edges in 375 s; requests 7,007 in 127 s, vs 2,086 with 2).
- **Hash seed.** PyCG's result depends on Python's hash seed even at the fixed point (fastapi: 2,146
  edges with PYTHONHASHSEED=0, 1,980 with 1). With a fixed seed, repeated runs are identical (sha256).
- **Rule used:** `--max-iter 2` for every repository, `PYTHONHASHSEED=0`. Times: fastapi 10 s, rich 35 s,
  requests 6 s. Raw timings: `study_2026-11-11.log` (columns: repo, iterations, seed, run, exit code,
  seconds, keys, edges, hash). Capped graphs are less complete than a fixed point would be (fastapi:
  1,627 edges at 2 iterations vs 2,146 at the fixed point); this is a stated limitation.
