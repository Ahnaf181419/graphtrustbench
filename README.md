# GraphTrustBench

This repository accompanies the paper
[**Same Score, Different Context: Call-Graph Errors, Retrieval Stability, and What Gemma 4 Does With Them**](paper/GraphTrustBench.md),
submitted to the Kaggle *Google – The Gemma 4 Developer Agent Paper Track*. It contains the code,
configurations, task lists, pre-registrations, deviations log, rankings, model outputs and aggregate
results from which every number in the paper is generated.

## Summary

The study follows call-graph errors along the chain from graph to patch on 115 tasks from the
Gemma 4 Developer Agent Competition (FastAPI, Rich and Requests). Call edges are deleted or rewired
at 10–40%, and, separately, the calls each task's tests execute are traced and added back.

- The supplied graphs miss 71% of the call edges the tests execute.
- Recall@5 barely changes under corruption, while the top five changes in about nine of ten cells
  at 40% rewiring, for a one-hop and a personalized-PageRank retriever alike.
- `gemma-4-31b-it-qat-w4a16-ct` changes its file choice in response, but no more often than after a
  matched random change.
- With the clean context the model resolves 6.5% of the 108 evaluable tasks; patch outcomes do not
  change measurably, although at this resolve rate the patch readout has little power.

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/GraphTrustBench.md` | The paper |
| `paper/figures/` | Figures 0 and 1 |
| `paper/prereg/` | Both pre-registrations, the deviations log and the frozen flip-risk score |
| `paper/scripts/` | Data audit, mapping, retrieval, corruption, call tracing, test harness and analysis |
| `paper/tests/` | Unit tests |
| `paper/config/`, `paper/data/` | Retrieval configuration and task lists with upstream tree identifiers |
| `paper/harness/` | Container image definition for running task tests offline |
| `paper/pycg/` | The PyCG patch and the study behind its settings |
| `paper/notebooks/` | Kaggle notebooks that ran the Gemma 4 readouts (4 × L4, vLLM 0.19.1) |
| `paper/results/` | Run manifests, rankings, decision-readout replies and run logs |
| `paper/reports/` | Data audits and the aggregate values behind every table and figure |

## Data availability

Section 4b of the competition rules does not permit the Competition Data to be made available to
non-participants. This repository therefore contains no competition corpus (task statements,
snapshots, graphs, embeddings or patches) and nothing derived from its labels: gold files,
fail-to-pass and pass-to-pass test sets, per-task metrics and outcomes, and the patches the model wrote
over the competition code are withheld. Only aggregate results are included.

Participants can regenerate the withheld files from their own copy of the competition data, placed at
the repository root; every script verifies its SHA-256 fingerprints. The commands for each run are
recorded in its `manifest.json` or `run_log.txt`. For the 115-task rankings:

```bash
python3 paper/scripts/prepare_pilot.py --data-root . --pilot paper/data/heldout_full_2026-11-07.json \
    --audit paper/reports/competition_audit_full_2026-11-06.json \
    --upstream-trees paper/data/heldout_full_upstream_trees_2026-11-08.txt --output-dir regen/mapping
python3 paper/scripts/run_localization_pilot.py --data-root . --pilot paper/data/heldout_full_2026-11-07.json \
    --audit paper/reports/competition_audit_full_2026-11-06.json --mapping-dir regen/mapping \
    --config paper/config/pilot_v3.json --output-dir regen/extension_v3
```

The analysis scripts read committed inputs only, so regenerated files are committed at the paths
named in the manifests before the analyses are rerun:

- `paper/scripts/context_stability.py`: one-hop rows of Table 1 and Figure 1;
- `paper/scripts/v7_g3_retrievers.py`: PageRank rows of Table 1, low corruption rates and the mitigation;
- `paper/scripts/gemma_readout.py paper/results/2026-11-10_gemma_readout`: Table 2 and hypotheses H1–H3;
- `paper/scripts/v7_g4_trace.py`: test sets, traced call edges and the repaired graph (H7), in Docker;
- `paper/scripts/v7_g7_evaluate.py`: evaluation of the generated patches, in Docker;
- `paper/scripts/v7_g8_fill.py`: every value of the second registration, written to
  `paper/reports/v7/fill_values_v7.json`.

The container image is built with
`docker build -t gtb-tasks:py3.12 -f paper/harness/Dockerfile.tasks paper/harness`.

## Pre-registration

The study was registered twice in the author's development repository before the corresponding
model runs. This repository is a clean export of that work, so the development commits are cited by
identifier and the registration files are reproduced as committed.

| Registration | Development commit | Committed (UTC+6) | File |
| --- | --- | --- | --- |
| First (H1–H3) | `b1044d0` | 2026-11-10 22:10 | `paper/prereg/READOUT_PREREGISTRATION_V5.md` |
| Second (H4–H8) | `4500dc4` | 2026-11-11 12:36 | `paper/prereg/preregistration_v2.md` |

The first registration carries one input-handling amendment, made before the main run and recorded
in the file itself. Every later departure from either registration is listed, with its reason and
effect, in `paper/prereg/deviations.md`.

## Running the tests

The code requires Python 3.12 (the rewiring sampler is pinned to Python 3.12 semantics). The
retrieval code uses the standard library only; the analyses and tests also use the packages pinned
in `requirements.txt`. No GPU is required.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s paper/tests -v
```

Tests that require the competition data, its labels or the Docker image are skipped when these are
absent.

## License

The code, configurations and manifests are released under the Apache License 2.0 (see `LICENSE`).
The competition corpus and all material derived from its labels are excluded.
