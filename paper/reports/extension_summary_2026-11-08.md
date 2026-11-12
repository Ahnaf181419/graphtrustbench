# Post-hoc extension results — November 8, 2026

This is a post-hoc 115-task extension using unchanged protocol 0.3. It includes the original 24-task evaluation; the two must not be pooled or described as independent replications. There are 64 FastAPI, 41 Rich and 10 Requests tasks, with no HTTPX task. All 115 source trees and patch preimages passed. Two independent executions each completed 4,370 observations and reproduced all substantive predictions, metrics and summaries exactly.

## Accuracy effects

Each row below contains 115 tasks and 345 technical-repeat cells. Seeds are averaged within task before means and uncertainty. Intervals are exploratory task-bootstrap intervals, unadjusted for multiplicity; they do not establish generalization across repositories.

| Intervention | Severity | Δ Recall@5 | Exploratory 95% task-bootstrap interval | Recall@5 changes / cells |
| --- | ---: | ---: | --- | ---: |
| missing | 10% | -0.0043 | [-0.0145, +0.0058] | 5/345 |
| missing | 20% | -0.0058 | [-0.0188, +0.0072] | 10/345 |
| missing | 40% | -0.0029 | [-0.0345, +0.0311] | 25/345 |
| spurious | 10% | +0.0041 | [-0.0130, +0.0229] | 12/345 |
| spurious | 20% | -0.0117 | [-0.0423, +0.0161] | 25/345 |
| spurious | 40% | -0.0309 | [-0.0714, +0.0095] | 45/345 |

The declared primary extension endpoint, 40% rewiring, is -0.0309, CI [-0.0714,+0.0095]; its exact sign-test p is 0.0614. All six intervals include zero. The 10% rewiring estimate is positive, while deletion does not worsen monotonically with severity. Thus the directional/monotonic pattern of the original 24-task subset does not hold across this broader selection. This is not evidence of no effect: intervals permit both harm and improvement, and some tasks improve under corruption. BM25's complete rankings remain invariant across graph interventions.

## Distinct outcome counts

| Intervention | Cells | Recall@5 changed | Top-five set changed | Top-five order changed | Full ranking changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| missing | 1035 | 40 | 446 | 662 | 786 |
| spurious | 1035 | 82 | 726 | 750 | 916 |

## Coverage and interpretation limits

Source binding covers 339,996/343,191 task-node instances. Of 248 changed-file labels, 58 have no candidate; these remain in all denominators. Five tasks have no retrievable gold file. The macro candidate/cutoff Recall@5 ceiling is 0.8800 (mean min(5, available gold)/total gold), not a measured method score. Together the development and extension selections cover all 127 distinct repository/base-commit groups in the 129-task corpus; two other task records share groups and were not evaluated as separate tasks here.

The extension remains limited to three Python repositories, one supplied edge type and a shallow traversal. It supplies no model/repair evidence and does not test naturally occurring graph errors. The original 24-task scores remain the primary frozen evaluation; the new findings are reported as an overlapping post-hoc extension. No confidence interval can be promised to tighten merely by adding tasks.

## Reproduction and provenance

Run commands and source versions are in the two run manifests. Independent comparison is recorded in extension_repeatability_2026-11-08.txt. Lossless byte hashes and sizes are in extension_storage_2026-11-08.txt. Compressed raw streams are authoritative committed copies; local uncompressed files are retained but ignored. The source corpus is unchanged and is not included in the release. No PDF, Word document or submission archive was regenerated.
