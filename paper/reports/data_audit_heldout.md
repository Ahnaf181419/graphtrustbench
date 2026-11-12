# Competition dataset audit

Raw-output checkpoint: `73ae5508d3ac7684807afb415e54e377f11b43fd`.

Status: **passed_schema_preflight**. 129/129 structurally usable tasks across 4 repositories.

| Repository | Tasks |
| --- | ---: |
| Textualize/rich | 48 |
| encode/httpx | 1 |
| fastapi/fastapi | 67 |
| psf/requests | 13 |

- 127 distinct graphs; 127 distinct repository/base-commit pairs.
- 367,985 graph nodes; 452,588 edges, counted once per graph. Edge types: `{'calls': 452588}`.
- Missing required assets: `{'embedding': 0, 'graph': 0, 'snapshot': 0}`; tasks with graph errors: 0.
- Consumed asset bytes: 22,386,440,576; tasks.jsonl and auxiliary wheels are additional.
- Audit runtime: 100.51 seconds; potentially external archive links: 18. Links are not followed.

Upstream worktree-tree identity and patch preimages are verified for these 24 pilot tasks. This verification has not been extended to the other 105 tasks.

Candidate mapping: 52,906/54,197 nodes (97.62%). Source-verified alias bindings: 7,735; unresolved ambiguous aliases: 303. Exact-ID body conflicts encountered: 187. Changed-file coverage: 28/35; all 7 unavailable files remain in recall denominators.

Mapping statuses: `{'precise': 15, 'file_only': 9}`. Symbol labels use the smallest available enclosing definition; class stubs do not establish exact function relevance. File-level evaluation is primary.

Pilot repository distribution: `{'fastapi/fastapi': 8, 'psf/requests': 8, 'Textualize/rich': 8}`. Sampling seed: 2027; selection: repository round-robin, one base commit per draw; excludes every development task and every task sharing a development base commit; no gold used.

Shared repository/base-commit groups, which must stay within one future split: `{'Textualize/rich@4d6d631a3d2deddf8405522d4b8c976a6d35726c': ['rich_3882', 'rich_3894'], 'psf/requests@7a13c041dbef42f9f3feb14110f02626f6892e9a': ['requests_6629', 'requests_6589']}`.

Dataset fingerprint: `e39fbcca3ccd76d39582587aa3c3d6e32e1adb5cfa011ed4845d55267c3afc7d`. Graphs have symbol IDs but no explicit source positions; supplied unperturbed edges are not independently established as semantically correct.

Sources: [raw audit](competition_audit_full_2026-11-06.json), [pilot selection](../data/heldout_2026-11-08.json), [configuration](../config/pilot_v3.json), [run manifest](../results/2026-11-07_heldout_v3/manifest.json), [mapping manifest](../results/2026-11-07_heldout_mapping/manifest.json).
