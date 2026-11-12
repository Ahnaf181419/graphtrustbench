# Competition dataset audit

Raw-output checkpoint: `95f60f4fc75a0997014863ee65848fd370e4f07d`.

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

Upstream worktree-tree identity and patch preimages are verified for these 115 pilot tasks. This verification has not been extended to the other 14 tasks.

Candidate mapping: 339,996/343,191 nodes (99.07%). Source-verified alias bindings: 58,893; unresolved ambiguous aliases: 1,959. Exact-ID body conflicts encountered: 932. Changed-file coverage: 190/248; all 58 unavailable files remain in recall denominators.

Mapping statuses: `{'file_only': 59, 'precise': 56}`. Symbol labels use the smallest available enclosing definition; class stubs do not establish exact function relevance. File-level evaluation is primary.

Pilot repository distribution: `{'fastapi/fastapi': 64, 'psf/requests': 10, 'Textualize/rich': 41}`. Sampling seed: 2027; selection: repository round-robin, one base commit per draw; excludes every development task and every task sharing a development base commit; no gold used.

Shared repository/base-commit groups, which must stay within one future split: `{'Textualize/rich@4d6d631a3d2deddf8405522d4b8c976a6d35726c': ['rich_3882', 'rich_3894'], 'psf/requests@7a13c041dbef42f9f3feb14110f02626f6892e9a': ['requests_6629', 'requests_6589']}`.

Dataset fingerprint: `e39fbcca3ccd76d39582587aa3c3d6e32e1adb5cfa011ed4845d55267c3afc7d`. Graphs have symbol IDs but no explicit source positions; supplied unperturbed edges are not independently established as semantically correct.

Sources: [raw audit](../home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent/paper/reports/competition_audit_full_2026-11-06.json), [pilot selection](../home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent/paper/data/heldout_full_2026-11-07.json), [configuration](../home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent/paper/config/pilot_v3.json), [run manifest](../home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent/paper/results/2026-11-08_extension_v3/manifest.json), [mapping manifest](../home/frostflux/Ahnaf_Shafin/Hackathon/gemma4-agent/paper/results/2026-11-08_extension_mapping/manifest.json).
