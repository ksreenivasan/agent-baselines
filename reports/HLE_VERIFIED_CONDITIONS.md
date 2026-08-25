# HLE-Verified direct and with-tools conditions

Date: 2026-08-25

## Public fork and lineage

GitHub metadata was checked before branch creation:

- Repository: `ksreenivasan/agent-baselines`
- Visibility: **public** (`private: false`)
- GitHub relationship: actual fork (`fork: true`)
- Parent and source: `allenai/agent-baselines`
- Local `origin`: `git@github.com:ksreenivasan/agent-baselines.git`
- Local `upstream`: `https://github.com/allenai/agent-baselines.git`
- Active authenticated GitHub account and repository owner: `ksreenivasan`

The AllenAI implementation base is `0e32055da8f094e9e71d047d23183064b6ebc9d0`. The existing `feat/hle-tools-v0` branch and its history were preserved at `4ae4e95`. New work was branched from that exact commit as `feat/hle-verified-evals`; no branch was deleted, renamed, rebased, or rewritten.

The with-tools condition starts from AllenAI's general ReAct agent and uses AstaBench/Inspect submission, tool execution, stateful Python, provider adapters, scoring interfaces, logs, and sandbox lifecycle. HLE-specific loading, selection, prompts, search/fetch policy, sandbox configuration, submit-only final turn, and scorer wiring are fork-local choices. ActiveLoop's `hle_with_tools@7a348ce23658ecbc6dd5206382c92c4016d2471a` was a comparison source only; no ActiveLoop loop, scientific-search service, or in-process interpreter was imported.

## Shared pinned evaluation data

Both conditions use the fork-local loader and scorer over:

- `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`
- Gold subset: 668 of 2,500 rows
- Structured `json.question`, `json.answer`, `json.answer_type`, and `json.image`
- The same protected deterministic smoke and 50-row pilot manifests
- The same `answer`, `confidence`, `explanation` schema
- Deterministic multiple-choice handling and normalized/semantic exact-answer judging

The protected snapshot, IDs, prompts, answers, images, model responses, judge responses, and logs are not in Git. HLE-Verified is public and ungated, but its dataset repository declares no license; this fork does not redistribute it.

## Separate scaffold conditions

| Property | Direct/no-tools | With-tools |
|---|---|---|
| Protocol | `hle-verified-direct-v0` | `hle-tools-v0` |
| Task | `hle_verified_direct` | `hle_tools_v0` |
| Evaluated-model generations | Exactly 1 | At most 15 |
| Model-visible tools | None | Exa search, controlled fetch, sandboxed stateful Python; inherited submit on final turn |
| Agent loop | None | AllenAI-derived ReAct loop |
| Sandbox | None | Disposable no-network Docker sandbox for Python |
| Submission | Direct JSON response | Same JSON encoded inside inherited `submission` string |
| Scorer | Shared | Shared |

A semantic judge call made by the scorer is evaluator work, not another evaluated-model attempt. Direct and with-tools scores are **separate scaffold conditions** and must never be merged or presented as interchangeable HLE results.

## Direct implementation and offline validation

The direct condition adds only:

- one system instruction describing the existing JSON schema;
- Inspect's single `generate()` solver step;
- a task that reuses the pinned loader and scorer with no sandbox or tools; and
- one focused mock test asserting exactly one multimodal generation and `tools == []`.

The existing with-tools task and solver source were unchanged on the new branch. Offline results: 18 HLE tests passed, including schema, deterministic selection, image serialization, scorer behavior, final-turn submission, isolation, web policy, and the direct one-generation invariant.

## Real GPT direct smoke

One authorized smoke used the protected multimodal Gold smoke row, `openai/gpt-5.6-sol`, high reasoning, and `caffeinate -i`:

- Code: `bcd52e0`
- Protocol revision: `2026-08-25-direct-one-generation`
- Inspect status: success
- Solver-model generations: **1**
- Model-visible tool events: **0**
- Sandbox: none
- Structured response: valid
- Confidence: 0.38
- Judge: invoked successfully
- Correctness: incorrect on this one item
- Solver usage: 13,449 tokens (960 input, 12,489 output, including 12,430 reasoning)
- Wall time: 4:29; host-awake assertion lifetime: 4:35
- HTTP retries: none
- Secret scan: clean
- Residual process/container cleanup: clean

This is plumbing evidence, not an accuracy estimate. No other model and no 50-question pilot was run.

## Public hygiene review

Before publication, the branch was checked for authorized key values and all 51 protected smoke/pilot IDs. Git tracks no `protected/` paths, Parquet files, `.eval` logs, run directories, or raw dataset artifacts. The new test uses only the existing synthetic fixture. No raw HLE question, answer, image, prompt, response, or judge record was added.

## Remaining caveats

- HLE-Verified declares no license.
- Three Gold records carry invalid-problem metadata despite the card's fully validated description; 24 Gold rationales are also flagged invalid. The answer field is marked valid for all 668 Gold rows.
- The one-row smoke cannot estimate accuracy or compare the two scaffolds.
- The direct Inspect log retains the shared loader's internal dataset name `hle-tools-v0`; the task name and protocol metadata unambiguously identify the direct condition.
- Any 50-row pilot or additional provider lane requires separate authorization.
