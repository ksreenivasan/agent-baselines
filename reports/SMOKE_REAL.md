# hle-tools-v0 real one-question smoke

> Historical CAIS-HLE evidence only. The active protocol now uses pinned HLE-Verified Gold data; this matrix is not release evidence for the migrated dataset. See `HLE_VERIFIED_MIGRATION.md`.

Date: 2026-08-25

Authoritative code revision: `a080cea`

Protocol revision: `2026-08-25-submit-wrapper-alignment`

Dataset: `cais/hle@5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`

Dataset file SHA-256: `6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a`

Smoke manifest SHA-256: `f55879af1744b19b6bdef5cee24ab91c7be1e0764e80467e23700a70c924208a`

## Access and final uniform condition

The pinned gated Parquet was downloaded through the authenticated Hugging Face CLI and validated without printing question content: 2,500 unique rows, 342 multimodal rows, expected answer types, and the file hash above. Exa authentication was verified with a non-HLE query; the real smoke used Exa free-tier search.

The authoritative matrix below used:

- one fixed multimodal HLE question outside the pilot manifest;
- high reasoning/thinking for all lanes;
- exactly 15 model turns maximum, with only `submit` exposed on turn 15;
- inherited submit-tool schema: one top-level `submission` string containing the answer/confidence/explanation JSON object;
- 32,768 output tokens maximum per generation;
- no cumulative token limit;
- message limit 80 and sample time limit 1,800 seconds;
- uniform provider attempt timeout 600 seconds, total request timeout 660 seconds, one API retry;
- Exa search with one two-second retry only on HTTP 429;
- Alibaba-only Qwen routing, OpenRouter fallbacks disabled;
- no monetary stop gate.

The 600-second attempt timeout is a uniform infrastructure ceiling, not a Claude-only advantage. Earlier valid GPT calls completed within 180 seconds, but all four lanes were rerun after the material final-turn, cumulative-token, and submit-wrapper corrections.

## Authoritative final smoke matrix

| Lane | Agent turns | Tool calls | Valid structured submission | Confidence | Judge used | Correct |
|---|---:|---|---|---:|---|---|
| GPT-5.6 Sol | 15 | 15 search, 5 fetch, 3 Python, 1 submit | **yes** | 0.46 | yes | no |
| Claude Opus 5 | 8 | 5 search, 4 fetch, 2 Python, 1 submit | **yes** | 0.40 | yes | no |
| Gemini 3.7 Flash | 15 | 3 search, 2 fetch, 9 Python, 1 submit | **yes** | 0.85 | yes | no |
| Qwen3.5-397B-A17B | 10 | 6 search, 3 fetch, 1 submit | **yes** | 0.65 | yes | no |

All four Inspect runs completed with status `success`; all four produced valid structured submissions; all four were evaluated by the pinned `openai/o3-mini-2025-01-31` equivalence judge. None had HTTP retries, Exa 429s, search/fetch transport errors, sample-limit events, or unresolved judge failures.

This is one deliberately difficult multimodal item, so 0/4 correctness is not a model ranking or benchmark estimate. The authoritative invalid-submission rate is **0/4**.

## Usage and spend

| Lane | Solver usage | Wall time |
|---|---|---:|
| GPT-5.6 Sol | 71,317 input, 294,927 cached input, 12,007 output; 10,947 reasoning | 365 s |
| Claude Opus 5 | 16 uncached input, 46,806 cache-write, 202,485 cache-read, 22,564 output | 392 s |
| Gemini 3.7 Flash | 122,803 input, 1,486 output, 6,820 reasoning | 168 s |
| Qwen3.5-397B-A17B | 109,123 input, 9,142 output; 8,082 reasoning | 209 s |

Using the planning-time direct-provider, cache, and pinned Alibaba prices, the solver calls are estimated at roughly **$1.8–$1.9** total, plus a small judge cost. Inspect recorded usage but not provider-billed cost, so this is not an invoice. Exa used the free tier.

## Trace, judge, and isolation review

- The selected sample was multimodal in every lane.
- Search, fetch, stateful Python, final submission, and semantic judging were exercised across the matrix.
- Raw logs contain no authorized OpenAI, Anthropic, Gemini, OpenRouter, or Exa key value; the scan checked all five values in memory without printing them.
- Generated-code sandboxes retained no network, privileges, host binds, Docker socket, or secrets.
- No HLE sandbox container remained after the runs.
- Raw questions, answers, model messages, tool results, and `.eval` files remain only under ignored `protected/` paths.

## Non-authoritative diagnostics

The following are preserved under ignored protected paths and excluded from the final matrix and all reporting aggregates:

- Early runs that exposed relative protected-path handling, web-tool exception propagation, judge-prompt brace formatting, missing final-turn reservation, and a cumulative 250k token limit that preempted the intended 15-turn condition.
- Gemini's earlier invalid submission: it called the inherited `submit` tool with top-level `answer`, `confidence`, and `explanation` fields, while the actual tool accepts one `submission` string. The prompt was aligned to the inherited schema and a focused mock/tool-call test was added. Under the authoritative revision, Gemini produced a valid submission and was judged normally.
- An accidentally launched pilot command that retained `--max-samples 1`. It was stopped in under 20 seconds, its four exact Compose projects were cleaned, no results were aggregated, and its ignored partial artifacts are diagnostics only.

## Gate result

The authoritative four-lane smoke passed semantic and infrastructure review: 4/4 valid submissions, uniform provider/scaffold settings, explicit judge outcomes, no rate-limit interference, and clean secret/sandbox checks. Pilot launch remains a separate held action pending review of the exact 50-sample command plan.
