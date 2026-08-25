# hle-tools-v0 real one-question smoke

Date: 2026-08-25

Code revision: `d1f017e` plus final-turn fix `d0e7827`

Dataset: `cais/hle@5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`

Dataset file SHA-256: `6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a`

Smoke manifest SHA-256: `f55879af1744b19b6bdef5cee24ab91c7be1e0764e80467e23700a70c924208a`

## Access and condition

The pinned gated Parquet was downloaded through the authenticated Hugging Face CLI and validated without printing question content: 2,500 unique rows, 342 multimodal rows, expected answer types, and the file hash above. Exa authentication was verified with a non-HLE query; the real smoke used Exa free-tier search.

Final uniform protocol revision: `2026-08-25-no-cumulative-token-limit`.

- one fixed multimodal HLE question outside the pilot manifest;
- high reasoning/thinking for all lanes;
- exactly 15 model turns maximum, with only structured `submit` exposed on turn 15;
- 32,768 output tokens maximum per generation;
- no cumulative token limit;
- message limit 80 and sample time limit 1,800 seconds;
- provider attempt timeout 600 seconds, total request timeout 660 seconds, one API retry;
- Exa search with one two-second retry only on HTTP 429;
- Alibaba-only Qwen routing, OpenRouter fallbacks disabled;
- no monetary stop gate.

The 600-second attempt timeout replaced the initial 180-second value uniformly. Claude's first 180-second attempt produced no tool events, usage, or outcome. GPT's previously valid calls had all completed within 180 seconds, but all four lanes were rerun after the material final-turn and cumulative-token protocol corrections.

## Final smoke outcomes

| Lane | Agent turns | Tool calls | Valid structured submission | Confidence | Judge used | Correct |
|---|---:|---|---|---:|---|---|
| GPT-5.6 Sol | 15 | 23 search, 7 fetch, 4 Python, 1 submit | yes | 0.76 | yes | no |
| Claude Opus 5 | 10 | 8 search, 6 fetch, 1 Python, 1 submit | yes | 0.32 | yes | no |
| Gemini 3.7 Flash | 15 | 2 search, 12 Python, 1 submit | **no** | — | no | no |
| Qwen3.5-397B-A17B | 15 | 7 search, 7 fetch, 1 submit | yes | 0.45 | yes | no |

All four Inspect runs completed with status `success`; none had HTTP retries, Exa 429s, search/fetch transport errors, or sample-limit events. The three valid free-form answers were evaluated by the pinned `openai/o3-mini-2025-01-31` equivalence judge and marked incorrect. Gemini called the reserved submit tool but supplied malformed structured JSON, so the scorer correctly recorded an invalid submission and did not call the judge.

This is one deliberately difficult multimodal item, so 0/4 correctness is not a model ranking or benchmark estimate. The final invalid-submission rate is 1/4; it is a model-format outcome rather than an infrastructure failure and will be measured on the 50-question pilot.

## Usage and spend

| Lane | Solver usage | Wall time |
|---|---|---:|
| GPT-5.6 Sol | 91,026 input, 351,433 cached input, 9,457 output; 7,955 reasoning | 307 s |
| Claude Opus 5 | 20 uncached input, 69,190 cache-write, 295,311 cache-read, 26,895 output | 464 s |
| Gemini 3.7 Flash | 141,106 input, 6,881 output, 10,972 reasoning | 209 s |
| Qwen3.5-397B-A17B | 381,019 input, 4,738 output; 3,773 reasoning | 176 s |

Using the planning-time direct-provider, cache, and pinned Alibaba prices, the solver calls are estimated at roughly **$2.3–$2.4** total, plus a small judge cost. Inspect recorded usage but not provider-billed cost, so this is not an invoice. Exa used the free tier.

## Trace, judge, and isolation review

- The selected sample was multimodal in every lane.
- Search, fetch, stateful Python, final submission, and semantic judging were all exercised across the matrix.
- Raw logs contain no authorized OpenAI, Anthropic, Gemini, OpenRouter, or Exa key value; the scan checked all five values in memory without printing them.
- Generated-code sandboxes retained no network, privileges, host binds, Docker socket, or secrets.
- No HLE sandbox container remained after the runs.
- Raw questions, answers, model messages, tool results, and `.eval` files remain only under ignored `protected/` paths.

## Diagnostic corrections excluded from the final matrix

Before the final uniform rerun, diagnostics exposed and fixed: relative protected-path resolution in commands, web timeouts/policy errors escaping into the harness, unescaped judge JSON braces, lack of a submit-only final turn, and a cumulative 250k token limit that preempted the 15-turn protocol. Those outcomes are preserved outside Git and are not aggregated with the final smoke.

## Gate decision

The final smoke is semantically clean: every lane received the same multimodal sample, tools and sandbox worked, final submission behavior was exercised, scorer/judge outcomes were explicit, provider settings were recorded, rate limits did not interfere, and secret isolation passed. Proceed to the approved stratified 50-question pilot with modest sample concurrency and stop before any expanded/full run.
