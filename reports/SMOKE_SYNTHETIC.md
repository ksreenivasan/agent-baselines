# hle-tools-v0 synthetic fixture smoke

Date: 2026-08-25

Code revision: `75d8c04`

Scope: one synthetic multimodal fixture, four requested model lanes

> **Not an HLE result.** The gated HLE dataset was unavailable, web search used the deterministic fixture backend, and the single synthetic answer has no benchmark meaning.

## Result

| Lane | Exact Inspect model ID | Effective reasoning config | Result | Confidence | Tool events | Tokens | Wall time |
|---|---|---|---:|---:|---|---:|---:|
| GPT-5.6 Sol | `openai/gpt-5.6-sol` | `reasoning_effort=high` | correct | 1.00 | `submit` | 411 in / 55 out, 14 reasoning | 6 s |
| Claude Opus 5 | `anthropic/claude-opus-5` | `reasoning_effort=high` | correct | 0.97 | `submit` | 4 uncached in / 1,144 cache-write / 1,108 cache-read / 98 out | 6 s |
| Gemini 3.7 Flash | `google/gemini-3.7-flash` | `reasoning_effort=high` | correct | 1.00 | `python_session`, fixture `web_search`, `submit` | 5,662 in / 106 out, 545 reasoning | 19 s |
| Qwen3.5-397B-A17B | `openrouter/qwen/qwen3.5-397b-a17b` | `reasoning_effort=high`, reasoning enabled | correct | 1.00 | `submit` | 856 in / 210 out, 122 reasoning | 8 s |

All lanes used `max_tokens=32768`, `max_retries=1`, `attempt_timeout=180`, `timeout=240`, `max_connections=1`, fixture search, the same Inspect solver/scorer, and no temperature override because these reasoning configurations do not accept it consistently.

Qwen requested an Alibaba-only OpenRouter route with `allow_fallbacks=false`, `require_parameters=true`, and `data_collection=deny`. Inspect recorded those request settings but did not surface independent endpoint-selection metadata. GLM/Nemotron fallbacks were not used.

The scorer used deterministic normalized exact match, so no judge-model call was needed. Raw `.eval` files, model messages, and tool traces remain under ignored `protected/smoke-synthetic/`; they are not committed.

## Spend accounting

Inspect recorded usage but not billed cost. Applying the planning-time direct-provider and pinned Alibaba list prices gives a conservative estimated total below **$0.03** for the four successful samples. Fixture search had no external cost. This is an estimate, not a provider invoice.

## GPT diagnosis and corrections

Three non-scoring GPT attempts were preserved outside Git before the final success:

1. An unbounded first request was manually stopped after no model response. Its trace contained zero tool turns and an idle sandbox, identifying provider-call latency rather than a ReAct loop or sandbox deadlock.
2. The first bounded attempt exposed `openai==3.3.1` incompatibility with the pinned Inspect stack (`httpx2.Timeout` passed into AnyIO). The solver environment now pins the repository-compatible `openai==2.30.0`.
3. The next attempt received an OpenAI 400 because the original synthetic PNG was invalid. The fixture is now a verified 16×16 blue PNG, and the loader validates base64 image data with Pillow.
4. The single final bounded retry completed successfully in 6 seconds with no HTTP retry.

## Isolation and cleanup

The generated-code sandbox remained no-network, unprivileged, read-only, non-root, without host binds, Docker socket, or secrets. Gemini exercised the stateful Python tool successfully. Inspect cleaned every completed sample sandbox, the manually interrupted sandbox was removed by exact Compose project name, and no HLE smoke container remained after the matrix.

## Gate decision

The synthetic plumbing smoke is clean. The 50-question pilot is **not run** because both semantic prerequisites remain blocked:

- no authorized local copy/token for gated `cais/hle@5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`;
- no authorized Exa or Tavily key under `~/secrets_and_keys`.

No full benchmark run or push was performed.
