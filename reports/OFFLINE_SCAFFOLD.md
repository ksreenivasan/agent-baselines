# hle-tools-v0 offline scaffold report

Date: 2026-08-25

## Scope completed

- Added a pinned local HLE loader with a one-row synthetic multimodal fixture.
- Wired the loader, thin Inspect/Asta ReAct solver, structured JSON submission, and scorer.
- Added only the approved solver tools: provider-neutral `web_search`, controlled `fetch_url`, and AstaBench's stateful `python_session`.
- Added an Exa host-side search adapter plus deterministic fixture search. ActiveLoop Scientific Search is excluded.
- Added a no-network generated-code Compose definition with no privileges, host mounts, Docker socket, host network, or secret environment.
- Added runtime provider-key injection restricted to named files under `~/secrets_and_keys`; key values are not logged.

## Verification

```text
16 passed
```

The check covered the HLE-specific tests plus the repository's basic mock-LLM and solver-scaffold tests. Python compilation and Inspect task construction also succeeded.

Static isolation checks verify `network_mode: none`, read-only root, dropped capabilities, non-root user, no broad host volume, and no Docker socket.

## Concrete blockers before a live HLE smoke

1. The pinned gated HLE Parquet URL and datasets-server endpoint both return HTTP 401 without authorized HLE access. No local HLE data file or approved HF key is available.
2. Neither `~/secrets_and_keys/exa.key` nor `tavily.key` exists. The search interface is therefore fixture-only; no substitute backend was engineered.
3. The first sandbox image build was stopped after its shared Python 3.11 base-image metadata pull timed out during concurrent Docker activity. Per coordination, it was not retried. The static isolation checks pass, but runtime sandbox validation remains pending until the shared base is local.

No live provider inference was performed. These blockers must be cleared before the one-question four-model HLE smoke can be semantically valid.
