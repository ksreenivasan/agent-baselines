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

After the shared base image became available locally, the sandbox was built with pulls disabled and exercised under the explicit Compose project `hle-tools-v0-offline`. Runtime inspection confirmed no network, privileges, capabilities, bind mounts, Docker socket, or secret environment. Container Python executed successfully, an outbound socket was blocked, and scoped cleanup left no project containers.

A full offline Inspect fixture then exercised all three approved tools in sequence—fixture search, controlled fetch, and stateful Python—followed by structured submission and scoring. The synthetic answer scored 1/1. Inspect removed its sample container; no HLE container remained. The locally built sandbox image was `sha256:e3c88d43cc9de4bfb2f73255f49c87165028b7e0882230ca62c753f2fc6c2d41`.

## Concrete blockers before a real HLE smoke

1. The pinned gated HLE Parquet URL and datasets-server endpoint both return HTTP 401 without authorized HLE access. No local HLE data file or approved HF key is available.
2. Neither `~/secrets_and_keys/exa.key` nor `tavily.key` exists. The search interface is therefore fixture-only; no substitute backend was engineered.

No live provider inference was performed during this milestone. The two remaining access blockers must be cleared before a real one-question HLE smoke can be semantically valid; the authorized next step is only a clearly labeled synthetic fixture smoke.
