# HLE evaluation adapters

The default direct evaluation follows the disclosed Artificial Analysis HLE protocol:

- Dataset: the 2,158 text-only questions in the 2,500-question May 2025 `cais/hle` release.
- Solver prompt: the original HLE answer-type-specific system prompt disclosed by Artificial Analysis.
- Scoring: the complete response is evaluated with the disclosed HLE equality-checker prompt using `openai/gpt-5.6-luna` at medium reasoning effort.
- Attempts: pass@1, with no tools in the `hle_direct` condition.

The `hle_tools_v0` condition uses the same dataset, response format, and scorer, but adds Keenable search, controlled fetch, stateful Python, and a 15-turn agent loop. It is an experimental tools extension and is not directly comparable to Artificial Analysis' no-tools leaderboard.

`dataset_variant="verified"` remains available. It uses `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`, selecting its 668-row `Gold subset`.

- Solver outputs are plain HLE-formatted text. The tools lane passes that text as the submit tool's single string argument; it does not nest a JSON document inside the tool call.
- Structured JSON is used only for the private equality judge's response.
- Raw datasets, manifests, and run logs stay under ignored `protected/` paths.
- `cais/hle` is gated. Accept its terms and authenticate on the machine that downloads it. Do not redistribute benchmark data.
- Generated Python runs in an isolated no-network sandbox. Docker Compose in
  `sandbox/` remains the portable default; M2 Slurm jobs explicitly select the
  `m2-enroot` backend described below.
- Model keys are injected at runtime by `run_with_secrets.py`; key values are never printed.
- Search uses Keenable in `pro` mode when `~/secrets_and_keys/keenable.key` is loaded. Exa remains available as an explicit alternative. Without a live-search credential, only fixture search is enabled and live runs are blocked.

## Required launch smoke test

Launch HLE evaluations through `run_with_secrets.py`. Before it starts `inspect eval`, the wrapper automatically checks:

1. `web_search` uses the configured live backend and returns structured results, and `fetch_url` can retrieve at least one returned page;
2. `python_session` preserves state across calls in the same sandbox declared by the evaluation task;
3. the `submit` tool preserves a plain-text HLE response;
4. a configured OpenAI-compatible endpoint's `/models` catalog contains the exact served ID (native Google models use the equivalent model lookup);
5. the evaluated-model endpoint returns a non-empty response with the requested reasoning settings; and
6. the judge endpoint returns a valid equality judgment for a known canary using GPT-5.6 Luna at medium reasoning effort.

Direct/no-tools HLE skips checks 1–3. A live tools run fails if `HLE_SEARCH_BACKEND` silently resolves to `fixture`. Any failed check emits a warning and exits with status 2 before evaluation samples run.

Example shape:

```bash
export HLE_SEARCH_BACKEND=keenable
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python solvers/hle-tools-v0/run_with_secrets.py \
  --provider openai --provider keenable \
  inspect eval agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0 \
  --model openai/gpt-5.6-sol \
  -T data_path="$HLE_DATA_PATH" \
  -T dataset_revision="$HLE_DATASET_REVISION"
```

To proceed deliberately despite a failed or unavailable smoke dependency, add `--skip-smoke-test` anywhere after `run_with_secrets.py`:

```bash
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python solvers/hle-tools-v0/run_with_secrets.py \
  --skip-smoke-test --provider openai --provider keenable \
  inspect eval agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0 ...
```

The override itself prints a warning. For a non-OpenAI judge endpoint, set `HLE_JUDGE_BASE_URL`; the evaluated model continues to use Inspect's normal `--model-base-url` option. Task and model config files and `-T`/`-M` arguments are reused by the smoke test.

For an arbitrary OpenAI-compatible or vLLM model, provide all endpoint behavior explicitly:

```bash
VLLM_API_KEY="$SERVED_MODEL_API_KEY" PYTHONPATH=. \
python solvers/hle-tools-v0/run_with_secrets.py --provider openai \
  inspect eval agent_baselines/evals/hle_verified_direct/task.py@hle_verified_direct \
  --model vllm/exact-served-model-id \
  --model-base-url https://model-host.example/v1 \
  --reasoning-history all \
  --timeout 660 --attempt-timeout 600 --no-fail-on-error \
  --log-format eval --log-buffer 1
```

`VLLM_API_KEY` must be set to the real key or an explicit dummy token; relying on Inspect's implicit vLLM default is not allowed for these runs. The wrapper refuses endpoint launches without explicit `--reasoning-history`, and the required smoke checks both `/models` and tiny inference before samples begin. Never place a secret value in a command, config, or log; inject the variable from the runtime secret file.

Native `google/gemini-3.8-flash` is supported with `--reasoning-effort high` and no temperature override. Inspect's Google adapter maps high effort to native high thinking; omitting temperature also avoids the legacy `temperature`, `top_p`, and `top_k` fields rejected by Gemini 3.8.

Use one Inspect process per model x condition. Always use `--log-format eval`; the
JSON recorder retains every complete sample in memory and rewrites the complete
JSON document at every flush. The `.eval` recorder moves completed samples into
a ZIP-backed temporary file and clears its full-sample buffer after each flush.
`--log-buffer 1` therefore bounds full-sample memory while preserving each sample
as it completes.

On M2, write the live `.eval` log to per-job storage under `/var/tmp`, not directly
to Weka. Run `m2/checkpoint_eval_logs.py` alongside Inspect to copy only validated
ZIP snapshots to Weka through a temporary file plus atomic rename. A concurrent or
failed copy can never replace the last valid checkpoint. Stop the helper with
`SIGTERM` after Inspect exits; it performs one final checkpoint before stopping.
Recover interrupted work with finite residual-ID manifests and judge-only repair of saved answers. Preserve every accepted scored answer, including incorrect answers.
`--no-fail-on-error` continues after bounded sample errors. Reports must retain
errored samples in the manifest denominator and distinguish an accounted run from
an all-scored run.

## Offline checks

```bash
uv sync --project solvers/hle-tools-v0 --python 3.11
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- pytest -q tests/hle_tools_v0
```

## Pinned standard data

Set `HLE_DATA_PATH` to a local snapshot of `cais/hle` and `HLE_DATASET_REVISION` to the immutable May 20, 2025 commit `021a3d71f516a7ac28ceb8d284969902edf1edeb`. Mutable refs such as `main` are rejected. The loader verifies 2,500 source rows, unique IDs, and exactly 2,158 rows with no image before evaluation.

The authenticated M2 snapshot is pinned to that exact revision; do not silently substitute the current `main` revision.

The task entry points are:

```text
agent_baselines/evals/hle_direct/task.py@hle_direct
agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0
```

Both default to `dataset_variant="standard"`. Pass `dataset_variant="verified"` to either task, or use the compatibility wrapper `agent_baselines/evals/hle_verified_direct/task.py@hle_verified_direct`, for HLE-Verified.

## M2 Enroot sandbox

M2 users cannot access the Docker daemon from Slurm jobs. Source the preparation
helper inside the allocation, then select the M2 backend on the task:

```bash
source solvers/hle-tools-v0/m2/prepare-enroot.sh
trap cleanup_m2_enroot EXIT

PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python solvers/hle-tools-v0/run_with_secrets.py \
  --provider openai --provider keenable \
  inspect eval agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0 \
  --model openai/gpt-5.6-sol \
  -T sandbox_backend=m2-enroot \
  -T data_path="$HLE_DATA_PATH" \
  -T dataset_revision="$HLE_DATASET_REVISION"
```

The helper builds a per-job Enroot container under `/var/tmp`; no container
state is placed on Weka. The custom Inspect backend gives every sample its own
workspace and persistent Python namespace, removes provider/search credentials,
blocks network access, and masks `/home` and `/mnt/weka`. Run
`python solvers/hle-tools-v0/m2/selftest.py` after preparation to verify those
properties. To reuse a tested prebuilt sandbox that already includes its Python
dependencies, export `M2_ENROOT_IMAGE` with its `.sqsh` path and
`M2_ENROOT_IMAGE_SHA256` with its recorded SHA256 before sourcing the helper.
The helper verifies both the source and node-local image before container
creation. Use a fresh per-job container for production. Cleanup removes the
per-job container; `/var/tmp` may be reclaimed by the node without affecting
evaluation artifacts.

## HLE-Verified data and manifests

Set `HLE_VERIFIED_DATA_PATH` to the protected snapshot directory containing all 15 Parquet parts. Deterministic protected manifests are generated with:

```bash
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --python 3.11 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli build-manifests \
  --data-path "$HLE_VERIFIED_DATA_PATH" \
  --output-dir protected/manifests-hle-verified-gold
```

Only the sanitized balance table and manifest hashes belong in Git.

## Host-awake requirement

Inspect 0.3.203 enforces attempt and sample time limits with AnyIO cancellation scopes. Those scopes cannot advance while the macOS host is asleep. Wrap live evaluation in `caffeinate -i` so wall-clock observations correspond to runnable event-loop time:

```bash
caffeinate -i uv run --project solvers/hle-tools-v0 --frozen -- \
  python solvers/hle-tools-v0/run_with_secrets.py ...
```

Provider SDK timeouts remain defense in depth; `caffeinate` does not replace them. The preserved aborted pilot was interrupted by host idle sleep, not evidence of a 42-minute OpenAI request.

## Static no-inference preflight

```bash
HLE_DATA_PATH=protected/hle-may-2025 \
HLE_DATASET_REVISION=<immutable-40-character-commit> \
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli preflight
```

This static command checks local files and configuration only. It does not replace the automatic live smoke test above.

See `protocols/hle-direct-v0.yaml`, `protocols/hle-tools-v0.yaml`, and `protocols/hle-verified-direct-v0.yaml` for machine-readable protocol records. Older reports describe the previous HLE-Verified/strict-JSON protocol and should not be treated as the current default.

## Bounded M2 continuation

Use a frozen Git worktree with `m2/run_lane.sbatch`. Set `HLE_REPO`,
`HLE_PYTHON`, `HLE_CONFIG`, `HLE_MANIFEST_DIR`, and `HLE_OUTPUT`.
The manifest directory contains ordered JSON files with an `ids` array.
Use 100–200 IDs per shard initially. Each launch uses a new attempt directory;
a lane lock prevents overlapping drivers.

The configuration declares `task` (`hle_tools`, `hle_direct`, or
`hle_tools_canary`), `run_phase` (`production` or `diagnostic`), `model`, `reasoning_effort`, `concurrency`,
`max_retries`, `timeout`, `attempt_timeout`, `dataset_path`,
`dataset_revision`, and `judge_model`. Live tools require
`search_backend: keenable`. For local serving also set `model_base_url`,
`reasoning_history`, and `vllm_key_file` (a path, never a key value).
Optional `temperature`, `top_p`, and `max_tokens` are sent only when
non-null. Historical full-matrix tools requests had no explicit output cap;
do not infer one from an older protocol document.

For K2 tools and canary runs, set
`model: k2-vllm/IFM/K2-Horizon-375B-A23B`. The shard runner selects
`python -m agent_baselines.evals.hle_tools_v0.k2_vllm_cli eval ...`, which
registers the adapter before Inspect resolves the model; use this bootstrap
for manual tools/canary launches through `run_with_secrets.py` as well.
The checkpoint's template requires a thinking key even on empty tool-call
turns. The adapter adds `reasoning_content: ""` only when an assistant replay
turn has no thinking field. It preserves existing nonempty reasoning, tool
calls, and history, and leaves the shared endpoint unchanged. Offline replay and bootstrap tests pass. A live canary also replayed an
empty-reasoning tool turn and then submitted successfully. Direct controls can keep
`vllm/IFM/K2-Horizon-375B-A23B`: alias registration during preflight currently
covers tools and canary tasks.

The driver prepares and checks the Enroot sandbox, then supervises each shard.
Inspect writes native `.eval` archives to node-local scratch, flushing each
completed sample. The publisher checks every 60 seconds and atomically copies
valid archives to Weka. A missing/dead publisher, a 300-second publication
failure, or a fatal tool-health signal stops the worker. Final publication has
a bounded timeout. Local diagnostics and a failed result remain available if
Weka cannot accept writes.

Exit 0 requires a successful, non-invalidated final archive, exact expected
ID/epoch membership, scores, and no unresolved errors or infrastructure
invalidation. Correct and incorrect scored answers are both retained.
Exit 3 means a durable, structurally complete shard has isolated known
provider or judge errors alongside usable scores: the lane continues and
records residual IDs without claiming completion. Other failures stop the
lane with exit 2. Diagnose residuals before selecting a finite retry manifest;
never rerun valid incorrect answers to improve accuracy.

Run `m2/canary.py@hle_tools_canary` before real shards. It checks an actual
model-generated search → successful fetch → two separate persistent-Python
turns → plain-text submission trajectory. The supervisor requires both the
trajectory check and equality judge to pass. Follow with a frozen diagnostic
pilot at concurrency one and three.

For historical recovery, use
`python -m agent_baselines.evals.hle_tools_v0.recovery build --help`.
The streaming builder verifies the frozen dataset and declared source slice,
records source hashes and attempt provenance, and partitions IDs into retained
scores, judge-only repair, and new generation. Retained native shards are
reread to verify their original generation and scores. Keep protected ledgers,
sample IDs, and raw traces outside Git. Recovery inputs must be explicitly
chosen original artifacts; an old completion marker is not evidence.

`m2/soak_logging.py` exercises the installed Inspect recorder with representative
long traces and injected publication failure. Unit tests cover publication,
tool health, judge retries, strict canaries, recovery, and supervisor failure
paths. Pin the runtime dependency lock and record the source commit and
configuration hashes with every launch.


## Final aggregation

`m2/aggregate_campaign.py` validates one completed model condition while reading
one sample at a time. It combines approved retained shards, a completed judge
repair, and explicitly supplied production attempt results. It verifies source,
archive, configuration, and protocol hashes; dataset and model settings; recorded
solver requests; and approved clean production commits. Historical source
revisions, including recorded dirty state, remain explicit in the provenance.
Diagnostic configurations are rejected.

```bash
PYTHONPATH=. python solvers/hle-tools-v0/m2/aggregate_campaign.py \
  --config "$campaign/configs/gpt-sol-full.json" \
  --expected-manifest "$campaign/manifests/all.json" \
  --protocol "$campaign/PROTOCOL.json" \
  --source-commit <approved-production-commit> \
  --ledger "$campaign/ledger-v2/gpt-sol/ledger.json" \
  --judge-result "$campaign/judge-repair/gpt-sol/attempt-1/result.json" \
  --judge-adoption "$campaign/judge-repair/gpt-sol/adoption-ledger-v2.json" \
  --attempt <production-shard-attempt/result.json> \
  --output "$campaign/final/gpt-sol"
```

Repeat `--attempt` and `--source-commit` as needed. Use the final audited ledger.
When a later ledger changes other rows but preserves the exact judge-only IDs
and saved generations, reuse the existing paid repair with an adoption record:

```bash
PYTHONPATH=. python solvers/hle-tools-v0/m2/aggregate_campaign.py adopt-judge \
  --ledger "$campaign/ledger-v2/gpt-sol/ledger.json" \
  --judge-result "$campaign/judge-repair/gpt-sol/attempt-1/result.json" \
  --output "$campaign/judge-repair/gpt-sol/adoption-ledger-v2.json"
```

This verifies both ledger/source hashes and the unchanged repaired generations;
it does not call a model or modify the original repair result. Supply the record
with `--judge-adoption` during aggregation. A repair already bound to the active
ledger needs no adoption record. Omit ledger/repair arguments for fresh GLM and
K2 conditions. For an interrupted archive, `--partial-selection <file.json>`
requires an explicit JSON object mapping absolute attempt `result.json` paths to
selected successful IDs. Started/cancelled archives contribute only those rows;
the index records their original status. Successful partial shards may contribute
their scored rows while known provider/judge failures must be resolved in another
supplied attempt. Overlapping scored outcomes are rejected without selecting a
better answer.

Completion requires exactly one accepted correct or incorrect outcome for every
expected ID. The new output directory contains `summary.json`, ordered
`outcomes.jsonl` with row correctness and provenance, and `shards.json` indexing
native archives and their selected IDs. Token totals, tool counts, and available
sample elapsed times are recorded without merging long traces into another JSON
log. Failed validation publishes no completed aggregate.


Judge-provider failures are eligible for judge-only repair only when the native
failed model request belongs to the HLE scorer span and its traceback points to
the canonical HLE judge invocation. Empty saved solver responses are preserved.
A nonempty response alone does not establish that generation completed. Missing
or ambiguous stage evidence keeps the existing disposition and requires review
before any residual generation; do not turn every `generate` disposition into
a retry manifest. Search invalidations and errors combined with scores remain
rejected. A separately verified recovery revision can classify and repair frozen
production archives without changing their generation, source, or configuration
checks. The lane runner does not automatically regenerate these residuals.

Keenable content failures are recognized by the exact HTTP status and JSON body:
403 with error `Upstream forbidden` and message
`The target server denied access to this URL`;
404 with error `Not found` and message `The requested URL could not be found`;
422 with error `Unprocessable entity` and message
`The page was reached but content could not be extracted`. These return ordinary
`content_access_denied`, `content_not_found`, or `content_not_extractable` tool
errors with the status, backend, and provider message. They reset the backend outage counter and do not
invalidate the sample. Other response bodies, statuses, and backends retain the
infrastructure guard behavior; status alone never establishes a content failure.
Unknown HTTP failures also record the status, response-body SHA, bounded parsed
error/message text, and a safe request ID in guard/sample metadata. The injected
backend key is redacted before text truncation; raw bodies and headers are not
stored. Existing historical status-only failures remain ambiguous and are not
reclassified by this change. The 403 exception applies only to the exact target
server denial body: platform authentication failures, unknown or malformed
responses, additional JSON fields, and other backends remain guarded. Existing
sample invalidations and fatal sentinels are not cleared by this classification.


## Residual selection

`m2/residual_selection.py` prepares retry manifests without launching work. Its
expected manifest is the initial new-generation ID set, excluding retained
historical answers. Supply every current production attempt directory, including
attempts with only `launch.json`, and revalidate immediately before dispatch.

```bash
PYTHONPATH=. python solvers/hle-tools-v0/m2/residual_selection.py \
  --config "$campaign/configs/gpt-sol-full.json" \
  --expected-manifest "$campaign/ledger-v2/gpt-sol/ids-generate.json" \
  --protocol "$campaign/PROTOCOL.json" \
  --source-commit <approved-production-commit> \
  --attempt <production-shard-attempt-directory> \
  --partial-selection <clean-IDs-by-attempt-directory.json> \
  --infrastructure-selection <diagnoses-by-ID.json> \
  --output "$campaign/residual-selection-1"
```

Repeat `--attempt` and `--source-commit` as needed; both selection inputs are
optional. Partial selections map absolute attempt directories to clean IDs from
failed shards. Each infrastructure selection maps an ID to an object containing
its latest absolute `attempt` directory, `closed: true`, and concrete
`evidence`. Closure must follow an actual stopped/finished job check; elapsed
time is insufficient. Every regeneration requires this explicit diagnosis.
If durable native archives exist without a finalized result/archive binding,
the attempt must be reviewed or finalized before regeneration can be selected.
Valid correct and incorrect scores and saved judge-only answers cannot be
overridden. Missing results remain held by default and still consume an attempt.

The immutable output contains the per-ID attempt ledger, a saved-answer judge
queue, aggregate-compatible retained selections, and retry manifests of at most
200 IDs. Each retry manifest binds its prior launch and selection ledger hashes.
Initial manifests must be disjoint, retries require their parent attempt records,
and more than one additional generation is rejected. The command neither
discovers omitted jobs nor dispatches work; the supervisor must supply the full,
current attempt inventory. Held and exhausted outcomes stay explicit.

## Direct-control finalization

`m2/aggregate_direct_controls.py` is a separate finalizer for the direct controls.
It does not relax the tools aggregator. The bound `DIRECT_PROTOCOL.json` defines
the exact dataset, replacement manifests, ownership partition, source files,
configuration, and endpoint evidence. Migration inputs must supply every
migration-owned ID; copied historical rows in the same archives are excluded.
Both partitions must be complete before any final output is published.

```bash
PYTHONPATH=. python solvers/hle-tools-v0/m2/aggregate_direct_controls.py \
  --protocol "$campaign/controls/DIRECT_PROTOCOL.json" \
  --lane glm-flash \
  --migration-source <closed-migration-source.json> \
  --replacement-attempt <replacement-attempt/result.json> \
  --output "$campaign/controls/glm-flash/final"
```

Repeat either source option as needed. Each migration source is a JSON object
with `protocol_sha256`, `lane`, `migration_job`, `closed: true`,
`closure_evidence: {"path": "...", "sha256": "..."}`, and
`archive: {"path": "...", "sha256": "..."}`. Closure evidence must record an
actual stopped/finished job check; elapsed time, archive counts, and an
`afterany` dependency do not establish closure or success. Optional
`selected_ids` permits explicit clean rows from a closed partial archive.
The archive evaluation identity must match the frozen lineage. A started
snapshot is never accepted from counts alone.

Optional `--selections` names a JSON mapping from absolute replacement
`result.json` paths to explicit clean IDs from failed shards. Every replacement
must still bind its original approved manifest, launch, config, source revision,
and single durable archive. Only the initial protocol-bound manifests are
admitted; residual manifests need a separately bound protocol revision and are
never admitted automatically. All accepted native rows require exact C/I scoring,
no error or invalidation, no tool use, and matching recorded solver settings.
Valid empty-answer incorrect scores are retained. Already valid migration
generations and scores must remain unchanged. Missing or failed judge results
stay unresolved; this entrypoint does not adopt judge repairs without a separate
preserved-generation provenance design and does not regenerate answers.

The migration's inherited header has `max_connections=3` despite task and actual
request concurrency 12. The finalizer checks this explicit legacy header profile
and validates actual requests against the migration concurrency. Replacement
header and request concurrency come from the bound configuration (currently 12).
The published summary keeps both provenance strata and qualifies comparison
against the tools condition at concurrency 3. It does not claim identical load,
an unbiased historically selected subset, or a first-attempt estimate.
