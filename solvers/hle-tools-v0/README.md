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
Resume an interrupted checkpoint with `inspect eval-retry <checkpoint.eval>`.
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
properties. Cleanup removes the per-job container; `/var/tmp` may be reclaimed
by the node without affecting evaluation artifacts.

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
