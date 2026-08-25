# `hle-tools-v0`: proposal and smoke-test-first implementation plan

**Status:** this original CAIS-HLE proposal is retained as design history. The active dataset and runtime amendment is `protocols/hle-tools-v0.yaml`: `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`, Gold subset, host idle sleep prevented. Only offline checks and at most one GPT smoke are currently authorized; the 50-question pilot, other model lanes, expanded evaluation, full evaluation, and push are not authorized.

**Planning snapshot:** 2026-08-25. Dataset-specific CAIS/HLE commands and hashes below are historical and must not be used for a new run. Availability and pricing are mutable and must be rechecked without inference immediately before a run.

### User execution overrides (2026-08-25)

- Use **high** reasoning/thinking for GPT-5.6 Sol, Claude Opus 5, Gemini 3.7 Flash, and Qwen3.5; record requested and effective provider settings exactly.
- Record token usage and spend, but do not impose monetary budget stop gates. Token, turn, time, retry, and sandbox limits remain protocol controls.
- Prefer the smallest correct implementation over defensive frameworks. Local hackiness is acceptable when it does not compromise benchmark semantics, traceability, provider neutrality, or host isolation.
- Reuse InspectAI and `agent-baselines`; do not build a new orchestration platform. Borrow only necessary, documented ideas from ActiveLoop and exclude ActiveLoop Scientific Search.
- Prefer an already available provider-neutral Exa adapter, otherwise Tavily if credentials exist. If neither is usable without new credentials, implement and validate the search interface with fixtures and report the blocker rather than engineering a substitute.
- Use local Git commits at meaningful milestones: approved plan, offline scaffold, smoke report, and pilot report. Use the existing effective author/committer identities returned by `git var`; never set or synthesize identity config and never add attribution trailers. Keep all work on `feat/hle-tools-v0`; do not push.
- Keep raw, large, or protected run artifacts outside Git. Commit only code, secret-free manifests/configuration, small synthetic fixtures, and sanitized reports.
- Inject secrets only at runtime from the explicitly authorized provider key files under `~/secrets_and_keys`; never print or commit their values. Do not inspect or depend on Pi/Codex session credentials, browser auth, `~/.pi/agent/auth.json`, or other harness-internal authentication. Generated-code sandboxes receive no secrets.
- Protect the host without building a platform: no privileged containers, Docker socket, host networking, broad writable host mounts, or unnecessary network; use explicit project/container names and scoped cleanup.
- If Qwen is unavailable, use GLM-5.1 and then Nemotron 3 Ultra as distinctly labeled fallbacks; never silently mix model identities.

## 1. Recommendation

Build `hle-tools-v0` as an explicitly internal, versioned **HLE tool-use condition**, not as a canonical or CAIS-endorsed benchmark. After approval, officially fork [`allenai/agent-baselines`](https://github.com/allenai/agent-baselines), keep `origin` pointed at the user fork and `upstream` pointed at AllenAI, and add:

1. a gated, revision-pinned HLE loader with multimodal support;
2. one common ReAct/InspectAI scaffold for every model;
3. common, versioned `web_search_v0`, `fetch_url_v0`, and `python_v0` contracts;
4. a strongly isolated, stateful code sandbox;
5. direct OpenAI, Anthropic, and Google adapters plus an OpenRouter adapter;
6. a structured answer/confidence submission contract;
7. the pinned CAIS judge path plus better small-pilot calibration diagnostics; and
8. complete run manifests, usage/cost accounting, and Inspect traces.

Execution should stop at each gate:

- **offline plumbing and fixture checks**;
- **one-question smoke matrix** (the same one question on all approved models);
- **stratified 50-question pilot**;
- review failures, judge behavior, cost, and traces;
- only then, if separately approved, a **250–500 question expanded pilot** (10–20% of 2,500);
- do not make a 2,500-question run the default next command.

The 50-question pilot is only 2% of HLE. The user has approved it as the bounded pilot for this execution; the earlier 10–20% expanded pilot and the full benchmark remain outside the current authorization.

## 2. Canonical-source assessment

There is **no canonical HLE-with-tools implementation**. HLE is canonical; “with tools” is a scaffolded condition whose search backend, sandbox, budgets, model provider, answer extraction, and judge materially affect results.

| Source | Inspected revision | License/access | What is useful | What is not canonical or needs replacement |
|---|---|---|---|---|
| [`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) | `73ae974b1844c3ffa64c3f4343d9f1f259575700` | MIT code; gated dataset separately | Official question format, direct generation prompt, judge prompt, answer/confidence extraction, accuracy/calibration intent | No tools protocol, tool agent, search policy, sandbox, action budget, or tool trace contract. Repository HEAD also tracks HLE-Rolling changes; that must not silently change the fixed 2,500-question evaluation. |
| [`cais/hle`](https://huggingface.co/datasets/cais/hle) | **`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`** | Auto-gated; card tagged MIT; access conditions and benchmark-integrity notices still apply | The proposed fixed dataset pin: test split, exactly **2,500** records, text/image, MC and exact answer | Raw data, gold answers, rationales, or traces containing questions must not be committed or redistributed. Access approval and external-provider use must be cleared first. |
| [`activeloopai/hle_with_tools`](https://github.com/activeloopai/hle_with_tools) | `7a348ce23658ecbc6dd5206382c92c4016d2471a` on `bigmain` | MIT code; independent fork | Clearly document as design inspiration: multimodal loading, a reason/tool/result loop, 15-iteration convention, stateful Python idea, web plus scientific retrieval, final answer/confidence format, token counters, incremental output, and traces | Not CAIS-endorsed; dataset revision is not pinned; search is mutable DuckDuckGo plus direct page fetch; scientific search is proprietary; model routing is OpenRouter-specific; Python executes in-process rather than in a strong sandbox; its loop can add a 16th forced model call; its public evaluator omits the official calibration output. Do not copy code silently or call it canonical. |
| [`allenai/agent-baselines`](https://github.com/allenai/agent-baselines) | `0e32055da8f094e9e71d047d23183064b6ebc9d0` | Apache-2.0 plus `NOTICE.txt` | Recommended implementation chassis: InspectAI solvers, basic ReAct loop, per-solver `uv` subprojects and lockfiles, model/tool usage logging, Docker development path, tests, and upstream-compatible organization | No HLE task or scorer. Basic ReAct defaults to 100 steps, which is inappropriate here. Root and ReAct subproject currently pin different AstaBench versions, so the new subproject must freeze one tested dependency set. E2E Asta Panda/CodeScientist entries are cached adapters, not reusable live agents. |
| [`allenai/asta-bench`](https://github.com/allenai/asta-bench) | `8fbdbbb68a73fe4a47af4ebcf1819b90b608bd36` | Apache-2.0 plus `NOTICE.txt` | Inspect task integration, structured tools, tool merging, stateful Jupyter execution, per-sample sandboxing, submission utilities, trace/cost conventions, and solve/score separation | Its native-provider search would give different search systems to different models, so it should be disabled in the controlled lane. Its stock sandbox uses bridge networking and a broad scientific environment; `hle-tools-v0` needs a smaller no-network sandbox with stricter limits. HLE should remain a separate adapter rather than being represented as an AstaBench task. |

### Attribution rule

Prefer a clean implementation against Inspect/Asta interfaces. In design documentation, identify which ideas came from ActiveLoop and which utilities came from AstaBench. If code is copied or substantially adapted, retain the applicable MIT or Apache notices, mark modified files as required by Apache-2.0, and preserve AllenAI’s `LICENSE` and `NOTICE.txt`. Do not imply CAIS, ActiveLoop, or AllenAI endorsement.

## 3. License, data-access, and policy gates

Implementation must not begin until these gates have owners and recorded outcomes.

1. **Hugging Face gate:** the operator must personally/organizationally accept access to `cais/hle`. The current API reports `gated: auto`, revision `5a81…`, and 2,500 test records.
2. **Dataset integrity:** honor the card’s requests not to publicly share, re-upload, distribute, or place HLE in training corpora. Preserve and scan for the HLE canary. Never use HLE for prompts, tuning, few-shot exemplars, tool testing, or development fixtures.
3. **External disclosure approval:** an evaluation sends question content and images to model providers. Model-generated search queries may also reveal question fragments to a search backend. Confirm this use is permitted by the HLE access terms and organizational policy before any smoke inference.
4. **Provider retention:** document account-level retention/training settings. For OpenRouter, require a pinned endpoint and `data_collection: deny` where supported. If zero-data-retention is mandatory, verify the selected endpoint is eligible rather than assuming the router setting is sufficient.
5. **No raw data in Git:** `.gitignore` must cover the dataset cache, protected manifests, Inspect `.eval` logs, raw traces, images, model responses, and judge records. Do not bake the dataset or credentials into container layers.
6. **Secret separation:** the host/orchestrator may receive model, search, and HF credentials. The generated-code container receives none. The HF token is used only by the loader and is never exposed to the model, tools, logs, or sandbox.
7. **Public fork implications:** a fork of a public GitHub repository will ordinarily be public. Only code, protocol documents, synthetic fixtures, hashes, and sanitized aggregate results may enter it.
8. **Legal interpretation:** MIT tags and repository licenses do not eliminate gated-access conditions or provider terms. Obtain the project’s normal legal/data-owner approval if external API submission or result publication is not already covered.

## 4. Proposed repository and upstream strategy

### 4.1 Safe fork/remotes workflow — execute only after approval

Use the **existing local planning repository at the exact path below**. It has an unborn `main` branch, zero local commits, and one expected untracked artifact, `PLAN.md`. Do not clone over it, move it, delete it, replace its `.git` directory, or graft unrelated history. After approval, fetch AllenAI’s objects into this repository, create `feat/hle-tools-v0` directly from the pinned upstream commit, and only then stage the existing `PLAN.md` as the first feature-branch change.

The workflow is deliberately one-shot: its preconditions fail if a local commit or local branch already exists. Correct existing `origin`/`upstream` remotes are accepted; wrong or extra remotes stop the workflow rather than being rewritten. `FORK_MODE` makes creation versus inspection explicit so a failed lookup is not silently treated as authorization to create a fork.

```bash
set -euo pipefail

# Required explicit values; stop rather than guessing identity or location.
readonly EXPECTED_REPO_ROOT='/Users/kartik.sreenivasan/kartik_workspace/ai4science_evals/implementations/hle-tools-eval'
readonly UPSTREAM_BASE='0e32055da8f094e9e71d047d23183064b6ebc9d0'
readonly UPSTREAM_URL='https://github.com/allenai/agent-baselines.git'
export EXPECTED_GH_LOGIN='<approved-single-active-github-login>'
export FORK_MODE='<create-new|inspect-existing>'
readonly ORIGIN_URL="git@github.com:${EXPECTED_GH_LOGIN}/agent-baselines.git"

# 1. Enter and verify this exact repository. Never clone into or replace it.
cd -- "$EXPECTED_REPO_ROOT"
test "$(pwd -P)" = "$EXPECTED_REPO_ROOT"
test "$(git rev-parse --show-toplevel)" = "$EXPECTED_REPO_ROOT"
test "$(git rev-parse --is-inside-work-tree)" = 'true'

# 2. Verify the unborn/empty local history and the only expected planning artifact.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo 'Refusing: this repository already has a local HEAD commit.' >&2
  exit 1
fi
test -z "$(git for-each-ref --format='%(refname)' refs/heads/)"
test -f PLAN.md
test -z "$(find . -mindepth 1 -maxdepth 1 ! -name .git ! -name PLAN.md -print -quit)"
test "$(git status --porcelain=v1 --untracked-files=all)" = '?? PLAN.md'

# 3. Verify the single active GitHub identity and existing effective Git identities.
# Do not set, invent, or modify user.name/user.email locally or globally.
gh auth status --hostname github.com
ACTUAL_GH_LOGIN="$(gh api user --jq .login)"
test "$ACTUAL_GH_LOGIN" = "$EXPECTED_GH_LOGIN"
AUTHOR_IDENT="$(git var GIT_AUTHOR_IDENT)"
COMMITTER_IDENT="$(git var GIT_COMMITTER_IDENT)"
test -n "$AUTHOR_IDENT"
test -n "$COMMITTER_IDENT"
printf 'Using existing author identity: %s\n' "$AUTHOR_IDENT"
printf 'Using existing committer identity: %s\n' "$COMMITTER_IDENT"

# 4. Explicitly create a new user fork or inspect an existing one.
case "$FORK_MODE" in
  create-new)
    if gh repo view "$EXPECTED_GH_LOGIN/agent-baselines" >/dev/null 2>&1; then
      echo 'Refusing: the destination already exists; rerun with FORK_MODE=inspect-existing.' >&2
      exit 1
    fi
    # Creates the GitHub fork only: no clone, local remote change, commit, or push.
    gh repo fork allenai/agent-baselines --clone=false
    ;;
  inspect-existing)
    gh repo view "$EXPECTED_GH_LOGIN/agent-baselines" >/dev/null
    ;;
  *)
    echo 'Set FORK_MODE to exactly create-new or inspect-existing.' >&2
    exit 1
    ;;
esac

# In both modes, verify that the destination is the expected user's fork of AllenAI.
test "$(gh repo view "$EXPECTED_GH_LOGIN/agent-baselines" --json owner --jq '.owner.login')" = "$EXPECTED_GH_LOGIN"
test "$(gh repo view "$EXPECTED_GH_LOGIN/agent-baselines" --json isFork --jq '.isFork')" = 'true'
test "$(gh api "repos/$EXPECTED_GH_LOGIN/agent-baselines" --jq '.parent.full_name')" = 'allenai/agent-baselines'
test "$(gh repo view "$EXPECTED_GH_LOGIN/agent-baselines" --json sshUrl --jq '.sshUrl')" = "$ORIGIN_URL"

# 5. Reject unexpected remote names. Add missing expected remotes, but never rewrite one.
test -z "$(git remote | grep -Ev '^(origin|upstream)$' || true)"

if git remote get-url origin >/dev/null 2>&1; then
  test "$(git remote get-url --all origin)" = "$ORIGIN_URL"
  test "$(git remote get-url --all --push origin)" = "$ORIGIN_URL"
else
  git remote add origin "$ORIGIN_URL"
fi

if git remote get-url upstream >/dev/null 2>&1; then
  test "$(git remote get-url --all upstream)" = "$UPSTREAM_URL"
  test "$(git remote get-url --all --push upstream)" = "$UPSTREAM_URL"
else
  git remote add upstream "$UPSTREAM_URL"
fi

test "$(git remote | LC_ALL=C sort)" = "$(printf 'origin\nupstream')"
test "$(git remote get-url --all origin)" = "$ORIGIN_URL"
test "$(git remote get-url --all upstream)" = "$UPSTREAM_URL"
git remote -v

# 6. Fetch both remotes without checking out, merging, rebasing, resetting, or pushing.
git fetch --no-write-fetch-head origin --prune
git fetch --no-write-fetch-head upstream --tags --prune

git show-ref --verify --quiet refs/remotes/origin/main
git show-ref --verify --quiet refs/remotes/upstream/main
git cat-file -e "${UPSTREAM_BASE}^{commit}"
test "$(git rev-parse "${UPSTREAM_BASE}^{commit}")" = "$UPSTREAM_BASE"
git merge-base --is-ancestor "$UPSTREAM_BASE" refs/remotes/upstream/main

# Fetching must not have touched the unborn worktree or planning artifact.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo 'Refusing: HEAD became committed before the feature branch was created.' >&2
  exit 1
fi
test -z "$(git for-each-ref --format='%(refname)' refs/heads/)"
test "$(git status --porcelain=v1 --untracked-files=all)" = '?? PLAN.md'
# Avoid an untracked-file checkout collision if upstream later adds this path.
test -z "$(git ls-tree -r --name-only "$UPSTREAM_BASE" -- PLAN.md)"

# 7. Create an untracked feature branch directly from the pinned AllenAI commit.
git switch --no-track -c feat/hle-tools-v0 "$UPSTREAM_BASE"
test "$(git branch --show-current)" = 'feat/hle-tools-v0'
test "$(git rev-parse HEAD)" = "$UPSTREAM_BASE"
test "$(git status --porcelain=v1 --untracked-files=all)" = '?? PLAN.md'

# Keep a bare future `git push` from auto-creating a remote branch.
git config --local push.autoSetupRemote false
git config --local push.default simple
test -z "$(git config --get branch.feat/hle-tools-v0.remote || true)"

# 8. Add only the existing plan as the first feature-branch change.
git add -- PLAN.md
test "$(git diff --cached --name-only)" = 'PLAN.md'
test "$(git status --porcelain=v1 --untracked-files=all)" = 'A  PLAN.md'
git diff --cached --check
git diff --cached --stat -- PLAN.md

# Stop here. Committing and any explicit push require separate instructions.
```

This sequence contains no deletion, clone, directory replacement, hard reset, rebase, commit, or push. It also gives the feature branch no tracking remote, so the final staged change cannot be pushed implicitly by the workflow. If an organization fork is desired instead, approve its owner and exact URL and replace the user-fork checks as a separately reviewed workflow; do not adapt these commands by inference.

### 4.2 Branch and upstream policy

- `main`: protected integration branch in the fork; initially identical to the recorded upstream base.
- `feat/hle-tools-v0`: implementation and offline tests.
- `exp/hle-tools-v0-smoke` only if run-configuration changes must be separated from implementation; do not commit raw run artifacts.
- Future upstream sync: fetch `upstream`, create `upstream-sync/YYYY-MM-DD`, and **merge** after tests; do not rebase published work or force-push.
- Record the exact `upstream` base SHA in each run manifest. Before implementation, re-check whether `upstream/main` has a necessary security or provider-compatibility fix. Moving off `0e320…` requires an explicit recorded base change, not an implicit pull.
- Keep changes modular enough for an upstream PR, but do not assume AllenAI will accept a gated HLE adapter.

### 4.3 Proposed layout

```text
agent_baselines/
  evals/hle_tools_v0/       # task, loader, scorer, schemas, manifest logic
  solvers/hle_tools_v0/     # common 15-turn solver and tool wiring
solvers/hle-tools-v0/
  pyproject.toml
  uv.lock
  README.md
  env                       # variable names only, never values
  demo.sh                   # defaults to synthetic/offline fixtures
protocols/
  hle-tools-v0.yaml
  schemas/
configs/
  models-v0.yaml
  tools-v0.yaml
docker/hle-tools-v0/
  Dockerfile
  compose.yaml
tests/
  evals/test_hle_tools_v0_*.py
  solvers/test_hle_tools_v0_*.py
```

Pin one tested Python 3.11, `astabench`, and `inspect_ai` set in the solver lockfile. The inspected ReAct subproject uses `astabench==0.5.3` and `inspect_ai==0.3.203`; use that only as the initial compatibility candidate, then freeze the actually tested lock. Do not depend on mutable `main` branches or unpinned Git URLs.

## 5. `hle-tools-v0` protocol proposal

Every result must be described as:

> `cais/hle@5a81a4…`, `hle-tools-v0`, one attempt, common external search/fetch and stateful Python scaffold, 15 model turns inclusive of final submission, pinned provider/model/judge/tool/container revisions.

It must not be described simply as “HLE with tools.”

### 5.1 Dataset and sample construction

- Dataset: `cais/hle`, revision **`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`**, split `test`.
- Loader hard assertions: 2,500 rows; unique nonempty `id`; known `answer_type`; question present; image field parseable when nonempty.
- Compute and record the downloaded Parquet SHA-256 locally. Fail closed if the HF revision, file hash, schema, or count changes.
- Retain `answer`, `rationale`, and rationale images only in scorer-side protected state. Solver state and sandbox receive only ID, question, answer type, and question image.
- Use one attempt per model/question (`Pass@1`). Do not rerun incorrect or low-confidence answers.
- Treat missing, malformed, timed-out, or exhausted samples as incorrect in primary accuracy; report completed-only accuracy only as a diagnostic.

### 5.2 Multimodal loader

Convert HLE records to Inspect `Sample` objects with `ContentText` and `ContentImage`:

- preserve the original data URI/bytes and MIME type where the provider accepts them;
- validate base64, MIME, decoded size, pixel dimensions, and decompression limits;
- do not silently convert or resize images;
- if provider compatibility requires conversion (for example GIF first-frame conversion), version the transform, retain original and transformed hashes, and apply the same transform to all models;
- optionally mount the question image read-only in the code sandbox, but never mount the answer or rationale;
- add synthetic fixtures for PNG, JPEG, GIF, malformed data URI, empty image, and provider serialization; never use a real HLE question as a test fixture.

The one-question smoke should use a multimodal record if all primary models are available; otherwise use a text record for the provider smoke and require a separate no-inference multimodal serialization test before the pilot.

### 5.3 Common scaffold

- Same system/developer instruction and tool descriptions for all models, with only the minimum provider role/serialization adaptations.
- Native function calling through Inspect; no provider-native browsing, code interpreter, URL context, or model-specific agent product.
- ReAct state is the full API-visible conversation plus tool events. Do not require or claim access to hidden chain-of-thought.
- A model may answer without using tools.
- A “turn” is one model generation plus all tool calls returned by that generation and their results.
- **15 model turns total, inclusive of final submission.** Turns 1–14 may use tools. Turn 15 disables tools and requires `submit_answer`. There is no uncounted 16th force-answer call.
- Proposed parallel/action cap: at most 4 tool calls in one turn and 30 tool calls per sample. Duplicate identical calls may be served from the campaign cache but still count as actions.
- Rationale for 15: it matches the most visible ActiveLoop convention while avoiding its hidden extra final call; it is enough for search/fetch/code iteration but bounds cost. The pilot must report accuracy versus turns used and the number hitting the cap. If many answers terminate before 8 turns, consider a cheaper v1; if more than 10% hit the cap while making progress, review traces before raising it.

### 5.4 Structured answer and confidence

Expose a submission tool with a strict payload:

```json
{
  "answer": "succinct final answer or MC choice",
  "confidence": 0.73,
  "explanation": "optional concise explanation"
}
```

Rules:

- `confidence` is the model’s probability that its submitted answer is correct, in `[0, 1]`.
- Preserve the raw final message and structured payload.
- Render a CAIS-compatible response (`Explanation`, `Answer`/`Exact Answer`, `Confidence`) for the compatibility judge.
- The primary judge receives only the question, gold answer, answer type, and rendered final response—not model identity, provider, trace, cost, or tools used.
- If turn 15 still lacks a valid submission, preserve the last response for diagnostics, mark `submission_invalid`, score incorrect, and set calibration confidence to 0. Do not silently invent 100% confidence.

Structured submission reduces extraction ambiguity and is a deliberate protocol deviation from direct HLE generation; it therefore belongs in the `hle-tools-v0` name and requires approval.

## 6. Versioned tool and isolation contract

### 6.1 `web_search_v0`

Proposed interface:

```text
web_search(query: string, max_results: integer = 5) ->
  [{rank, title, url, snippet, backend, retrieved_at, content_hash}]
```

- One common external backend for all models; **Tavily versus Exa is unresolved**.
- `query` at most 512 characters; `max_results` 1–10; 30-second timeout; normalized output at most 15,000 characters.
- Block `huggingface.co/datasets/cais/hle`, known mirrors, direct HLE answer/rationale pages, localhost/private/link-local/cloud-metadata addresses, and non-HTTP(S) schemes.
- Scan queries and results for the HLE canary and obvious benchmark leakage; quarantine the sample and retain a redacted incident record if detected.
- Cache exact normalized requests within the campaign so identical requests receive identical results. Record backend version, request ID, timestamp, and raw-response hash. The protected raw response enables replay/audit; it is not committed.

### 6.2 `fetch_url_v0`

```text
fetch_url(url: string) ->
  {requested_url, final_url, status, title, text, retrieved_at, content_hash, truncated}
```

- Fetch only HTTP(S) URLs returned by that sample’s previous `web_search_v0` call, unless a future protocol explicitly permits question-supplied URLs.
- Resolve and validate DNS before each redirect; block private/link-local/metadata targets; maximum 2 redirects.
- 15-second timeout, 2 MB response limit, allowlisted textual/document MIME types, no JavaScript execution, readable-text output capped at 20,000 characters.
- Preserve source URL and hash. Return explicit errors to the model; the harness does not reinterpret failures as empty success.

### 6.3 `python_v0`

```text
python(code: string) -> {stdout, stderr, exit_status, duration_ms, truncated}
```

- Stateful within one sample; fresh environment between samples.
- Suggested preinstalled set: Python standard library, NumPy, SciPy, SymPy, pandas, NetworkX, matplotlib, and a small number of justified pure-computation packages. Freeze every package in the lock/image digest.
- No package installation during a run; no shell escape that reaches the orchestrator; no network.
- Per call: 60 seconds, 2 CPU, 8 GB RAM, 256 PIDs, 10,000 output characters. Per sample: 5 minutes cumulative code time and 100 MB writable workspace.
- Store code cells and outputs in the protected trace.

### 6.4 Strong isolation

Generated code must run in a per-sample disposable container (or stronger disposable VM) with:

- `network_mode: none`;
- non-root UID, read-only root filesystem, `cap_drop: [ALL]`, `no-new-privileges`, seccomp/AppArmor where available;
- CPU, memory, PID, file-size, and wall-time limits;
- only an ephemeral work directory and optional read-only question image mount;
- no API keys, HF cache, host home, repository `.git`, Docker socket, or model/search credentials;
- container removal and volume deletion after every sample, including failures.

Avoid mounting the host Docker socket inside a container that processes untrusted model code; that effectively grants host control. Prefer Inspect running against rootless Docker/Podman or a dedicated disposable runner/VM. AstaBench’s stateful Jupyter integration is the design reference, but its stock bridge-networked 48 GB sandbox is not the final security profile.

## 7. Model matrix and adapters

### 7.1 Live catalog result, without inference

| Requested model | Provider and exact API ID | Planned Inspect ID | Availability checked 2026-08-25 | Proposed condition |
|---|---|---|---|---|
| GPT-5.6 Sol | OpenAI direct: **`gpt-5.6-sol`** | `openai/gpt-5.6-sol` | Live. Do not use mutable alias `gpt-5.6`. No dated snapshot is currently documented. | Responses API; high reasoning; common tools only. |
| Claude Opus 5 | Anthropic direct: **`claude-opus-5`** | `anthropic/claude-opus-5` | Live; Anthropic documents the dateless ID as the canonical pinned model ID. | Messages API; high effort; common tools only. |
| Newest current Gemini Flash | Google direct: **`gemini-3.7-flash`** | `google/gemini-3.7-flash` | Live and currently the newest stable general-purpose Flash. | Google GenAI API; high thinking; common tools only. |
| Qwen3.5-397B-A17B | OpenRouter: **`qwen/qwen3.5-397b-a17b`** | `openrouter/qwen/qwen3.5-397b-a17b` | **Live**, so fallbacks are currently dormant. Catalog reports text/image/video input, tools, 262,144 context. | OpenRouter, reasoning enabled, common tools only, endpoint pinned. |

Proposed shared generation settings are temperature `0` where accepted, maximum 32,768 output tokens per generation, and high reasoning/thinking where the provider exposes it. The adapter must record requested and effective settings and fail preflight if a parameter is silently ignored. Qwen exposes reasoning enablement rather than the same effort scale; record that asymmetry rather than claiming identical compute.

For OpenRouter Qwen, proposed endpoint condition is **Alibaba-only**, currently listed as `Alibaba | qwen/qwen3.5-397b-a17b-20260216`, with router fallbacks disabled, required tool parameters, and data collection denied where supported. Capture the endpoint catalog response and price immediately before the campaign. If Alibaba does not satisfy retention or image/tool requirements, select another endpoint before the smoke and version that change; never permit opaque mid-run routing.

### 7.2 Requested fallbacks

If Qwen becomes unavailable before the campaign, do not silently substitute a model. Pause, obtain approval, update the run matrix/protocol metadata, and restart that lane under one of these exact OpenRouter IDs:

1. GLM-5.1: **`z-ai/glm-5.1`** (`openrouter/z-ai/glm-5.1` in Inspect).
2. Nemotron 3 Ultra: **`nvidia/nemotron-3-ultra-550b-a55b`** (`openrouter/nvidia/nemotron-3-ultra-550b-a55b` in Inspect).

The paid IDs above are the evaluation candidates. Although `z-ai/glm-5.1:free` and `nvidia/nemotron-3-ultra-550b-a55b:free` are listed, free routing/rate limits are too mutable for the controlled lane.

Critical limitation: both fallback catalog entries are currently **text-only**, so they are not drop-in replacements for Qwen on a multimodal subset. Approved options are either (a) run the fallback only on a predeclared text-only lane and label it non-comparable, or (b) delay the open-model lane until a suitable vision-capable endpoint is available. Do not add an unplanned vision captioner; that would create a different scaffold.

### 7.3 Adapter invariants

- Direct provider APIs for OpenAI, Anthropic, and Google; OpenRouter only for Qwen/fallbacks.
- No provider-native search or code tools.
- Same tool schemas, prompt, limits, images, and subset.
- Provider serialization differences are covered by synthetic tests.
- OpenRouter: `provider.order` pinned, `allow_fallbacks: false`, `require_parameters: true`, and endpoint/data-policy metadata retained.
- If an endpoint fails mid-lane, pause. Do not mix endpoints or models in one aggregate.

## 8. Evaluator, judge, and metrics

### 8.1 Primary correctness path

- Start from the CAIS `JUDGE_PROMPT` and structured `ExtractedAnswer` schema.
- Proposed compatibility judge: **`o3-mini-2025-01-31`**, the exact current CAIS default. It is still available but deprecated, so approval and a preflight availability check are required.
- Pin judge model ID, prompt text/hash, decoding parameters, client version, and retry policy.
- For MC questions, also compute deterministic normalized-choice agreement. Any disagreement with the LLM judge is an audit item.
- For exact answers, the judge compares only the structured submitted answer with gold; it must not solve the question or inspect the trace.
- Judge transport/schema failures receive up to two retries. Persist all attempts. An unresolved judge result blocks pilot acceptance; it is not silently converted to correct or omitted.
- Blind manual audit: all deterministic/LLM disagreements, all judge failures, and a predeclared 10% stratified sample of remaining judgments. Record adjudication separately; do not overwrite the original judge output.

If `o3-mini-2025-01-31` is retired, choose and validate a replacement on synthetic and non-HLE equivalence fixtures, version it as a protocol amendment, and rejudge the whole campaign. Never mix judge models inside one aggregate.

### 8.2 Reported metrics

Primary:

- accuracy with **Wilson 95% interval**;
- failures/timeouts counted as wrong;
- answer coverage and valid-confidence coverage;
- total and per-correct-answer cost, tokens, wall time, turns, and tool calls.

Calibration (from the evaluated model’s submitted confidence, not judge confidence):

- Brier score;
- log loss with documented clipping;
- 5-bin equal-frequency ECE and reliability table for the 50-question pilot;
- pilot RMS calibration gap with the bin definition printed in metadata.

The repository’s legacy CAIS calibration routine uses roughly 100-example bins and is unsuitable for `n=50` as written; it also requires a compatibility audit before use. Do not present a small-pilot replacement as the official HLE calibration number. For any later large run, report both a byte-compatible legacy metric (if reproducible) and a clearly named corrected metric.

Secondary/descriptive:

- accuracy by category, answer type, and multimodal status;
- macro category accuracy alongside micro accuracy;
- paired model win/loss/tie table on the same 50 IDs;
- judge agreement and manual-audit disagreement;
- tool-use rate and termination reason.

At `n=50`, subgroup and pairwise results are diagnostic, not leaderboard-grade. Avoid strong ranking claims.

## 9. Subset selection

### 9.1 One-question smoke

- Select one question outside the 50-question pilot using metadata only.
- Prefer multimodal and exact-answer plumbing if all primary models support it.
- Use the same question for all four adapters so one “smoke task” exercises the matrix.
- Do not include smoke scores in the pilot aggregate. Configuration may change after the smoke.
- The model APIs are stateless, but the smoke ID remains excluded from the 50 to avoid any appearance of repeated tuning on an evaluated item.

### 9.2 Stratified 50-question pilot

Build the manifest once, before any HLE model output is examined:

1. Load only `id`, `category`, `raw_subject`, `answer_type`, and `image_present`; do not inspect answer/rationale or manually judge difficulty.
2. Allocate proportionally by top-level category using largest remainder, with at least one sample for every sufficiently represented category.
3. Match the population’s MC/exact proportions as closely as integer constraints allow.
4. Guarantee enough multimodal coverage to test the loader (proposed minimum 8 of 50) while publishing both raw and population-weighted diagnostics if this oversamples images.
5. Within cells, rank by `SHA256("hle-tools-v0-pilot-50" || id)` and take the first required records. Resolve sparse cells with deterministic swaps that minimize deviation across category, answer type, and image status.
6. Emit a protected ID manifest, public manifest hash, population-versus-pilot balance table, algorithm version, and seed. Do not publish question text or gold.
7. Use exactly the same 50 IDs and order for all primary models. Interleave model execution order by a deterministic schedule to reduce time-of-day/search drift.

The pilot is intended to find plumbing, scaffold, evaluator, and cost problems, not to estimate a final score precisely.

### 9.3 Optional 10–20% expanded pilot

Only after the 50-question review and separate approval, construct 250 or 500 questions with the same algorithm, excluding the smoke but not necessarily excluding the initial 50. Decide in advance whether the 50 are nested (cheaper) or held out (cleaner). Freeze that decision before inspecting pilot correctness. A full-set command is intentionally absent from this plan.

## 10. Budgets and retries

### 10.1 Proposed per-sample ceilings

| Resource | Ceiling |
|---|---:|
| Model turns | 15 total, final turn submit-only |
| Tool calls | 4/turn, 30 total |
| Metered model input | 200,000 cumulative tokens |
| Metered model output | 50,000 cumulative tokens; 32,768 max in one generation |
| Wall time | 30 minutes |
| Active working time | 20 minutes |
| Stateful code | 60 seconds/call, 5 minutes cumulative |
| Search | 30 seconds/call, max 10 results |
| Fetch | 15 seconds/call, 2 MB and 20,000 extracted characters |

Implement Inspect `turn_limit`, `token_limit`, `time_limit`, and `working_limit` where supported, plus explicit tool/action counters. Record spend continuously but do not configure `cost_limit` or any other monetary stop gate. Record which non-monetary protocol limit terminated a sample.

### 10.2 Retry policy

- Model transport/rate-limit retries: maximum 2 with bounded exponential backoff; preserve attempt metadata.
- Tool backend retry: maximum 1 for a clearly transient transport failure; logical “no results” is not retried by the harness.
- Whole-sample retry: at most 1, only for a verified infrastructure failure before a valid answer. Preserve the failed attempt and mark the replacement. Never retry wrong, low-confidence, timed-out, or budget-exhausted answers.
- Judge: up to 2 retries for transport/schema failure.
- OpenRouter provider fallback: disabled. Mid-run provider/model substitution is prohibited.

### 10.3 Cost estimate

Current public list prices used for planning (per million input/output tokens):

- GPT-5.6 Sol: `$4 / $20`;
- Claude Opus 5: `$5 / $25`;
- Gemini 3.7 Flash introductory: `$0.75 / $3.75`;
- OpenRouter Qwen catalog: conservatively `$0.50 / $3.60`; the currently proposed Alibaba endpoint lists about `$0.39 / $2.34`.

At the hard 200k-input/50k-output model budget, one four-model question is approximately `$4.67` using the conservative Qwen price. The one-question smoke plus 50-question pilot is 51 × 4 = **204 sample attempts**, or about **$238 maximum solver-token spend** before search, judge, and retry buffer. With up to 30 paid searches at `$0.005` each, the absolute search allowance is about `$31`; fetch is assumed to have no per-call fee. Judge cost should be measured in smoke and is expected to be small relative to solver cost.

Accounting expectation, not a stop gate: the smoke plus 50-question pilot is expected to cost **$70–$110** if samples use about 60k input/15k output tokens and terminate well before the hard protocol limits. Actual spend must be recorded from provider usage and reported after each phase. Do not stop or omit a scheduled sample solely because a monetary estimate is exceeded.

Prices and promotional periods can change. The no-inference preflight must capture price/catalog evidence and recompute the estimate for reporting. A 250–500 expanded pilot is roughly 5–10× the 50-question solver burden and remains outside the current authorization.

### 10.4 Time estimate

Implementation to a trustworthy 50-question pilot: approximately **5–8 engineer-days**:

- fork/environment/pins and synthetic fixtures: 0.5–1 day;
- dataset/multimodal loader and subset manifest: 1 day;
- search/fetch broker and hardened stateful sandbox: 1.5–2.5 days;
- model adapters, budgets, and trace integration: 1–1.5 days;
- judge, metrics, artifact schemas, and offline tests: 1–1.5 days;
- smoke diagnosis and pilot operations: 0.5–1.5 days.

At an expected 5–12 minutes per sample, 204 attempts represent about 17–41 serial agent-hours. Running at one sample per provider concurrently gives roughly 4–10 hours of model wall time; the hard 30-minute ceiling implies 102 serial hours or about 25.5 hours at four-way provider concurrency. Start with concurrency 1 per provider and raise it only after the smoke confirms memory, rate limits, and sandbox cleanup.

## 11. Artifact and results schema

### 11.1 Storage classes

- **Git-safe:** protocol YAML, JSON Schemas, source code, synthetic fixtures, subset hash/balance report, container digest, sanitized aggregate tables.
- **Protected:** subset IDs, prompts, images, gold/rationales, raw responses, judge records, search/fetch bodies, code cells, Inspect `.eval` logs, and traces.
- **Secrets:** never written to artifacts; scan logs and diffs before sharing.

### 11.2 Run manifest

Each `run_manifest.json` should include:

```text
run_id, protocol_version, code_sha, upstream_sha, dirty_tree
harness_versions, dependency_lock_hash, container_image_digest, platform
dataset_repo, dataset_revision, dataset_file_sha256, split, expected_count
subset_name, subset_size, subset_manifest_hash, selection_seed/algorithm
model_label, inspect_model_id, provider, exact_api_model_id
provider_endpoint, endpoint_catalog_hash, requested/effective generation config
scaffold_prompt_hash, tool_contract_versions, search_backend/version
judge_model_id, judge_prompt_hash, scorer_version
turn/token/time/cost/retry budgets, concurrency
start/end timestamps, operator-approved budget ID
```

Fail a release run if the Git tree is dirty unless the manifest explicitly marks a smoke/development run.

### 11.3 Per-sample result

One JSONL row per model/question:

```text
run_id, protected_sample_id, sample_id_hash, strata
status, termination_reason, submission_valid
answer, confidence, explanation_hash
correct, judge_status, judge_model_id, judge_output_hash, audit_status
input/output/cache/reasoning tokens, reported and estimated cost
wall/working/code time, turns, tool-call counts by version
provider attempts/retries, tool errors, sample retry lineage
trace_path/hash, search/fetch cache hashes, image transform/hash
```

Keep the plain sample ID and content only in protected storage. Sanitized exports use salted/stable hashes as approved.

### 11.4 Trace events

Use Inspect’s trace/eval logs and add normalized events for model request/response metadata, tool calls/results, sandbox lifecycle, limits, retries, submission, judge, and scoring. Store API-visible reasoning only when the provider returns it and policy permits retention; do not infer hidden reasoning. Redact authorization headers, query parameters that contain credentials, environment dumps, and local paths.

## 12. Phased implementation and commands

All commands below are **future commands after approval**, not commands run during planning. The implementation should provide the named CLI before use.

### Phase 0 — approve and pin

Exit criteria: all decisions in section 16 resolved, fork owner approved, dataset/provider use cleared, dollar cap approved.

Then execute the identity-checked repository-reuse workflow in section 4.1. It creates `feat/hle-tools-v0` from the pinned AllenAI commit inside this exact repository and stages the existing `PLAN.md` as the first feature-branch change; it does not clone, commit, or push.

### Phase 1 — isolated environment and offline plumbing

```bash
# Build only the pinned development/runner image.
make build-image SOLVER=hle-tools-v0
make shell SOLVER=hle-tools-v0

# Inside the isolated development shell; exact lock, no updates.
uv sync --project solvers/hle-tools-v0 --python 3.11 --frozen
uv run --project solvers/hle-tools-v0 --frozen -- pytest -q \
  tests/evals/test_hle_tools_v0_loader.py \
  tests/evals/test_hle_tools_v0_scorer.py \
  tests/solvers/test_hle_tools_v0_tools.py \
  tests/solvers/test_hle_tools_v0_sandbox.py \
  tests/solvers/test_hle_tools_v0_adapters.py

# Catalog/data/sandbox checks only; explicitly prohibit inference.
uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli preflight \
  --protocol protocols/hle-tools-v0.yaml --no-inference
```

Offline acceptance:

- pinned dataset resolves to 2,500 and the expected schema without logging content;
- synthetic multimodal serialization works for every adapter;
- search/fetch SSRF and HLE-block tests pass;
- code sandbox has no network, credentials, host mounts, or cross-sample state;
- model IDs/endpoints/prices are present in catalogs without calling inference APIs;
- judge/scorer fixtures and calibration calculations pass;
- all artifacts route to protected or Git-safe locations correctly.

### Phase 2 — freeze manifests

```bash
uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli manifest build \
  --dataset cais/hle \
  --revision 5a81a4c7271a2a2a312b9a690f0c2fde837e4c29 \
  --smoke-size 1 --pilot-size 50 \
  --seed hle-tools-v0-pilot-50 \
  --protected-output protected/manifests/

uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli manifest audit \
  protected/manifests/pilot-50.json
```

Review only metadata balance and hashes. Freeze the protocol, code SHA, manifests, and model endpoint catalog before inference; record prices for accounting without creating a monetary stop gate.

### Phase 3 — one-question smoke matrix

```bash
uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli run \
  --protocol protocols/hle-tools-v0.yaml \
  --models configs/models-v0.yaml \
  --manifest protected/manifests/smoke-1.json \
  --max-samples 1 --max-concurrency-per-provider 1 \
  --log-dir protected/runs/smoke-1/

uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli score \
  protected/runs/smoke-1/
```

Smoke acceptance:

- exactly one result for every approved primary model;
- correct model/provider/endpoint IDs and effective settings in manifests;
- image delivered if the smoke is multimodal;
- valid structured answer and confidence or a clearly diagnosed protocol failure;
- complete model/tool/sandbox/judge trace, token/cost accounting, and cleanup;
- no secret, gold answer, rationale, or host path exposed to model/sandbox/search logs;
- no unexpected provider-native tool or OpenRouter route;
- total spend and time recorded and reviewed.

Do not proceed automatically. Fix kinks, rerun offline checks, and if the protocol changes, create a new smoke run rather than overwriting evidence.

### Phase 4 — stratified 50-question pilot

```bash
uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli run \
  --protocol protocols/hle-tools-v0.yaml \
  --models configs/models-v0.yaml \
  --manifest protected/manifests/pilot-50.json \
  --max-concurrency-per-provider 1 \
  --log-dir protected/runs/pilot-50/

uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli score \
  protected/runs/pilot-50/ \
  --audit-manifest protected/manifests/judge-audit.json

uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli report \
  protected/runs/pilot-50/ \
  --sanitized-output artifacts/pilot-50-summary/
```

Pilot acceptance:

- 50 scheduled outcomes per model on the identical manifest (200 total), with missing/failure outcomes retained;
- no mixed model, endpoint, protocol, dataset, judge, or tool revisions within a lane;
- all limits/retries/costs/traces accounted for;
- all judge failures and MC disagreements adjudicated; predeclared audit completed;
- no unresolved secret/data leakage incident;
- spend fully recorded and included in the sanitized report;
- balance report, accuracy/CI, calibration diagnostics, subgroup diagnostics, cost, tool use, and failure table produced;
- a written kink review decides whether to stop, modify/version the protocol, rerun the 50, or request an expanded-pilot approval.

### Phase 5 — optional expanded pilot, not default

After a separate user decision, generate and run 250 or 500 IDs with a newly approved manifest and budget. Reuse the Phase 4 command shape; do not add a 2,500-record target or full-run shortcut. A full evaluation requires a later proposal based on observed pilot usage, judge reliability, failure rate, and cost.

## 13. Failure modes and responses

| Failure | Response |
|---|---|
| HF access denied or acceptance terms unclear | Stop before download/inference; obtain access/approval. |
| Revision no longer resolves to exactly 2,500 | Stop; do not follow `main` or HLE-Rolling. Review a protocol revision. |
| Raw question/gold enters Git, image layer, or public artifact | Quarantine; do not push; remove from history only with an approved safe procedure; rotate affected credentials if present. |
| Search returns HLE answers/canary | Stop/quarantine sample, retain redacted evidence, review block policy; do not count leaked answer as ordinary success. |
| Search changes over time | Retain timestamps/hashes and campaign cache; interleave models; label results as time-bound. |
| Provider-native search activates | Fail the sample/preflight; the controlled scaffold requires common external tools. |
| OpenRouter silently changes endpoint | Disable fallbacks, verify response endpoint metadata, pause lane on mismatch. |
| Qwen unavailable | Pause; approve a fallback. GLM/Nemotron are text-only and cannot silently join the multimodal aggregate. |
| Model rejects image/tool/schema/temperature | Fail preflight or smoke; adapt serialization only, record effective settings, and rerun smoke under a versioned config. |
| Context/token/turn/time limit | Preserve trace; score incorrect; report termination reason; do not semantic-retry. |
| Rate limit/provider outage | Bounded transport retry; then pause if systemic. Do not switch providers mid-lane. |
| Code hangs, forks, or attempts network/host access | Enforce container limits, kill and destroy sample sandbox, preserve security event, review before continuing. |
| Sandbox state leaks between samples | Stop campaign; invalidate affected runs; fix isolation and rerun from a clean protocol version. |
| Judge schema failure or disagreement | Retry only schema/transport failures; audit disagreements; unresolved judgment blocks acceptance. |
| Deprecated judge disappears | Select/validate one replacement and rejudge the complete campaign under a protocol amendment. |
| Confidence missing/malformed | Mark invalid submission and wrong with confidence 0 under the proposed schema; report separately. |
| AstaBench/Inspect version skew | Use the frozen subproject lock; no in-place upgrades during a campaign. |
| Cost telemetry absent or price changes | Preserve the run, reconcile provider usage, and label any estimate from partial logs; monetary spend alone is not a stop gate. |
| Pilot subset is too small for ranking | Treat results as diagnostics; do not claim leaderboard equivalence or definitive model ordering. |

## 14. Cleanup and retention

Use scoped cleanup only; never broad `docker system prune` or recursive deletion through unresolved variables.

After each sample, automatically destroy that sample’s container and ephemeral volume. After a phase:

```bash
# Use the exact compose project name emitted in the run manifest.
docker compose --project-name hle-tools-v0-<verified-run-id> \
  -f docker/hle-tools-v0/compose.yaml down --volumes --remove-orphans
```

Then:

- verify no `hle-tools-v0-*` containers or volumes remain;
- remove only verified temporary build/test artifacts;
- delete temporary `.env`/secret mounts and shell history entries containing secrets;
- retain protected run artifacts according to the approved policy, encrypted and access-controlled;
- optionally remove the gated HF cache with an exact verified path if retention is not approved;
- retain Git-safe hashes/manifests and sanitized summaries;
- scan `git status`, `git diff`, container build context, and report directories for dataset content, secrets, and large generated files.

Do not delete raw traces needed for an unresolved leakage, billing, or judge investigation until that review is complete.

## 15. Acceptance criteria before any larger evaluation

A larger-than-50 campaign is not approved merely because commands complete. It requires all of:

1. fixed HLE revision/file hash/count and protected-data handling verified;
2. fork/remotes/identity correct and reproducible environment locked;
3. common scaffold and tools confirmed across all model lanes;
4. multimodal success on primary models;
5. no sandbox escape, cross-sample state, credential exposure, or HLE search leak;
6. valid trace/usage/cost data for every scheduled sample, including failures;
7. judge audit complete with no unresolved systematic bias or schema issue;
8. observed failure/invalid-submission rate acceptable (proposed threshold: <5%, with no common root-cause cluster);
9. observed cost and time are reported for planning any larger campaign;
10. protocol changes from smoke/pilot are versioned and, if material, the 50-question pilot is rerun rather than mixed;
11. user explicitly approves the expanded manifest, models/endpoints, judge, and concurrency.

## 16. Protocol decisions

The bounded smoke-and-50 execution is approved with the user overrides above. The defaults below govern this run unless a concrete availability, access, or semantic blocker is recorded; choices about an expanded or full run remain unapproved:

1. **Pilot interpretation:** approve one-question smoke → 50-question pilot → optional 250 or 500 expanded pilot, acknowledging 50 is 2%, not 10–20%.
2. **Fork owner/visibility:** approved GitHub user versus organization, expected Git name/email, and whether an existing fork must be reused.
3. **Upstream base:** use inspected `agent-baselines@0e320…` or re-pin a newer reviewed upstream SHA at implementation start.
4. **Dataset/provider legality:** approve sending gated HLE text/images to the four providers and model-generated query fragments to the chosen search backend.
5. **Scope:** all fixed HLE categories (recommended) versus a STEM-only derivative, which would need a different protocol name and weighting.
6. **Search backend:** Tavily or Exa; retention, corpus, region, pricing, and campaign cache policy.
7. **Scientific retrieval:** exclude ActiveLoop Scientific Search from v0 (recommended) or define a separate, non-comparable lane.
8. **Scaffold budget:** approve 15 turns inclusive of final submission, 30 total tool calls, token/time/code ceilings, and no hidden final call.
9. **Generation policy:** temperature 0 where accepted, high provider reasoning/thinking, 32,768 per-generation output cap, and how to handle a model that cannot honor a setting.
10. **Qwen endpoint/privacy:** Alibaba-only proposal, no router fallback, required parameters, retention requirements, and response if that endpoint lacks needed image/tool behavior.
11. **Open-model fallback:** if needed, choose text-only GLM-5.1, text-only Nemotron 3 Ultra, or delay; approve that no fallback can silently enter the multimodal aggregate.
12. **Judge:** keep deprecated but official `o3-mini-2025-01-31` for compatibility, or select/validate a modern judge before the first campaign.
13. **Answer contract:** strict structured `answer`/`confidence` submission and invalid-submission-as-wrong behavior.
14. **Metrics:** Wilson accuracy interval, failures-as-wrong, pilot calibration diagnostics, and legacy HLE calibration only as a separately labeled compatibility metric.
15. **Subset constraints:** proposed seed, minimum 8 multimodal records, proportional category/answer-type allocation, smoke outside pilot, and whether a later expanded pilot nests the initial 50.
16. **Spend/concurrency:** record actual spend with no monetary stop gate; start at concurrency 1 per provider and increase only if smoke evidence supports it.
17. **Trace retention/publication:** duration and access for raw questions/responses/search results/code; what sanitized aggregates may be published.
18. **Success threshold:** proposed <5% infrastructure/invalid-submission rate and mandatory resolution of judge/security/data incidents before expansion.

## 17. Immediate next action after approval

The first action is the identity-checked fork/remotes workflow—not model inference. The first implementation milestone is an offline synthetic fixture passing through loader → common tools → isolated sandbox → structured submission → scorer → trace. Only after that and a no-inference catalog/data preflight should the one-question, four-provider smoke be authorized.
