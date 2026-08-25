# HLE pilot stall diagnosis and HLE-Verified migration

Date: 2026-08-25

## Pinned canonical sources

- ActiveLoop: [`activeloopai/hle_with_tools@7a348ce23658ecbc6dd5206382c92c4016d2471a`](https://github.com/activeloopai/hle_with_tools/tree/7a348ce23658ecbc6dd5206382c92c4016d2471a)
- AllenAI: [`allenai/agent-baselines@0e32055da8f094e9e71d047d23183064b6ebc9d0`](https://github.com/allenai/agent-baselines/tree/0e32055da8f094e9e71d047d23183064b6ebc9d0)
- Dataset: [`skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`](https://huggingface.co/datasets/skylenage-ai/HLE-Verified/tree/0bc83643672d4f68a5f89998617a639d85e7318b)
- Local runtime at the incident: `inspect_ai==0.3.203`, `openai==2.30.0`, `httpx==0.28.1`, code `4a0f85d`.

## Why the pilot appeared stalled

The preserved `.eval` header is `status=started`, declares 50 samples, and contains no model usage or completed sample. The runner began at 12:04 local time, created two sample sandboxes, and had two OpenAI HTTPS connections when it was stopped at about 12:47. The secret wrapper only reads authorized key files and then uses `os.execvpe`; it adds no waiting, retry, timeout, or process layer.

The decisive evidence is the host power log:

- 12:08:25: macOS entered idle sleep, about four minutes after pilot launch.
- Subsequent activity was maintenance dark wake, not normal runnable user time.
- 12:46:10: the host woke from deep idle on lid/HID activity.
- The run was stopped shortly afterward.

Therefore the observed 42 minutes were mostly suspended-host wall time, not 42 minutes inside two runnable OpenAI calls. There is no evidence of an OpenAI outage, Exa failure, or Inspect ignoring an expired active timer.

Inspect 0.3.203 wraps each provider generation in `anyio.move_on_after(attempt_timeout)` and each sample in another AnyIO cancellation scope. These are cooperative event-loop timers. They cannot progress or cancel network work while the machine is asleep. The configured 600-second attempt and 1,800-second sample limits had not received their corresponding amount of runnable time before operator termination. The OpenAI SDK also defaults to a 600-second HTTP timeout and two internal retries, but the partial log does not establish that either SDK timeout or retry occurred.

**Smallest credible correction:** keep Inspect and the existing agent, but run live campaigns under a host-awake guard (`caffeinate -i` on macOS). Do not infer provider failure from process elapsed time without checking host sleep. No custom watchdog or orchestration framework is warranted. Provider SDK timeout/retry overrides would change the model condition and are not supported by this incident evidence.

## Implementation lineage and fork relationship

This repository is the user's GitHub fork of AllenAI's repository: local `origin` is `git@github.com:ksreenivasan/agent-baselines.git`, while `upstream` is `https://github.com/allenai/agent-baselines.git`. The feature branch `feat/hle-tools-v0` starts at and remains based on AllenAI commit `0e32055da8f094e9e71d047d23183064b6ebc9d0`; nothing has been pushed by this work.

### Upstream components reused

- **AllenAI `agent-baselines`:** repository layout, per-solver environment pattern, and `agent_baselines/solvers/react/basic_agent.py` as the agent-loop starting point. Upstream provides a general ReAct agent, not an HLE task.
- **AllenAI/AstaBench dependency:** inherited `submit` tool and submission manager, tool merging/execution, and `stateful_python`. We did not replace the submit wrapper or create a custom submission tool.
- **Inspect AI:** task/dataset/scorer interfaces, provider adapters, model generation and retry configuration, AnyIO-backed sample/attempt limits, evaluation logs, concurrency controls, Docker sandbox lifecycle, and `eval-retry`.
- **Official HLE evaluation conventions:** one-attempt evaluation, MC versus exact-answer distinction, confidence collection, and the pinned `o3-mini-2025-01-31` answer-equivalence judge convention. The scorer here is a narrow local implementation; it is not claimed to be byte-identical official HLE code.
- **HLE-Verified dataset:** final/revised question, answer, answer type, images, class label, and verification metadata from the pinned `json` record. No dataset content is copied into Git.

ActiveLoop commit `7a348ce23658ecbc6dd5206382c92c4016d2471a` was inspected as a comparison source only. No ActiveLoop source code, proprietary scientific-search service, custom asyncio runner, in-process interpreter, or extra final-answer call was imported.

### Local changes before HLE-Verified migration

Relative to the pinned AllenAI starting point, the branch added:

1. an HLE task, fixture loader, structured scorer, deterministic manifest generator, and preflight CLI;
2. an HLE configuration of AllenAI's ReAct agent with common Exa search, controlled fetch, and Asta stateful Python;
3. a disposable no-network Docker sandbox and runtime-only secret injector;
4. one local extension to AllenAI's `basic_agent`: `final_step_submit_only`, which exposes only the inherited submit tool on turn 15;
5. a strict local submission payload (`answer`, `confidence`, `explanation`) serialized inside the inherited top-level `submission` string;
6. focused fixes that return web transport/policy failures to the model, correct judge-prompt brace formatting, reserve the final turn, remove the unintended cumulative token cap, and align the prompt with the inherited submit schema; and
7. synthetic and four-provider CAIS-HLE smoke tests, protected artifacts, and sanitized reports.

Those are harness choices. They are not behavior supplied by upstream AllenAI, AstaBench, Inspect, CAIS, or ActiveLoop.

### Local changes during this migration

The HLE-Verified migration then narrowly changed the local adapter to:

1. pin `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b` and retain all 15 protected Parquet files plus hashes;
2. accept a snapshot directory, require 2,500 unique source rows, parse each row's structured `json`, and cross-check its ID/class/question/answer against flattened columns;
3. select the card-recommended 668-row Gold evaluation pool and use `json.answer`, `json.answer_type`, and `json.image` without changing provider serialization;
4. rename the live data variable to `HLE_VERIFIED_DATA_PATH`, reject HLE-Verified dataset URLs in search, and leave old protected CAIS data untouched;
5. regenerate smoke and 50-row pilot manifests with new dataset-specific deterministic seeds, keeping protected IDs out of Git; and
6. add the macOS host-awake requirement because the incident evidence showed host suspension, while leaving Inspect's provider/sample timeout machinery intact.

## Direct implementation comparison

| Concern | ActiveLoop pinned implementation | AllenAI pinned implementation | This project |
|---|---|---|---|
| Orchestration | Custom `asyncio` runner; semaphore by `num_workers`; atomic JSON checkpoint after each completed question | Inspect task/solver chassis; general ReAct agent, no HLE task | Inspect task plus narrow HLE adapter; protected `.eval` logs and `eval-retry` |
| Model timeout | One OpenRouter `AsyncOpenAI` client, 600-second SDK timeout, two SDK retries; no outer per-question wall timeout | No timeout in `basic_agent`; delegates model/sample limits to Inspect and task | Inspect attempt 600 seconds, retry budget 1, request retry window 660 seconds, sample limit 1,800 seconds |
| Host suspension | No host-awake handling | No host-awake handling | Active protocol now requires `caffeinate -i` |
| Tool loop | Up to 15 iterations, loop detectors, then a possible extra tool-disabled final model call | ReAct until submission/completion/max steps | Exactly 15 model turns; turn 15 exposes only inherited `submit` |
| Search | DuckDuckGo plus page fetch; separate proprietary ActiveLoop scientific search | Task/config-provided Asta tools | Common Exa search and controlled fetch; HLE dataset/result URLs blocked; no scientific-search asymmetry |
| Code | Stateful in-process `exec` in a restricted globals dictionary; 60-second `wait_for` around an executor thread, which cannot forcibly terminate a running thread | Asta/Inspect tool supplied to the agent | Asta stateful Python inside a disposable no-network Docker sandbox |
| Submission | Final formatted free text; no submit tool | Inherited structured `submit` string and submission manager | Same inherited submit wrapper, with prompt alignment and focused schema test |
| Judging | Separate concurrent script; OpenAI client timeout 300 seconds, one SDK retry | Benchmark-specific scorer outside the generic agent | In-task pinned `o3-mini-2025-01-31` semantic judge after valid submission |
| Concurrency/resume | Semaphore; completed results atomically checkpointed and skipped on restart | Inspect concurrency and eval logs | `--limit 50`, concurrency controls, protected log, `inspect eval-retry`; lane sequencing remains operator-controlled |

ActiveLoop provides useful iteration and incremental-checkpoint precedents, but adopting its custom loop would weaken sandbox isolation, add a possible 16th generation, change search, and not solve host suspension. AllenAI remains the smallest credible chassis.

## HLE-Verified inspection

The pinned repository is public and ungated. Its card and repository metadata declare no license, so the protected snapshot must not be redistributed or treated as licensed merely because the original HLE code/data carried other terms.

The snapshot contains 15 Parquet files and 2,500 unique rows:

- Gold: 668
- Revision: 1,143
- Uncertain: 689

The flattened Parquet schema is `id`, verification-class/category/validity columns, `question`, `answer`, and `json`; all are stored as strings. The `json` string contains the full structured record, including final or revised `question`, `answer`, `answer_type`, `image`, originals when revised, rationale fields, and nested verification metadata.

The final evaluation answer is `json.answer`; `json.original_answer` preserves a superseded answer when present. The loader cross-checks flattened identity/question/answer/class fields against the JSON record and then uses the JSON record. It preserves the same base64/data-URI image path as the former HLE loader. Across all rows, 342 questions have images. The card recommends Gold for stability-sensitive or leaderboard evaluation, so the active pool is the 668-row Gold subset: 459 exact-match, 209 multiple-choice, and 93 multimodal.

Caveat: the pinned Gold files label all 668 answers valid, but their component metadata labels three problems and 24 rationales invalid despite the card's “fully validated” description. Rationale validity does not affect scoring; the three problem flags remain a dataset-quality caveat to disclose rather than silently reinterpret. Revision and Uncertain rows are excluded from this controlled condition.

## Migration and deterministic manifests

Protected data was downloaded at the exact dataset revision without deleting the former CAIS snapshot. The 15 per-file SHA-256 values are retained in protected `SHA256SUMS`; that checksum manifest hashes to `db74a9a50570e3a9362b59e69c6f6b8e91f05dd227808403f6125c2bd666b784`.

The loader now requires all 2,500 unique rows, validates each structured JSON record, selects exactly 668 Gold rows, and then applies a protected ID manifest. New deterministic seeds prevent accidental reuse of the former selection:

- Pilot seed: `hle-tools-v0-hle-verified-gold-pilot-50`
- Pilot: 50 rows, 34 exact-match, 16 multiple-choice, all eight categories, eight multimodal
- Pilot ID-sequence hash: `4b44e3d0923c97409a65d413fd899319c7aaafa4ecce0676d562467241d5198b`
- Protected pilot-file hash: `51fbde3cb0f46d7c3211e49e41b32774af562c7748e43513e2f64c77caef8a85`
- Smoke: one multimodal Gold row outside the pilot
- Smoke ID hash: `ab2e48debf76678f7104c473498b29df8662e5fd32ed39b8cc8a44235646cc51`
- Protected smoke-file hash: `3e9e49ef76f83de28281ccf971f3cd89e0928baea327d0cf74a7427ae154a3e3`

Former protected HLE data, manifests, smoke logs, and the aborted-pilot diagnostic remain untouched and excluded from new aggregates.

## Verification and GPT smoke status

Focused schema/selection and URL-block tests plus the complete offline HLE test suite pass: 17 tests passed. The real loader independently produced 668 unique Gold samples, the protected smoke resolved to one multimodal sample, and a clean regeneration reproduced both protected manifests byte for byte.

One authorized GPT smoke ran against the new multimodal Gold smoke manifest under protocol revision `2026-08-25-hle-verified-gold-host-awake` and code `1621538`:

- Inspect status: success
- Host-awake guard: `caffeinate -i`; the macOS power log records its sleep-prevention assertion for the full 3:17 command lifetime
- Wall time: 3:10
- Agent turns: 15
- Tools: 25 search, 3 fetch, 2 Python, 1 submit
- Structured submission: valid; confidence 0.18
- Judge: invoked successfully
- Correctness: incorrect on this one item
- Solver usage: 215,802 tokens (42,171 uncached input, 168,243 cached input, 5,388 output, including 4,285 reasoning)
- HTTP retries, Exa 429s, search/fetch transport errors, and unresolved judge errors: none
- Secret scan: clean; sandbox/container cleanup: clean

This successful bounded run is consistent with the host-suspension diagnosis and does not establish that OpenAI or Inspect had a timeout defect. It is plumbing evidence, not an accuracy estimate. The 50-question pilot and non-GPT lanes remain blocked; neither was launched during this migration.
