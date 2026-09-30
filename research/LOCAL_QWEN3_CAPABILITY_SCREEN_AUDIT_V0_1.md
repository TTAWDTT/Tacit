# Local Qwen3 receiver-capability screen audit v0.1

**Status:** interface diagnostic only. The independent receiver capability gate did not pass, so no communication-condition comparison was run. All outputs and task ledgers remain in ignored local `.cache/` files; no evaluator labels or raw outputs were committed.

## Frozen task and resource boundary

The diagnostic used the existing v0.4 exact-selection task with `k=4`, `task_seed=0`, and a calibration split seed of 17, independent of evaluation split seed 23. The full-information control exposes the sender's tuple to the receiver; the runner requires every answer to be exact and format-valid across three balanced training candidate sets (12 episodes). This is a capability gate, not evidence about any communication language.

The initial read-only preflight passed at 2026-09-30 00:14:59 UTC: host CPU mean 16.54%, maximum 17.16%, GPU utilization 18%, 1,101 MiB GPU memory used, and 12,446 MiB free system memory. The experiment used a local Qwen3-4B Q4_K_M GGUF through the existing llama.cpp server, single slot, context 4096, four CPU threads, low process priority, and a hard 12-request runner batch. The server was stopped after each attempt.

## Observed interface outcomes

Three 12-request attempts were completed under separately passing preflights. Their local results are retained at:

- `capability-17-qwen3-4b.jsonl`: all 12 raw answers end in the exact gold candidate ID, but each begins with an empty `<think>...</think>` block. The strict output parser marks 0/12 format-valid and 0/12 exact-selection.
- `capability-17-qwen3-4b-deepseek.jsonl`: configuring `--reasoning-format deepseek` without disabling Qwen reasoning leaves `message.content` empty on all 12 responses; each hits the 48-token output cap. Format-valid: 0/12.
- `capability-17-qwen3-4b-reasoning-off.jsonl`: `--reasoning off --reasoning-format none` again emits the empty think block before the correct candidate ID. Format-valid: 0/12.

These records show a response-contract/configuration mismatch: this llama.cpp build's empty think tags are present in the content consumed by Tacit's strict parser. They do not establish failure to identify the target. Do not edit the scorer to strip arbitrary wrappers after seeing these answers; that would change the frozen answer contract. The next untested server configuration is `--reasoning off --reasoning-format deepseek`, which should put parsed reasoning in the separate API field and leave `message.content` for the final answer. Test it only after a fresh eligible preflight.

## Stop condition and next gate

The latest preflight at 2026-09-30 00:25:30 UTC was rejected because GPU utilization was 30%, above the frozen 25% limit. CPU mean was 12.83%, maximum 16.6%, GPU memory 1,102 MiB, and free RAM 12,662 MiB. No model artifact was read, no service was started, and no request was made during this rejected gate. At inspection time port 8000 had no listener and no llama-server process remained.

Therefore the full-information capability ledger is still invalid. Do not run no-message, natural-language, or other communication arms from this task until the reasoning-off/deepseek interface check passes 12/12 under a newly eligible preflight. If it still fails, investigate the endpoint adapter under a separately frozen format contract; do not loosen the scorer based on test outcomes. The three attempts cost 36 local receiver calls in total and are recorded as engineering diagnostics, not language-evaluation data.

## Upstream interface reconciliation (offline follow-up)

The official [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) now document two independent controls: `--reasoning off` changes the template's thinking mode, while `--reasoning-format deepseek` routes parsed thought text to `message.reasoning_content`; `none` leaves thought markers/content unparsed in `message.content`. The official [Qwen3 repository](https://github.com/QwenLM/Qwen3) documents `/no_think` as a prompt-level mode switch for the original hybrid Qwen3 models and separately describes `enable_thinking=False` template behavior. These sources explain why the `none` attempts can fail the exact answer parser even when the answer ID is correct, and why `deepseek` without a matching thinking-mode change is not a valid test of the desired response contract.

**Updated prediction, not a result:** after a fresh eligible resource preflight, retry the already-frozen 12-call full-information block using `--reasoning off --reasoning-format deepseek`. The predicted interface outcome is that any parsed reasoning is carried outside `message.content` and the strict candidate-ID parser sees only the final answer. This prediction is conditional on the locally installed llama.cpp build and its Qwen3 template; upstream documentation does not prove that this particular binary behaves accordingly. Keep the task, prompt, answer scorer, token cap, and endpoint fixed. Record the full server version/configuration and both API fields' *presence/type* (not private reasoning text); stop if content is still malformed or empty. The capability gate remains 12/12 and no communication arm becomes eligible until it passes.

This was a source/documentation audit only. No local binary, model, server, GPU, or endpoint was accessed; no inference was run.
