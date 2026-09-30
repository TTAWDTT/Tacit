# Corrected exact-runner cost–success frontier for private sum (v0.2)

## Why this revision exists

The v0.1 tokenizer audit used format-specific sender/receiver text but omitted the shared receiver privacy, missing-message, and Bayes-prior instructions actually present in `examples/multiparty_private_sum.py`. This undercounted the operational baseline prompts. The [P24 preregistration](MULTIPARTY_SUM_CODEBOOK_FRONTIER_AUDIT_PREREG_V0_2.md) freezes the correction and adds the no-message control. The previous v0.1 comparison remains in the archive but is superseded for cross-condition cost ranking.

## Main result

The exact-runner accounting confirms that lower information-theoretic payload does not imply lower complete prompt cost. At the zero-error codebook points, codebook versus decimal known-content tokens are 695 vs 510 for `m=2` (both 100% exact, 248 vs 250 application bytes, 3 calls), 1,558 vs 637 for `m=3` (both 100%, 375 vs 375 bytes, 4 calls), and 4,068 vs 764.0586 for `m=4` (both 100%, 500 vs 500 bytes, 5 calls). Decimal uses far fewer content tokens at equal success and call count; application bytes tie for `m=3,4`, while the codebook saves only 2 bytes at `m=2`. Thus the full codebook point is dominated on all reported axes at `m=3,4`, but trades a 2-byte reduction for much higher token cost at `m=2`.

The no-message reference costs 257 known content tokens, zero transmission bytes, and one receiver call at all three sizes. Its exact Bayes success is 25% for `m=2`, 18.75% for `m=3`, and 17.1875% for `m=4`. Lossy codebooks move above that no-message accuracy, at additional cost. For example, at `m=4`, a one-bit-budget codebook reaches 18.75% exact success for 538 content tokens, 124 application bytes, and two calls. That is a +1.5625 percentage-point oracle gain over no-message at substantially higher cost; actual model adherence may erase the gain.

All reported communication arms (decimal, JSON, binary text, and codebook) are ideal-output references: senders are assumed to emit the prescribed faithful messages and receivers return the specified oracle answer. They define an operational tokenizer/transport frontier for this finite task, not a measured model frontier. The full 17-row report also keeps success, model-input tokens, completion tokens, application bytes, call count, card distribution cost, and within-`m` Pareto flags separate.

## Reproduction and verification

`research/audit_multiparty_codebook_frontier_v0_2.py` pins the existing Qwen3-4B tokenizer and reconstructs the exact runner prompts, JSON payloads, message echo, final-answer instructions, and protocol IDs. The baseline role prompts are supplied by `_role_instructions`, a shared constructor called by both the runtime example and the audit; a regression test compares those strings to the runtime-observed system prompts. It exhaustively evaluates `4^m` vectors for `m=2,3,4`; SDK byte framing comes from the pure `LocalTCPMessageChannel.measure` method and opens no socket.

```powershell
python research/audit_multiparty_codebook_frontier_v0_2.py `
  --output research/data/MULTIPARTY_SUM_CODEBOOK_FRONTIER_AUDIT_V0_2.json
```

The audit takes roughly 20 seconds with the project-cached tokenizer. It loads no model weights, starts no service, and sends no inference requests. The output is bound by source/preregistration/tokenizer SHA-256 values. The 11 private-sum example tests and 8 bundle/codebook runner tests pass; all source hashes match, and no-message's exact success and every codebook's measured exhaustive success match the formulas and committed finite frontier.

## Limits and next experimental implication

The no-message instruction is the actual existing runner scaffold, not a shortest custom prompt. This prevents v0.2 from claiming prompt optimality. Counts exclude chat templates, provider-added special tokens, hidden reasoning, real model errors, latency, compute, and billing. App bytes cover the measured loopback application envelope/length prefix/acknowledgment, not network headers. Decimal remains an unusually strong task-specific code because each private value is one digit; this task cannot validate a reusable general-purpose language.

For any later LLM experiment, pair identical tuples and include no-message, full-information capability, and the strongest operational formats. Report actual model input/output usage and measured wire bytes; compare the success-cost frontier, not only the ideal payload-bit frontier. Local inference remains subject to a fresh passing resource gate and independent receiver-capability gate. The latest recorded preflight is rejected at 60% GPU utilization, so this audit does not authorize inference.
