# PrefixSum hybrid role diagnostic v0.9

## Question and scope

The v0.8 model pilot had no fully correct episodes, although v0.8's deterministic oracle passed all 12 tasks through the same engine and scorer. The hybrid diagnostic localizes the model-side bottleneck without changing task data or message representation:

1. **Oracle sender + Qwen3-4B receiver:** tests whether the model can receive a correct `s=<subtotal>` message, decode it, apply the offset, and return its exact prefix segment.
2. **Qwen3-4B sender + oracle receiver:** tests whether the model calculates and emits the correct subtotal, while the receiver deterministically decodes the wire value and computes the exact output.

This uses the same 12 held-out v0.8 tasks in both paired arms, the frozen compact-KV condition, the same role prompts, model weights, deterministic decoding, message adapter, engine, and scorer. The same task seeds make this a diagnostic replication, not new held-out evidence. Oracle outputs are not model results. Do not treat this as a protocol comparison, generalization estimate, or efficiency frontier.

## Frozen predictions

- With the oracle sender, Agent 0 should be exact and send the correct subtotal in all 12 tasks by construction. Agent 1's exact score directly measures Qwen receiver execution after actual message delivery.
- With the oracle receiver, Agent 1 should be exact whenever it receives a parseable, correct subtotal. Agent 0's exact score and message syntax/value fidelity measure the Qwen sender side.
- If either model role remains weak under the opposite role's oracle, the corresponding capability is a primary bottleneck in v0.8. If both hybrid roles work but the original pair fails, investigate interaction and turn scheduling.

## Metrics

Per arm, report Agent 0 and Agent 1 exact outputs separately, fully correct episodes, delivered/received messages, compact-KV syntax and value fidelity, backend tokens, wall time, and rounds. Also report conditional exactness: Agent 1 exact when the correct subtotal arrived, and Agent 1 exact when any subtotal arrived. Do not compare per-arm total model tokens as if they were equal-budget protocol alternatives: only one role uses the model in each arm, and prompt histories differ by role.

## Run

Start the pinned server as documented in [v0.8](../pilot_v0_8/README.md), then run:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
$env:TLU_GGUF_PATH = '.cache/models/Qwen3-4B-Q4_K_M.gguf'
python experiments/pilot_v0_9/run_hybrid_diagnostic.py
```

The runner writes raw traces under ignored `.cache/pilot_v0_9/` and writes its aggregate only after all 24 cells complete. The preregistration is committed before model calls.
