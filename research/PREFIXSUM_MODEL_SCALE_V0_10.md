# PrefixSum v0.10 local model-scale diagnostic

**Design:** Qwen3-8B Q4_K_M in the two preregistered oracle/model hybrid roles, compared with Qwen3-4B v0.9 on the same 12 tasks and compact-KV wire condition. The task set is reused, and the 4B values are the paired v0.9 results; this is not new held-out evidence or a language ranking.

Model revision `7c41481f57cb95916b40956ab2f0b139b296d974`, file SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4.

## Oracle sender + model receiver

| Metric | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| Agent 0 exact | 12/12 | 12/12 |
| Agent 1 exact given receive | 0/11 | 0/12 |
| Fully correct episodes | 0/12 | 0/12 |
| Correct subtotal messages | 12/12 | 12/12 |

## Model sender + oracle receiver

| Metric | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| Agent 0 exact | 3/12 | 4/12 |
| Agent 1 exact given receive | 0/12 | 1/12 |
| Fully correct episodes | 0/12 | 0/12 |
| Correct subtotal messages | 0/12 | 1/12 |

## Interpretation

At the receiver, Qwen3-8B returned 0 exact global segments after receiving the correct oracle subtotal in 12/12 episodes; 7/12 outputs were exactly the local prefix alone. Qwen3-4B was also 0 exact in its 11 received-message cases. This run shows no receiver-side gain from the tested local scale increase.

At the sender, Qwen3-8B produced a faithful subtotal in 1/12 and an exact local prefix list in 4/12, versus 0/12 and 3/12 for Qwen3-4B. The oracle receiver applied the received wire value to its exact local prefix in 11/12 cases. One correct wire subtotal did not coincide with a correct Agent 0 answer, so neither hybrid arm produced a fully correct episode.

Compare role-specific accuracy and message fidelity with the preregistered prediction; do not interpret token or latency differences as protocol efficiency. A model-size comparison also changes the checkpoint and its training, so it cannot isolate parameter count alone. One greedy draw on 12 reused cases supports only a local diagnostic.

Per-cell summaries and exact wire messages are available as [JSONL run rows](data/PREFIXSUM_MODEL_SCALE_V0_10_RUNS.jsonl). Full local tool/model traces: `.cache/pilot_v0_10/scale_hybrid_runs_20260927T104604Z.jsonl`. Aggregate data: [JSON](PREFIXSUM_MODEL_SCALE_V0_10.json).
