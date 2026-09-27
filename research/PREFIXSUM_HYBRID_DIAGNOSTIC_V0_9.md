# PrefixSum v0.9 hybrid role diagnostic

**Design:** paired diagnostic on the 12 v0.8 task seeds, using one greedy Qwen3-4B Q4_K_M run per task and hybrid arm. The message condition remains compact-KV (`s=<sum>`). The preregistration was committed before these model calls. This is not new held-out evidence or a language ranking.

Pinned model checksum: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`. Task manifest checksum: `f6e22454ab1a56b9606f9d668ce74828e80588ca97b5fc9ec8c75ca54341bc79`. Runtime: llama.cpp b11202, commit fcb3074f2, Windows x64 CUDA 12.4.

## Results

| Hybrid arm | Agent 0 exact | Agent 1 exact | Joint exact episodes | A1 received before submit | Compact-KV syntax | Correct subtotal | A1 exactly followed wire offset | Mean backend tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle sender + Qwen receiver | 12/12 | 0/12 | 0/12 | 11/12 | 12/12 | 12/12 | 0/12 | 5121 |
| Qwen sender + oracle receiver | 3/12 | 0/12 | 0/12 | 12/12 | 12/12 | 0/12 | 12/12 | 2507 |

The oracle sender supplied a correct subtotal in every episode and submitted Agent 0's exact segment in 12/12. Qwen Agent 1 received the non-empty payload before submitting in 11/12, but returned the exact global segment in 0/11 of those received-message cases. Its outputs were exact local-only prefix sums in 4/12, other incorrect arrays in 7/12, and missing in 1/12. This isolates a receiver-side failure after correct information was available; it does not prove the model never parsed any message token.

With Qwen as sender, all 12 messages had compact-KV syntax and were delivered, but none contained Agent 0's true subtotal. Agent 0's local prefix array was exact in 3/12. The deterministic receiver's Agent 1 outputs exactly matched the received wire subtotal plus local prefixes in 12/12, confirming the receiver control used the sent value; the joint task still failed in all 12 because the sender information and/or Agent 0 output was wrong.

## Interpretation and limits

The paired controls support two independent failure modes in this setup: Qwen3-4B often computes or reports the wrong subtotal as sender, and it does not reliably apply even a correct subtotal as receiver. The sender and receiver bottlenecks were not separable from v0.8 alone; this hybrid makes their presence visible. The benchmark therefore remains a model/tool-following diagnostic, not evidence for or against any general communication language.

This reuses the same 12 cases as v0.8, has one greedy run per arm, and changes which role invokes the model. Backend-token means are descriptive only: the model handles different role prompts and histories, and the oracle role has no model token cost. No efficiency comparison is valid. See the frozen [preregistration](../experiments/pilot_v0_9/preregistration.json), [runner](../experiments/pilot_v0_9/run_hybrid_diagnostic.py), and [analyzer](../experiments/pilot_v0_9/analyze_hybrid_diagnostic.py).

Raw local traces are under ignored `.cache/pilot_v0_9/hybrid_runs_20260927T100450Z.jsonl`. Aggregate data are also preserved in [JSON](PREFIXSUM_HYBRID_DIAGNOSTIC_V0_9.json).
