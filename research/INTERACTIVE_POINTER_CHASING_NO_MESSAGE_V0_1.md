# Exact no-message prior audit for pointer chasing v0.1

**Status:** exhaustive, model-free finite-prior diagnostic; not an LLM result and not a protocol-complexity experiment.

The v0.1 task requires both agents to submit the parity of the final pointer, with their final answers sealed from each other and all seed metadata withheld from their prompts. To ensure the no-message arm is not mislabeled as 50% by assumption, the artifact enumerates every pair of local functions for `n=2` and `n=4` under the target's full uniform function prior. This covers 16 pairs per depth for `n=2` and 65,536 pairs per depth for `n=4`. This is an exact calculation on the ideal distribution; the public task shard is a deterministic SHA-256 pseudorandom instantiation and is not claimed to be IID.

For each agent independently, the diagnostic computes its exact Bayes-optimal local classifier from its private map. Since any no-message protocol's joint success requires each participant to be correct, `min(Bayes_A, Bayes_B)` is a rigorous upper bound on joint no-message accuracy, including randomized local strategies and shared randomness independent of inputs. The report also includes two attainable joint controls: both agents use their independently optimal local MAP classifier (ties break to 0), or both output the same best public constant. These are feasible lower bounds, not a search for the globally optimal pair of local decision rules.

| n | k | Answer-1 prior | Agent A Bayes | Agent B Bayes | Joint local-MAP | Best shared constant | Joint no-message upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 2 | 2 | 0.5000 | 0.5000 | 0.7500 | 0.5000 | 0.5000 | 0.5000 |
| 2 | 3 | 0.5000 | 0.7500 | 0.6250 | 0.5000 | 0.5000 | 0.6250 |
| 2 | 4 | 0.6250 | 0.6250 | 0.8750 | 0.5000 | 0.6250 | 0.6250 |
| 4 | 1 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 4 | 2 | 0.5000 | 0.5000 | 0.6875 | 0.4375 | 0.5000 | 0.5000 |
| 4 | 3 | 0.5000 | 0.6875 | 0.5762 | 0.4250 | 0.5000 | 0.5762 |
| 4 | 4 | 0.5938 | 0.6055 | 0.7344 | 0.4346 | 0.5938 | 0.6055 |

## Findings and limits

- At `n=4,k=2`, the best public constant attains 0.5 and the exact individual-Bayes upper bound is also 0.5. Thus the optimal joint no-message accuracy is exactly 0.5 on this finite stratum, while the deterministic four-message oracle relay reaches 1.0.
- At `n=4,k=3`, the no-message joint accuracy is bounded above by 0.5762. This is an informative control band, but it is not an exact optimum; the two explicit feasible strategies score 0.4250 and 0.5.
- The answer prior is not guaranteed balanced merely because `n` is even: at `n=4,k=4`, answer 1 occurs on 59.375% of inputs, making the public-constant baseline materially better than 50%.
- `k=1` exposes why both parties must submit: Agent A alone knows the answer perfectly, but the joint no-message upper bound remains 50% because Agent B has no such information.

These small sizes verify evaluator semantics and reveal prior shortcuts; they cannot substantiate the asymptotic pointer-chasing theorem or predict an LLM's behavior at larger sizes. Any later model study must still measure no-message outcomes on its held-out task shard and pass the separate full-information gate.

The row-level exact counts are in [`POINTER_CHASING_NO_MESSAGE_DIAGNOSTIC_V0_1.jsonl`](data/POINTER_CHASING_NO_MESSAGE_DIAGNOSTIC_V0_1.jsonl). Reproduce them with:

```powershell
python experiments/pointer_chasing_v0_1/analyze_no_message.py `
  --sizes 2 4 --depths 1 2 3 4 `
  --output research/data/POINTER_CHASING_NO_MESSAGE_DIAGNOSTIC_V0_1.jsonl
```
