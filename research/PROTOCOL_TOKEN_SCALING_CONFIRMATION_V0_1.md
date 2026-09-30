# Dense protocol token crossover confirmation v0.1

**Status:** preregistered deterministic confirmation of a tokenizer-cost pattern; no model-performance or statistical-inference claim.

## Question and frozen predictions

The exploratory extension found compact-fields v0.3 minus JSON known-content-token differences of `+7`, `−33`, and `−113` at `d=20,40,80`, with the same difference at each tested cardinality `V`. Before measuring intermediate dimensions, [P20a–P20c were frozen](PROTOCOL_TOKEN_SCALING_CONFIRMATION_PREREG_V0_1.md): replicate the endpoint signs; locate the first non-positive integer difference in `{23,24,25}` for every cardinality; and verify the fixed-symbol cost remains positive and increases with `V`.

## Method

- Full grid: every integer `d=20…40` crossed with `V∈{2,4,8,10}` (84 settings), 16 deterministic episode-shaped records per setting, and four receiver candidates per record.
- Same modular identity-rank support fixture, v0.4 runner prompt reconstruction, JSON/v0.3/fixed-symbol conditions, and pinned Qwen3-4B tokenizer revision as the preceding sweeps.
- Primary outcome: mean known content tokens per episode, including both agent inputs, the delivered message echoed in the receiver transcript, and ideal sender output. Receiver output, model execution, endpoint templates, provider caching, billing, latency, and compute are excluded.
- The JSON artifact binds this preregistration, both preceding sweep artifacts, the tokenizer, v0.4 runner, codec, and protocol card. All 84 settings and component summaries are retained in [`PROTOCOL_TOKEN_SCALING_CONFIRMATION_V0_1.json`](data/PROTOCOL_TOKEN_SCALING_CONFIRMATION_V0_1.json).

## Results against preregistered predictions

**P20a — endpoint bracket: supported.** At every `V`, the difference is `+7` at `d=20` and `−33` at `d=40`.

**P20b — dense crossover location: supported.** At all four cardinalities, `d=23` gives `+1` token and `d=24` gives `−1`; therefore the first non-positive integer dimension is exactly `d=24` in this grid.

**P20c — fixed-symbol cost: supported.** It is more expensive than JSON in all 84 settings, and its excess increases strictly with `V` at every fixed dimension. At `d=20`, the excesses for `V=2,4,8,10` are `663, 1,383, 2,823, 3,543`; at `d=40` they are `1,303, 2,743, 5,623, 7,063`.

Across the entire grid, the compact-fields minus JSON difference is exactly `47−2d` tokens per constructed episode and is invariant to `V`. This identity is an observed property of this pinned tokenizer and these serialized synthetic prompts; it is not a fitted statistical law or a general scaling theorem.

## Interpretation and limits

The new intermediate measurements confirm and sharpen the earlier exploratory bracket: under the exact v0.4 content construction and Qwen3-4B tokenizer, the v0.3 card's recurring known-content-token total becomes lower than JSON starting at 24 dimensions. The result quantifies when this card's per-field serialization savings overtake its repeated prompt premium in this synthetic construction. The parallel cardinality-invariant difference shows that, in this setup, increasing the label inventory changes the fixed-symbol codebook cost but not the measured compact-fields/JSON gap.

This is not an LLM communication-success result. The fixtures are synthetic, every setting has only four candidates, and there was no model execution. It says nothing about semantic fidelity, ambiguity, error recovery, utility, equal-quality efficiency, actual provider billing, latency, cross-model transfer, or whether 24-field tasks are practically useful. `d=40` contexts may be near deployment limits after model outputs, templates, and retained history are included. P20a–P20c concern deterministic accounting and provide no inferential uncertainty estimate.

The next meaningful test is not to extrapolate this identity. A task-level protocol comparison needs useful, communication-dependent tasks, eligible receivers, matched quality/cost frontiers, and the existing resource and capability gates. Until then, retain this only as a tokenizer-specific cost-geometry result.

## Reproduction

With the pinned tokenizer already present locally:

```powershell
python research/audit_protocol_token_scaling.py `
  --tokenizer-json .cache/tokenizers/hfhub/models--Qwen--Qwen3-4B/snapshots/eb971e9fb1f41c13b5e5a56e56886305c5ad94a0/tokenizer.json `
  --output research/data/PROTOCOL_TOKEN_SCALING_CONFIRMATION_V0_1.json `
  --confirmation
```

This reads the tokenizer and tokenizes constructed text only. It loads no model weights and makes no inference calls.
