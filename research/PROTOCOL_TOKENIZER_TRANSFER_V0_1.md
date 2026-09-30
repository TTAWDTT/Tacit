# Cross-tokenizer protocol cost transfer v0.1

**Status:** preregistered deterministic tokenizer comparison. It measures serialized content cost; it is not a model-performance comparison.

## Frozen question and source

The preceding Qwen3-4B sweep located the compact-fields/JSON known-content-token crossover at `d=24`. Before measuring another vocabulary, the project froze P21a–P21c in the [cross-tokenizer preregistration](PROTOCOL_TOKENIZER_TRANSFER_PREREG_V0_1.md).

The second tokenizer is `tokenizer.json` from [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/adadfb3fbae87ecc77cd5bf2c3318434d5da04cf/tokenizer.json), pinned at revision `adadfb3fbae87ecc77cd5bf2c3318434d5da04cf`. The official repository lists a 1.96 MB tokenizer file and Apache-2.0 licensing. Only this tokenizer file was downloaded; no model weights were requested or loaded.

## Method

- Dimensions `d∈{4,10}∪{20,…,40}` and `V∈{2,4,8,10}`: 92 settings, 16 deterministic prompt fixtures per setting, and four receiver candidates per fixture.
- The JSON, compact labeled-fields v0.3, and fixed digit-symbol conditions use the same exact v0.4 prompt constructor, candidate fixtures, protocol card, and message construction as the Qwen runs. All three arms are tokenized from content strings with `add_special_tokens=False`.
- “Known total” includes both agents' input content, the delivered message echoed in the receiver input, and ideal sender output. Chat templates, receiver output, provider token usage, caching, model compute, latency, billing, and task outcomes are excluded.
- The 92-setting JSON binds the preregistration, Qwen confirmation data, Mistral tokenizer bytes, script, card, codec, and runner hashes. See [`PROTOCOL_TOKENIZER_TRANSFER_V0_1.json`](data/PROTOCOL_TOKENIZER_TRANSFER_V0_1.json).

## Results against the preregistration

**P21a — endpoint bracket: supported.** For each `V`, compact-fields minus JSON is `+16` at `d=20` and `−24` at `d=40` under Mistral tokenization.

**P21b — Qwen crossover-location transfer: falsified.** The first non-positive Mistral difference is at `d=28`, outside the frozen predicted set `{23,24,25}`. At `d=24`, the Mistral card still costs 8 more tokens than JSON; at `d=28` it ties, and at `d=29` it becomes cheaper. Qwen crossed earlier, at `d=24`.

**P21c — fixed-symbol burden: supported.** Fixed symbols cost more than JSON in all 92 settings, and their excess rises strictly with `V` at every tested dimension.

For every shared `(d,V)` setting, the Mistral compact-fields/JSON difference is exactly 9 tokens higher than the Qwen difference. In the dense region, the two measured functions are:

| Tokenizer | `C_card − C_JSON` for `20≤d≤40` | First non-positive integer `d` |
|---|---:|---:|
| Qwen3-4B | `47−2d` | 24 |
| Mistral-7B-Instruct-v0.3 | `56−2d` | 28 |

Both functions are invariant to `V` in this fixture grid. The constant 9-token offset is also observed at `d=4` and `d=10`. This is an identity over the declared deterministic inputs, not a fitted law or a claim about other models.

The component ledger shows why equal slopes do not imply equal crossover locations. At `d=24,V=2`, the Qwen card has a 50-token instruction premium over JSON, a 25-token reduction in receiver input context, and a 26-token sender-output reduction, for a net `−1`. The Mistral instruction premium is 60 tokens, the receiver-context reduction is 50 tokens, and sender-output reduction is only 2 tokens, for a net `+8`. Counting the receiver echo is material to this total-cost comparison.

## Interpretation and limits

The cost crossover direction transfers from Qwen to a second tokenizer family in this narrow synthetic setting, but the exact location does not. The Mistral card's larger instruction premium delays its equal-known-content-cost point by four dimensions. The component ledger also shows that the same aggregate slope can arise from different allocations between repeated receiver context and generated sender output. Therefore a statement such as “the format saves two tokens per field” is incomplete unless it names the tokenizer and counts both model inputs and outputs at the real request boundary.

This comparison does **not** run either model or measure whether either receiver understands the format. It does not include model-specific chat templates or provider accounting and does not demonstrate cross-model protocol transfer, equal-quality efficiency, billing savings, or a useful high-dimensional task. The task fixtures are synthetic with four candidates, and `d=20…40` may be impractical under full deployment prompts. The failed P21b prediction narrows the claim: the Qwen crossover location is tokenizer-dependent, even though the direction and slope match here.

## Reproduction

With the pinned Mistral tokenizer already in project-local `.cache`:

```powershell
python research/audit_protocol_tokenizer_transfer.py `
  --tokenizer-json .cache/tokenizers/Mistral-7B-Instruct-v0.3-adadfb3f/tokenizer.json `
  --output research/data/PROTOCOL_TOKENIZER_TRANSFER_V0_1.json
```

The script tokenizes constructed text only. It loads no model weights and makes no inference calls.
