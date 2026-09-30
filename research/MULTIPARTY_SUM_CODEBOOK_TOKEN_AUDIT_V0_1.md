# Qwen3-4B tokenizer audit of finite private-sum codebooks (v0.1)

## Result

The frozen explicit codebook is not token-efficient on its exact zero-error operating points. Across all 17 nonempty codebook points in the exhaustive `m=2..4` frontier, the mean role-instruction input alone is 312 to 3,524 tokens larger than the protocol's ideal payload-bit count. Across exhaustive IID source vectors, total known content tokens range from 520 to 4,068 per episode for codebooks, against 381–635 for the existing decimal sender condition. At zero-error success, codebook totals are 839 tokens (`m=2`), 1,558 (`m=3`), and 4,068 (`m=4`); decimal costs are 381, 508, and 635 respectively.

The codebook used fewer total known tokens than decimal at only one of 17 points: `m=4`, budget at most 1 bit, where codebook exact-sum success is 18.75% and decimal is 100%. That lossy point also schedules fewer model calls (2 versus 5), so it is not a matched-success comparison and cannot establish a communication-language advantage. The codebook is cheaper than the JSON baseline at two points and binary text at three, but those comparisons also mix different success rates and, where senders are omitted, different call counts.

This falsifies the preregistered tokenizer-level prediction that ideal payload bits would approximate role-instruction cost by scale: the complete model-facing encoder and decoder descriptions dominate the bit budget at every tested point. It does not falsify the exact bit-channel frontier. The quantities use different cost units and boundaries.

## Method

The preregistered [P23 audit](MULTIPARTY_SUM_CODEBOOK_TOKEN_AUDIT_PREREG_V0_1.md) pins `Qwen/Qwen3-4B` tokenizer revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0` (SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`). `research/audit_multiparty_codebook_token_costs.py` reconstructs the system and user content used by the SDK, including receiver-visible message echo, and enumerates every private vector for `m=2,3,4` at each nonempty frontier budget. It compares the frozen partition codebook with the existing decimal, JSON, and two-character binary-text instructions on identical vectors. For each codebook transcript, the receiver output is the frozen MAP answer. The machine-readable [`17-point report`](data/MULTIPARTY_SUM_CODEBOOK_TOKEN_AUDIT_V0_1.json) contains per-role input, completion output, exact success, call counts, card tokens as a separate distribution diagnostic, artifact hashes, and condition summaries.

The count excludes chat templates, provider-added special tokens, hidden reasoning, actual model errors, service latency, inference compute, and provider pricing. A count of serialized-card tokens is not included in per-episode totals: the runtime sends role instructions on model requests rather than sending the JSON card blob. Conversely, card JSON count describes a possible distribution representation, not an inference prompt. These boundaries are intentionally separate.

The deterministic audit took about 16 seconds with the cached tokenizer. It loaded no model weights, started no service, and made no inference requests. Rerun from the project root with:

```powershell
python research/audit_multiparty_codebook_token_costs.py `
  --output research/data/MULTIPARTY_SUM_CODEBOOK_TOKEN_AUDIT_V0_1.json --force
```

## Interpretation and next question

This is evidence that a mathematically compact code can be an expensive natural-language protocol for a model to use: codebook interpretation has to be specified and repeated in prompts. On this task, explicit table descriptions swamp the few payload bits they save. It supports a concrete design constraint for further work: reusable semantics need amortization, model-native learned tokens/decoders, or a sufficiently demanding task to repay their instruction cost. That is a hypothesis to test, not a result here.

At lossy budgets, the codebook also changes the number of active sender calls. Any next comparison must expose the full success-cost frontier, include the no-message prior baseline at the same task size, keep task tuples paired, and report call/input/output/channel costs separately. A model-backed comparison remains closed until the independent capability and fresh resource gates pass. These tokenizer calculations do not test whether Qwen follows the instructions and cannot rank semantic reliability.
