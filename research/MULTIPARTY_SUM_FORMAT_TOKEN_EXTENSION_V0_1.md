# Complete shipped private-sum format cost extension (P26)

## Result

P26 adds the shipped `labeled` and fixed-sentence formats missing from P24, using the exact shared role prompt constructor and all 336 IID source vectors across `m=2,3,4`. At each sender count, decimal has the lowest complete content-token count among the five communicating formats already shipped:

| Senders | Decimal | JSON | `v=N` | Binary text | Fixed English sentence |
|---:|---:|---:|---:|---:|---:|
| 2 | 510 | 517 | 515 | 580 | 550 |
| 3 | 637 | 650 | 645 | 736 | 695 |
| 4 | 764.0586 | 783.0586 | 775.0586 | 892.0586 | 840.0586 |

All five are idealized at 100% exact sum success. The compact labeled form costs 5, 8, and 11 more content tokens than decimal as `m` grows; the fixed English sentence costs 40, 58, and 76 more. Decimal's measured loopback application bytes are 250, 375, and 500; labeled costs 254, 381, and 508; the sentence costs 298, 447, and 596. These costs include the exact common receiver scaffold, role contexts, receiver message echo, and completion text, as well as each protocol's own encoder/decoder instructions.

This establishes the strongest *currently shipped* operational representation on this tiny sum task under these deterministic cost measures: decimal beats the other fixed formats. It is not an optimized natural-language search result, a general language claim, or a model-backed ranking. The scalar task makes decimal an unusually strong task-specific representation.

## Method and reproduction

The [P26 preregistration](MULTIPARTY_SUM_FORMAT_TOKEN_EXTENSION_PREREG_V0_1.md) freezes labeled and sentence output templates, the pinned Qwen3-4B tokenizer, source vectors, and measurement boundaries. The [auditor](audit_multiparty_sum_format_token_extension_v0_1.py) uses the `_role_instructions` constructor shared with the existing runtime example, computes SDK envelope bytes without opening a socket, and joins the results with the hash-bound P24 no-message, decimal, JSON, binary, and zero-error-codebook references.

```powershell
python research/audit_multiparty_sum_format_token_extension_v0_1.py `
  --output research/data/MULTIPARTY_SUM_FORMAT_TOKEN_EXTENSION_V0_1.json
```

The [`machine-readable report`](data/MULTIPARTY_SUM_FORMAT_TOKEN_EXTENSION_V0_1.json) includes per-role and aggregate content tokens, output tokens, exact success, calls, application bytes, cross-baseline token deltas, and source/tokenizer hashes. No model, endpoint, GPU, or service was used.

## Limits and next step

The fixed English sentence is one pre-existing template, not an optimized English baseline. Both natural-language search and an evaluation budget remain absent. Token counts exclude chat templates, provider tokens, model errors, latency, compute, and billing. No-message remains cheaper than every communicating arm but has only 25%, 18.75%, and 17.1875% Bayes exact success for `m=2,3,4`; compare full cost-success frontiers rather than calling it a winner. A meaningful continuation needs a development-only optimized-English procedure with a frozen proposal/query budget, untouched validation/test source tuples, and real model adherence checks behind the capability/resource gates.
