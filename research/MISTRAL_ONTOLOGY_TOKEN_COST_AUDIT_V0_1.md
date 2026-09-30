# Mistral tokenizer costs on the three frozen v0.4 ontologies

**Status:** preregistered deterministic cost replication. No model weights, model responses, or task-utility outcomes were used.

## Question and frozen inputs

The synthetic cross-tokenizer sweep used generic `axisNN` labels. This study asks whether its cost direction holds on the existing default, robotics, and music task vocabularies. The preregistration fixed the three bundle manifest hashes, the unchanged prompts and protocols, and predictions that both compact-fields and symbolic remain more costly than JSON.

- Default: seed 23, 64 test episodes.
- Robotics: seed 31, 64 test episodes.
- Music: seed 31, 64 test episodes.

The auditor checked each bundle's manifest hash and episode count against the earlier Qwen report before comparing it. The official Mistral tokenizer is pinned to revision `adadfb3fbae87ecc77cd5bf2c3318434d5da04cf` with SHA-256 `e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49`. No episode or ontology was regenerated. Measurements include the runner's sender and receiver prompt content, receiver-side message echo, and ideal sender output; chat templates, receiver output, provider accounting, model behavior, and inference compute are excluded.

The [frozen plan](PROTOCOL_TOKENIZER_ONTOLOGY_TRANSFER_PREREG_V0_1.md), [Mistral audit source](audit_mistral_ontology_token_costs.py), and complete [machine-readable report](data/MISTRAL_ONTOLOGY_TOKEN_COST_AUDIT_V0_1.json) bind the tokenizer, source code, protocol card, and bundle hashes.

## Results

Mean per-episode known content tokens; deltas are paired against JSON within each frozen episode set:

| Ontology | Episodes | Mistral fields − JSON | Qwen fields − JSON | Mistral symbolic − JSON | Qwen symbolic − JSON |
|---|---:|---:|---:|---:|---:|
| Default | 64 | +49.000 | +47.594 | +180.156 | +162.656 |
| Robotics | 64 | +49.000 | +46.938 | +191.969 | +171.562 |
| Music | 64 | +48.000 | +47.531 | +195.312 | +171.938 |

**P22a and P22b are supported.** Every Mistral episode/domain has a positive compact-fields delta (observed ranges: +48 to +49 tokens per episode) and a positive symbolic delta (domain means +180.156 to +195.312). The fixed per-axis digit maps remain particularly costly.

Relative to Qwen on the exact same bundles, the Mistral card-minus-JSON margin increases by only 0.469–2.062 tokens across the three ontology means. The symbolic-minus-JSON margin increases by 17.500–23.374 tokens. Thus these real task vocabularies reproduce the negative direction and show that tokenizer shifts are representation-dependent: small for the compact card, larger for the explicit symbol-map instructions.

## Interpretation and limits

Using the frozen attribute names rather than synthetic axis labels does not reverse the earlier token-cost result. The card stays roughly 48–49 tokens above JSON in each ontology under both tokenizer families. Explicit symbolic maps cost considerably more under Mistral and remain worse than JSON in all three domains. This strengthens the claim that the current protocol card and prompt construction are not content-token-efficient on these tasks.

This does not show that JSON is the best task-success protocol, that a card cannot improve fidelity, or that actual Mistral and Qwen models behave alike. It measures tokenizer output over serialized prompts only; chat templates, actual provider usage, receiver outputs, latency, quality, and inference cost are missing. These are the same episodes used in the earlier Qwen audit, so this is tokenizer replication, not an independent task sample.

## Reproduction

With the tokenizer and local role bundles available:

```powershell
python research/audit_mistral_ontology_token_costs.py `
  --tokenizer-json .cache/tokenizers/Mistral-7B-Instruct-v0.3-adadfb3f/tokenizer.json `
  --default-bundle .cache/compact_fields_dryrun_v1/episodes `
  --robotics-bundle .cache/compact_fields_ontology_dryruns/robotics `
  --music-bundle .cache/compact_fields_ontology_dryruns/music `
  --output research/data/MISTRAL_ONTOLOGY_TOKEN_COST_AUDIT_V0_1.json
```

The role bundles and Mistral tokenizer remain in ignored project `.cache`; only hashes and aggregate measurements are included in the committed JSON.
