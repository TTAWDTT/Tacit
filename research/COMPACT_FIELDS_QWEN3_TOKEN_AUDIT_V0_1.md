# Explicit-field and symbolic Qwen3 token audit v0.1

**Status:** tokenizer-only, model-free cost audit. It measures one pinned Qwen3 tokenizer on frozen v0.4 task bundles. It is not a model run, task-success result, or cross-model claim.

## Question

Do the v0.3 compact labeled-fields card's 18-byte payload reduction or the runner's four-digit symbolic message translate into fewer complete communication tokens than the built-in JSON arm, once sender/receiver instructions and the repeated receiver message are included?

## Frozen inputs and method

- Tokenizer: `Qwen/Qwen3-4B`, revision `eb971e9fb1f41c13b5e5a56e56886305c5ad94a0`, file SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Protocol card: v0.3, SHA-256 `2646cd19c10aa5f4bec19d8e38fdb43eac6c15238630b86f79c794bff1a46aa2`.
- The runner's fixed `symbolic` arm uses one digit per attribute and includes the full value-to-digit map in both endpoint instructions.
- Task bundles: 64 test episodes each for default (split seed 23), robotics (31), and music (31). The audit loads each through the runner's verified bundle loader and records each manifest hash in the machine-readable report.
- For each episode and arm, the analysis reconstructs the runner's exact sender system/user content, ideal canonical sender message, and receiver system/user content with that message inside the visible transcript. It tokenizes each content string with `add_special_tokens=False`, then sums both requests' input-content tokens and the sender's ideal message-output tokens. The receiver's output is omitted because its actual text is unknown; chat-template and server-added special tokens are excluded.

The script is [`audit_protocol_token_costs.py`](audit_protocol_token_costs.py). The complete aggregate is [`COMPACT_FIELDS_QWEN3_4B_TOKEN_AUDIT_V0_1.json`](data/COMPACT_FIELDS_QWEN3_4B_TOKEN_AUDIT_V0_1.json). The local tokenizer cache is ignored and contains only `tokenizer.json`; no model weights were downloaded or read.

To reproduce, install the optional `tokenizers` and `huggingface_hub` packages, use `hf_hub_download(repo_id="Qwen/Qwen3-4B", filename="tokenizer.json", revision="eb971e9fb1f41c13b5e5a56e56886305c5ad94a0", cache_dir=".cache/tokenizers/hfhub")`, then pass the returned path to the script with the three bundle directories and an output path. The bundle seed/keyed-generation workflow is documented in the [v0.4 runner guide](../experiments/emergent_ood_v0_4/README.md#keyed-communication-episodes). The script verifies tokenizer, card, bundle, and runner/codec hashes before saving results.

## Results

Mean exact content-token counts per episode; sender-output tokens assume perfect canonical serialization.

| Ontology | JSON known total | v0.3 known total | v0.3 minus JSON | Symbolic known total | Symbolic minus JSON |
|---|---:|---:|---:|---:|---:|
| Default | 422.703 | 470.297 | +47.594 | 585.359 | +162.656 |
| Robotics | 433.531 | 480.469 | +46.938 | 605.094 | +171.562 |
| Music | 446.219 | 493.750 | +47.531 | 618.156 | +171.938 |
| Equal-weight mean | 434.151 | 481.505 | **+47.354** | 602.870 | **+168.719** |

For v0.3, input content increases by 49.677 tokens while ideal sender output saves 2.323. For the symbolic arm, input content increases by 184.026 tokens while the four-digit output saves 15.307. Thus neither representation lowers known token use against JSON under this tokenizer and exact runner setup; the symbolic prompt's two explicit codebooks dominate its short message. Unknown receiver-answer tokens, chat-template tokens, service time, task success, and accuracy may change the quality/cost frontier; no complete superiority conclusion follows.

## Limits and decision

This is a tokenizer measurement, not provider-reported usage or inference compute. It applies to the pinned Qwen3 tokenizer and these exact runner prompts; it cannot establish the same margin for other tokenizers or providers. The receiver's serialized transcript is constructed exactly as the local runner does, but server-side chat templates were intentionally excluded. No model was loaded, no endpoint contacted, and no LLM output or task outcome was scored.

The compact card should remain a message-payload baseline, not be described as a token-efficient protocol on these conditions. A future frontier run could still find that its different success rate compensates for its additional cost; that requires the independently gated paired model experiment.
