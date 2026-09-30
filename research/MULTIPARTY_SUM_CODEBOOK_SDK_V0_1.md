# Executable finite sum codebook card (v0.1)

## Question

Can the exhaustive model-free fixed-width private-sum frontier be represented as an actual, reusable role protocol in Tacit's multi-agent runtime, with measured setup and transport boundaries and an end-to-end scorer?

This is an implementation and accounting milestone. It does not test whether an LLM can learn, follow, or transfer the codebook.

## Frozen task and oracle

There are `m ∈ {2,3,4}` senders. Sender inputs are independent uniform values in `{0,1,2,3}`; sender `i` sees only its value; the referee must output the exact sum. A sender code is a partition of the four-value alphabet. Each cell is assigned a fixed-width ID, costing `ceil(log2(number_of_cells))` bits. The receiver's exact MAP output is enumerated for every joint ID tuple under the product-uniform prior. The selected partition tuple maximizes exact-sum success under the stated aggregate upper bound on payload bits; ties use the existing deterministic frontier rule. See the frozen [finite frontier artifact](data/MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.json) and [frontier derivation](MULTIPARTY_SUM_LOSSY_FRONTIER_V0_1.md).

The protocol card freezes those sender partitions and the full decoder. A one-cell sender emits a constant ID and is therefore omitted from calls; its unknown private value remains in the receiver's prior. If every sender is omitted, the codebook runner refuses the point and the separate no-message arm is the applicable control. Unused fixed-width codewords are invalid and receive no assumed source cell in the transcript oracle.

## Accounting and runner

`experiments/multiparty_sum_v0_1/codebook_protocol.py` creates a versioned SDK `DialogueProtocolCard`, deterministic protocol ID, complete sender/receiver instructions, and SHA-256 of canonical serialized card bytes. It reports:

- ideal fixed-width payload bits from the exact frontier;
- serialized-card UTF-8 bytes;
- UTF-8 bytes per role instruction and the sum of role instruction bytes supplied across one episode's model calls;
- serialized-card bytes amortized over 1, 10, 100, and 1000 episodes.

The serialized card is distribution/storage size. Per-role instruction bytes are repeated in model requests on every episode. They are separate measures, not additive duplicates. Neither is a token count: actual tokenization depends on each model's tokenizer and chat template. SDK wire bytes include the runtime's application envelope, length prefix, and acknowledgment, while excluding network headers and model inference.

`codebook_runner.py` validates the selected HMAC-keyed bundle rows and gold sums before making any request, requires the exact role/client set for the frozen card, and caps the batch at 12 planned model calls. Its CLI requires the same fresh local-resource preflight used by other model runners and checks configured loopback endpoints plus ports 8000, 8001, and 8002. It records bundle/card hashes, per-episode SDK diagnostics, model IDs, endpoint settings, and preflight hash. No local model, service, or endpoint was used in this milestone.

## Offline verification

`python -m unittest tests.test_multiparty_codebook_protocol -v` passed five tests. They round-trip protocol cards through the SDK schema across the enumerable `(m,budget)` settings with at least one active sender, check cell coverage and bit-budget accounting, execute a full-information fake-client exchange through the real SDK loopback transport, check private-context isolation and malformed-message decomposition, validate bundle batching and reject an over-cap batch before any fake call. `python -m experiments.multiparty_sum_v0_1.codebook_runner --help` displays the CLI without loading a model or contacting an endpoint.

These tests establish serialization, scheduling, task-boundary, and scoring mechanics only. Fake senders are scripted from the known source value and frozen partition; the fake receiver implements the oracle rule. They cannot establish instruction-following, realized task success, model-token cost, latency, robustness, cross-model transfer, or language superiority.

## Predictions and next falsification step

1. Given valid codewords, the receiver's oracle prediction is the exact MAP answer from the frozen frontier table for every delivered transcript. A counterexample would falsify the card/decoder implementation.
2. If an LLM follows this card perfectly, its exact-sum success should equal the exact frontier's success on an IID source; departures measure sender/receiver model errors, not a change in the coding theorem.
3. At matched measured end-to-end cost, this explicit codebook may lose to an already token-efficient structured representation because instructions and repeated prompts can exceed payload savings. The card's payload-bit optimum therefore does not predict a provider-token or cost optimum.
4. Removing an optimal zero-bit sender from the schedule should preserve the MAP posterior and save a request; the private input must remain latent. Treating omission as value zero would violate the frozen source model.

The next model-backed step is gated by the existing full-information capability screen, batch call cap, and a newly passing resource report. It should compare the selected card with decimal/JSON/English and the exact no-message reference on identical tuple IDs, charging full role instructions, completion tokens, application bytes, and setup. A failed capability or resource gate stops that run and is reported without loosening limits.

## Scope and limits

This finite-alphabet scalar-sum task has known sufficient statistics and an explicit exact code, so it is useful for checking lossy coding, receiver uncertainty, and SDK costs. It is not a communication-necessity benchmark for general reasoning and cannot support a claim that this codebook is a general-purpose LLM language. Model-free optimality applies only to deterministic simultaneous fixed-width partitions under the stated prior and bit budget; it excludes framing, model behavior, stochastic encoders, interaction, compute, and tokenizer cost.
