# Communication protocol layers and comparison rules

**Research date:** 2026-09-29  
**Purpose:** prevent this project from treating message scheduling, wire transport, and message language as the same intervention.

## Three separable questions

1. **Interaction policy:** who speaks, when, how often, to whom, and whether to stay silent. DALA is a strong example: it allocates scarce turns through a centralized auction and learns value-density bids. This can reduce tokens without improving any individual message representation. Compare it as a scheduling/policy baseline; when testing representations, hold its schedule fixed or report a factorial ablation.
2. **Content representation:** how an agent encodes task-relevant information for another agent: natural language, AutoForm-style adaptive formats, learned task-conditioned formats (OPTiMACS), symbolic codes, or a compositional protocol. Private Match v0.2 is a narrow first controlled screen for this layer.
3. **Transport and system protocol:** envelopes, delivery, discovery, retries, topology, security, and interoperability. Agora and ProtocolBench work here. These are engineering-relevant and can affect measured bytes and latency, but a transport-format win is not evidence that a language represents meaning better.

Latent communication cuts across representation and substrate: embeddings, hidden states, or KV state may avoid text generation, but require compatible model internals, alignment, receiver integration, and byte/compute accounting. A text-only result must state that deployment boundary instead of claiming superiority to latent transfer.

Shared KV reuse is an additional system-state mechanism, distinct from transmitting a compact semantic payload. Prompt Choreography shares text-associated KV encodings across calls within a compatible runtime, reducing repeated prefill; the cached tensor is not a zero-byte network codec. Its same-device assumption, accuracy shifts without adaptation, memory needs, and possible context leakage define an explicit deployment stratum. See the [source audit](PROMPT_CHOREOGRAPHY_AUDIT_V0_1.md).

## What the source review changes

ProtocolBench explicitly measures task quality, latency/throughput, byte overhead, and failure robustness, pins factors such as model, prompts, hardware image and rate limits, and normalizes retries and streaming across A2A/ACP/ANP/Agora. Its router selects protocols per scenario and does not change application semantics. That is strong guidance for systems comparisons; those results do not identify which message content language is best.

Agora describes a meta-protocol combining standardized routines for frequent interactions, natural language for rare interactions, and LLM-written routines for intermediate cases. It motivates adaptive/hierarchical protocols and reuse-horizon accounting. It is not simply a shorthand vocabulary.

The 2026 latent-communication taxonomy distinguishes communicated object (embedding/hidden state/KV cache), sender-receiver alignment, and receiver fusion. This suggests future comparisons must report the interface assumptions and serialized state size, rather than placing a latent result on a text-token axis.

Prompt Choreography sharpens the distinction between *transmitting* a representation and *reusing* an internal representation already resident in shared memory. For a compatible same-runtime workflow, report the cache baseline's latency and inference compute separately from the logical channel payload. For independent endpoints, only compare a KV transfer if actual serialized bytes and receiver integration are measured.

## Falsifiable design consequences

- Keep message representation fixed while varying scheduling, then keep schedule fixed while varying representation. Do not attribute gains across these interventions to “language.”
- Count policy-selection prompts/calls, protocol descriptions, codebooks, receiver instructions, bridges, and transport framing. Report one-time setup and reuse horizon separately.
- For every fixed symbolic/structured arm, log both syntax validity and whether decoding recovers the sender's exact private state. Then score receiver task success independently. A correct-looking message can still be semantically wrong, and a faithful message can still be misdecoded.
- Compare under a task budget and total system costs. Also show the measured application boundary; do not infer physical-network bytes from application payloads.
- A message format selected by an LLM is a policy over a format inventory. Its choice cost, inventory/setup cost, and receiver compatibility belong on its frontier.
- When evaluating end-to-end inference efficiency on repeated-context workflows, include prefix-cache and (where the runtime and task permit it) shared-KV reuse as systems baselines. Preserve identical task visibility and score semantic fidelity, accuracy, privacy, cache memory, and serialized bytes separately.
- A nominal separation between routing and content does not guarantee causal separation: if a router embeds the message payload, a representation change can alter recipients. Estimate content effects at fixed routing and routing effects at fixed content first. Treat the crossed contrast as an interaction only if the router receives an invariant metadata view; otherwise label it as a coupled system intervention. See the [PACT/Proxifield audit](PACT_PROXIFIELD_FACTORIAL_AUDIT_V0_1.md).

## Scope boundary for v0.2

The current Private Match runner is a one-turn, two-role content-representation experiment using one fixed local model endpoint configuration. It has no learned schedule, retries/fault injection, model-family transfer, latent tensors, or long-horizon task. Its hex-nibble arm is only an exact task-code baseline; the existing mixed-radix rank control in v0.1 already reaches the task's 20-bit worst-case zero-error floor for five 16-valued fields. Nothing in v0.2 supports a universal language claim.

## Sources

- Du et al. (2025), [Which LLM Multi-Agent Protocol to Choose? ProtocolBench and ProtocolRouter](https://arxiv.org/abs/2510.17149), arXiv:2510.17149v2 (26 Oct 2025). Its [paper PDF](https://hongyi-du.github.io/publication/protocolbench/ProtocolBench.pdf) specifies the pinned factors and four evaluation axes.
- Marro et al. (2024), [A Scalable Communication Protocol for Networks of Large Language Models (Agora)](https://arxiv.org/abs/2410.11905), arXiv:2410.11905.
- Liu (2026), [Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems](https://arxiv.org/abs/2606.05711), arXiv:2606.05711v3 (15 Jul 2026). This is a preprint taxonomy, not an empirical superiority result.
- Fan et al. (2026), [Cost-Effective Communication: An Auction-based Method for Language Agent Interaction (DALA)](https://ojs.aaai.org/index.php/AAAI/article/view/40182), AAAI 2026, 40(35):29412–29420.
- Chen et al. (2024), [Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication (AutoForm)](https://aclanthology.org/2024.findings-emnlp.623/), Findings of EMNLP 2024.
- Gupta et al. (2026), [Learning Optimal Message Representations for Agentic Communication (OPTiMACS)](https://aclanthology.org/2026.findings-acl.1441/), Findings of ACL 2026.

The source categories are deliberately not pooled: ProtocolBench/Agora/DALA inform adjacent layers and experimental controls, whereas AutoForm/OPTiMACS are content-representation baselines.
