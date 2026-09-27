# Related work map (initial)

**Research date:** 2026-09-27. Claims below are short summaries of public abstracts/specifications and need full-paper review before being used as experimental facts. Search is ongoing; this is not a systematic review.

## Directly relevant: representations and formats

### Chen et al. (2024), *Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication*, Findings of EMNLP

[ACL Anthology](https://aclanthology.org/2024.findings-emnlp.623/)

The closest prior work found so far. It studies alternative formats for reasoning and agent communication and reports model-selected formats, transfer between LLMs, and up to 72.7% lower token usage in its multi-agent setting while maintaining communication effectiveness. This rules out “let an LLM invent shorthand” or “structured format saves tokens” as a sufficient novelty claim. Next: inspect the full paper, benchmark, model configurations, baseline prompting, denominator for token savings, and whether communication is needed to solve the tasks.

## Latent and hidden-state transfer

### Zou et al. (2025), *Latent Collaboration in Multi-Agent Systems* (LatentMAS)

[arXiv](https://arxiv.org/abs/2511.20639) · [code](https://github.com/Gen-Verse/LatentMAS)

Uses continuous hidden representations/shared latent working memory for collaboration; the preprint reports reduced output tokens and faster inference. Key audit questions: model-family assumptions, actual serialized or shared-memory cost, receiver dependence, and comparison equivalence.

### Du et al. (2026), *Enabling Agents to Communicate Entirely in Latent Space* (Interlat), ACL 2026

[ACL Anthology](https://aclanthology.org/2026.acl-long.1248/)

Studies direct transmission of continuous last hidden states and learned compression, including heterogeneous-model claims. This is a direct competing communication mechanism, not merely a transport baseline. We need inspect reproducibility, architecture compatibility, channel accounting, and task setup.

### Chen et al. (2026), *CondenseFlow: Scalable Latent Space Collaboration via Semantic Compression for Multi-Agent Systems*, Findings of ACL 2026

[ACL Anthology](https://aclanthology.org/2026.findings-acl.669/) · [code](https://github.com/xxy33/condenseflow)

Compresses latent/KV information into fixed-size representations; its abstract claims constant-in-context communication complexity and reports cross-model/multi-benchmark results. Strong baseline candidate for compatible local models; account for learned probes, memory, and required architecture access.

## Emergent communication and language structure

### Chaabouni et al. (2020), *Compositionality and Generalization in Emergent Languages*

[arXiv](https://arxiv.org/abs/2004.09124)

Finds that compositionality and generalization are not interchangeable: compositionality may ease transmission to new learners, but does not simply track generalization. Implication: measure novel combinations, receiver adaptation, and population transfer separately.

### Auersperger & Pecina (2022), *Defending Compositionality in Emergent Languages*

[arXiv](https://arxiv.org/abs/2206.04751)

Argues that conclusions about compositionality depend on the evaluation dataset and supports its role for generalization under a suitable split. Implication: benchmark splits must test unseen primitive combinations, not only random held-out examples.

## Protocol and system layers

### Agent2Agent (A2A) Protocol v1

[Specification](https://github.com/a2aproject/A2A/blob/main/docs/specification.md)

An interoperability standard for independent/opaque agents: discovery, task lifecycle, messages, streaming, and artifacts. This project should integrate with or sit above such transports rather than duplicate network-level plumbing. A2A does not by itself answer the semantic representation efficiency question.

### ProtocolBench / ProtocolRouter (2025 preprint)

[arXiv](https://arxiv.org/abs/2510.17149) · [project page](https://hongyidu.ai/projects/protocolbench)

Compares protocol choices on task success, latency, message/byte overhead, and failure robustness, and studies scenario-aware routing. Important adjacent benchmark; inspect overlap and reuse artifacts before creating another protocol benchmark.

### Shen et al. (2025), *Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems*, EMNLP

[ACL Anthology](https://aclanthology.org/2025.emnlp-main.623/)

Studies the communication graph/topology and information/error propagation. Topology and message encoding interact, but topology is not the core novelty target here.

### Wu et al. (2023), *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*

[arXiv](https://arxiv.org/abs/2308.08155)

Representative framework where agents converse using flexible natural language and code. Useful workflow baseline, but framework success is not evidence that conversational text is an efficient semantic code.

## Information-theoretic frame to develop

The appropriate abstraction is task-oriented rate-distortion / information bottleneck: sender state is relevant only insofar as it changes the receiver's decision and eventual task utility. Classical rate-distortion establishes that optimal compression depends on the distortion function; here distortion must be downstream task loss under a particular receiver and interaction policy. Formalization should distinguish message bits/bytes, model-token cost, decoding compute, and shared prior/context. No theorem is claimed yet.

## Search gaps / next reading pass

- Read full papers and inspect code/data for the format-selection paper, ProtocolBench, LatentMAS, Interlat, and CondenseFlow.
- Review Lewis signaling games, referential games, iterated learning, and reproducibility critiques of emergent-language benchmarks.
- Review rate-distortion, information bottleneck, communication complexity, interactive compression, and semantic/task-oriented communications.
- Survey coding theory/error correction and protocol negotiation under noisy or adversarial channels.
- Inspect established agent communication language work (FIPA ACL, KQML) and current A2A/MCP implementations without confusing envelope interoperability with semantics.
- Search mechanistic interpretability/representation alignment for implications of hidden-state transfer across models.
