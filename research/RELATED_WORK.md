# Related work map (initial)

**Research date:** 2026-09-27. Claims below are short summaries of public abstracts/specifications and need full-paper review before being used as experimental facts. Search is ongoing; this is not a systematic review.

## Directly relevant: representations and formats

### Chen et al. (2024), *Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication*, Findings of EMNLP

[ACL Anthology](https://aclanthology.org/2024.findings-emnlp.623/) · [paper](https://aclanthology.org/2024.findings-emnlp.623.pdf) · [official code (AutoForm)](https://github.com/thunlp/AutoForm)

The closest prior work found so far. AutoForm adds a prompt inviting an LLM to choose structured/concise formats. In three QA settings it reports lower message token counts and comparable Rouge-L; it also includes HotpotQA with evidence split between two agents, so it does test communication necessity. In that separate-context test, GPT-4/GPT-4 improves F1 0.65→0.69 as reported token count falls 33.8%, while GPT-3.5/GPT-3.5 drops 0.62→0.53 despite a 22.4% reduction. The large 72.7% reduction is from the GPT-4/GPT-3.5 WikiHop pairing, not this HotpotQA split. The paper also tests JSON and KQML and reports task-format transfer across several model pairings.

This establishes a real positive and negative prior, not an empty baseline. The distinct gap to test is the operating frontier under explicit equal budgets, multiple rounds, failure/recovery and current local/open receivers, with total setup/instruction/decoding cost. The paper's tables label the measure “# Tokens”; do not assume it is a complete end-to-end cost measure. The released implementation lists OpenAI and Google API keys in its setup, so exact replication has a different access/cost profile from an open local run. Next: inspect the full prompting, data and token-count implementation before finalizing comparisons.

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

## Communication-dependent task suites

### Zhang et al. (2026), *Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems*

[arXiv](https://arxiv.org/abs/2603.01045) · [paper code and benchmark](https://github.com/jwyjohn/acl26-silo-bench)

30 generated algorithmic tasks divide private data into agent-local shards; the suite varies task communication complexity and agent count (2–100) and includes peer-to-peer, broadcast, and shared-file-system baselines. It directly addresses task necessity and scaling. It evaluates coordination protocols/topologies, however, so this project should first check whether message representation can be swapped while leaving these mechanics fixed. Upstream declares Unlicense. The official runner uses an OpenAI-compatible API base and may therefore permit local-server evaluation. Candidate first suite; inspect task license/data schemas and source before relying on it.

### Eisenstein et al. (2026), *MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games*

[arXiv](https://arxiv.org/abs/2602.24188)

Uses collaborative games with private information and divides a fixed token budget over variable numbers of turns. The paper reports that interaction often fails to improve over a one-shot summary baseline despite remaining headroom. This is a strong complementary suite for interactive communication policies and turn/budget trade-offs. Reuse if licensing and execution artifacts allow; otherwise reproduce the task design without copying restricted assets.

### Avsian & Heck (2024), *SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models*

[project/paper links](https://adaravsian.github.io/sneak-website/)

Evaluates selective information sharing under asymmetric knowledge. Potentially useful when protocol efficiency interacts with privacy; not a direct compression baseline. Keep as a later robustness/privacy extension.

## Information-theoretic frame to develop

The appropriate abstraction is task-oriented rate-distortion / information bottleneck: sender state is relevant only insofar as it changes the receiver's decision and eventual task utility. Classical rate-distortion establishes that optimal compression depends on the distortion function; here distortion must be downstream task loss under a particular receiver and interaction policy. Formalization should distinguish message bits/bytes, model-token cost, decoding compute, and shared prior/context. No theorem is claimed yet.

## Search gaps / next reading pass

- Read full papers and inspect code/data for the format-selection paper, ProtocolBench, LatentMAS, Interlat, and CondenseFlow.
- Review Lewis signaling games, referential games, iterated learning, and reproducibility critiques of emergent-language benchmarks.
- Review rate-distortion, information bottleneck, communication complexity, interactive compression, and semantic/task-oriented communications.
- Survey coding theory/error correction and protocol negotiation under noisy or adversarial channels.
- Inspect established agent communication language work (FIPA ACL, KQML) and current A2A/MCP implementations without confusing envelope interoperability with semantics.
- Search mechanistic interpretability/representation alignment for implications of hidden-state transfer across models.
