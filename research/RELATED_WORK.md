# Related work map (initial)

**Research date:** 2026-09-27. Claims below are short summaries of public abstracts/specifications and need full-paper review before being used as experimental facts. Search is ongoing; this is not a systematic review.

## Directly relevant: representations and formats

### Chen et al. (2024), *Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication*, Findings of EMNLP

[ACL Anthology](https://aclanthology.org/2024.findings-emnlp.623/) · [paper](https://aclanthology.org/2024.findings-emnlp.623.pdf) · [official code (AutoForm)](https://github.com/thunlp/AutoForm)

The closest prior work found so far. AutoForm adds a prompt inviting an LLM to choose structured/concise formats. In three QA settings it reports lower message token counts and comparable Rouge-L; it also includes HotpotQA with evidence split between two agents, so it does test communication necessity. In that separate-context test, GPT-4/GPT-4 improves F1 0.65→0.69 as reported token count falls 33.8%, while GPT-3.5/GPT-3.5 drops 0.62→0.53 despite a 22.4% reduction. The large 72.7% reduction is from the GPT-4/GPT-3.5 WikiHop pairing, not this HotpotQA split. The paper also tests JSON and KQML and reports task-format transfer across several model pairings.

This establishes a real positive and negative prior, not an empty baseline. The distinct gap to test is the operating frontier under explicit equal budgets, multiple rounds, failure/recovery and current local/open receivers, with total setup/instruction/decoding cost. The paper's tables label the measure “# Tokens”; do not assume it is a complete end-to-end cost measure. The released implementation lists OpenAI and Google API keys in its setup, so exact replication has a different access/cost profile from an open local run. Next: inspect the full prompting, data and token-count implementation before finalizing comparisons.

Full-paper review confirms that AutoForm is a strong baseline to instantiate directly: its multi-agent prompt invites code, pseudocode, JSON, tables, logical operators, and equations rather than prescribing one format. Its HotpotQA split-context experiment is explicitly communication-dependent and shows model-pair-dependent outcomes. Therefore this project must distinguish a format-selection prompt from a fixed encoding and evaluate adherence as well as task quality. A local small-model run that ignores a format instruction is evidence about instruction adherence, not about the representational efficiency of the requested format.

### Xia et al. (2024), *FOFO: A Benchmark to Evaluate LLMs' Format-Following Capability*, ACL

[ACL Anthology](https://aclanthology.org/2024.acl-long.40/)

FOFO reports that open-weight models lag closed models in complex format adherence, that adherence and content quality can vary independently, and that format proficiency differs by domain. This makes format adherence a necessary measurement axis in communication experiments. In DuoSum v0.4, the local Qwen3-1.7B used valid compact-key-value syntax in 10/10 messages and concise-NL syntax in 13/13, but only 9/10 and 9/13 respectively carried the sender's own value. One binary-arm message had binary-digit syntax but none of eight decoded to the sender's value; none of eight JSON messages parsed as JSON. Those arms cannot support claims about correct binary or JSON performance.

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

### Kharitonov & Baroni (2020), *Emergent Language Generalization and Acquisition Speed are not tied to Compositionality*

[ACL Anthology](https://aclanthology.org/2020.blackboxnlp-1.2/)

Finds task-dependent cases where non-compositional protocols generalize or are acquired as well as, or better than, compositional ones. Alongside the Chaabouni/Auersperger results, this means compositionality is a proposed mechanism and transfer property to measure—not a quality score that can stand in for task success.

### Lee (2024), *One-to-Many Communication and Compositionality in Emergent Communication*

[ACL Anthology](https://aclanthology.org/2024.emnlp-main.1157/)

Moves beyond one-speaker/one-listener games and studies broadcasts to multiple listeners who coordinate. Relevant because agent systems often broadcast across role-diverse receivers; a protocol may need to optimize for a receiver population, not one decoder.

### Carmeli et al. (2024), *Concept-Best-Matching: Evaluating Compositionality in Emergent Communication*

[ACL Anthology](https://aclanthology.org/2024.findings-acl.189/)

Proposes a direct best-match mapping between emerged symbols and concepts to make compositionality evaluation more interpretable. Potential metric for future learned-code experiments, in addition to novel-combination transfer and decoder training.

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

### Huang et al. (2026), *When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems*

[arXiv](https://arxiv.org/abs/2601.16280)

The preprint separates tool reliability into stages including initialization, parameter handling, execution, and result interpretation. Its abstract reports tool-initialization failures as a bottleneck for smaller models. This supports measuring message acquisition and subsequent semantic application as distinct stages in v0.11/v0.12; it does not explain our results, since the benchmark and model families differ.

## Communication-dependent task suites

### Zhang et al. (2026), *Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems*

[ACL 2026 paper](https://aclanthology.org/2026.acl-long.1354/) · [arXiv](https://arxiv.org/abs/2603.01045) · [paper code and benchmark](https://github.com/jwyjohn/acl26-silo-bench)

30 generated algorithmic tasks divide private data into agent-local shards; the suite varies task communication complexity and agent count (2–100) and includes peer-to-peer, broadcast, and shared-file-system baselines. Its reported communication-reasoning gap localizes failures to integration of distributed state, not merely information exchange. The upstream II-11 Prefix Sum generator is Unlicense, and the local P2P runner works with an OpenAI-compatible local endpoint. Our DuoSum pilot deliberately left this task family's integration behavior untested. A first v0.7 adaptation exposed both role reversal and an upstream XML parser that mutates message contents; the corrected v0.8 replication holds the model/simulator fixed, preserves the wire string, and uses fresh tasks.

The v0.8 and v0.9 controls reproduce this integration failure at small scale with Qwen3-4B: an oracle can solve the task through the same engine, but the model fails both sender-side subtotal formation and receiver-side use of a correct subtotal. This is consistent with Silo-Bench's diagnosis, but it is not an independent validation of the paper's scale claims or an estimate of model-family behavior.

### Hasan & BusiReddyGari (2026), *DPBench: Large Language Models Struggle with Simultaneous Coordination*

[arXiv](https://arxiv.org/abs/2602.13255) · [code and benchmark](https://github.com/najmulhasan-code/dpbench)

Studies Dining-Philosophers resource contention under sequential and simultaneous decisions. The paper reports that simultaneous choices can deadlock even with communication, while pre-commitment / external coordination changes outcomes. This is not a message-compression comparison, but it reinforces a design distinction for this project: distinguish what the message says from when agents act, what they commit to, and what the runtime enforces. The current PrefixSum harness is sequential and role-explicit, so DPBench's concurrency results do not directly explain our current failures.

### Eisenstein et al. (2026), *MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games*

[arXiv](https://arxiv.org/abs/2602.24188)

Uses collaborative games with private information and divides a fixed token budget over variable numbers of turns. The paper reports that interaction often fails to improve over a one-shot summary baseline despite remaining headroom, while humans achieve comparable success with more token-efficient, coherent dialogue. This is a strong complementary suite for interactive communication policies and turn/budget trade-offs. It motivates treating turn count and task success as protocol properties, while v0.7 isolates a deterministic sufficient statistic in an existing Unlicense task.

### Nath et al. (2026), *CRAFT: Grounded Multi-Agent Coordination Under Partial Information*

[arXiv](https://arxiv.org/abs/2603.25268) · [code](https://github.com/csu-signal/CRAFT)

Introduces complementary private wall views for constructing a shared 3D object, with three directors broadcasting to a builder who chooses among oracle-verified moves. The public repository is MIT licensed and includes local-model support and 20 evaluation structures. This is a stronger candidate communication task than exact sum: the sender set collectively sees information no individual has, and the benchmark distinguishes grounding, belief modeling, pragmatic sufficiency, and task progress. Reproduce its natural-language condition first; a protocol comparison must hold roles, turns, candidate moves, model, and shared state fixed. Its oracle-assisted builder makes it a controlled communication test, not a fully autonomous end-to-end deployment.

### Kriuk & Ng (2025), *Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression*

[arXiv](https://arxiv.org/abs/2512.17914)

Proposes quantized, compressed KV-cache transfer with a heterogeneous-model calibration step and reports 5–6× compression on three QA datasets. This is an important representation-transfer alternative to tokenized messages, but its transfer cost is not directly comparable to text-token count: report transmitted bytes, compatibility/calibration cost, receiver compute, task utility, and persistent model-specific state. Before treating it as a baseline, inspect the full method, implementation availability, and reproducibility; its abstract-level claims do not establish superiority on interactive hidden-information coordination.

### Sevestre & Dupoux (2025), *Frequency & Compositionality in Emergent Communication*

[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1387/)

In referential games, finds that limited exposure—not frequency itself—can induce compositional structure. If a learned LLM protocol is evaluated, report exposure/training-data frequency and held-out primitives; an apparently systematic code may reflect sparse examples rather than a general compositional bias.

### Avsian & Heck (2024), *SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models*

[project/paper links](https://adaravsian.github.io/sneak-website/)

Evaluates selective information sharing under asymmetric knowledge. Potentially useful when protocol efficiency interacts with privacy; not a direct compression baseline. Keep as a later robustness/privacy extension.

## Information-theoretic frame to develop

The appropriate abstraction is task-oriented rate-distortion / information bottleneck: sender state is relevant only insofar as it changes the receiver's decision and eventual task utility. Classical rate-distortion establishes that optimal compression depends on the distortion function; here distortion must be downstream task loss under a particular receiver and interaction policy. Formalization should distinguish message bits/bytes, model-token cost, decoding compute, and shared prior/context. No theorem is claimed yet.

## Search gaps / next reading pass

- Read full papers and inspect code/data for the format-selection paper, ProtocolBench, LatentMAS, Interlat, CondenseFlow, CRAFT, and Q-KVComm.
- Review Lewis signaling games, referential games, iterated learning, and reproducibility critiques of emergent-language benchmarks.
- Review rate-distortion, information bottleneck, communication complexity, interactive compression, and semantic/task-oriented communications.
- Survey coding theory/error correction and protocol negotiation under noisy or adversarial channels.
- Inspect established agent communication language work (FIPA ACL, KQML) and current A2A/MCP implementations without confusing envelope interoperability with semantics.
- Search mechanistic interpretability/representation alignment for implications of hidden-state transfer across models.
