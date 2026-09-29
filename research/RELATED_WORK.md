# Related work map (initial)

The layer distinction used in this project is documented in [Protocol layers and comparability](PROTOCOL_LAYERS_AND_COMPARABILITY_2026_09.md): interaction policy, content representation, and transport/system protocol are separate experimental factors. In particular, Agora/ProtocolBench/DALA are important adjacent system baselines but do not by themselves test which semantic message representation an LLM should use.

**Research date:** 2026-09-29. Claims below are short summaries of public abstracts/specifications and need full-paper review before being used as experimental facts. Search is ongoing; this is not a systematic review.

## Directly relevant: representations and formats

### Chen et al. (2024), *Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication*, Findings of EMNLP

[ACL Anthology](https://aclanthology.org/2024.findings-emnlp.623/) · [paper](https://aclanthology.org/2024.findings-emnlp.623.pdf) · [official code (AutoForm)](https://github.com/thunlp/AutoForm)

The closest prior work found so far. AutoForm adds a prompt inviting an LLM to choose structured/concise formats. In three QA settings it reports lower message token counts and comparable Rouge-L; it also includes HotpotQA with evidence split between two agents, so it does test communication necessity. In that separate-context test, GPT-4/GPT-4 improves F1 0.65→0.69 as reported token count falls 33.8%, while GPT-3.5/GPT-3.5 drops 0.62→0.53 despite a 22.4% reduction. The large 72.7% reduction is from the GPT-4/GPT-3.5 WikiHop pairing, not this HotpotQA split. The paper also tests JSON and KQML and reports task-format transfer across several model pairings.

This establishes a real positive and negative prior, not an empty baseline. The distinct gap to test is the operating frontier under explicit equal budgets, multiple rounds, failure/recovery and current local/open receivers, with total setup/instruction/decoding cost. The paper's tables label the measure “# Tokens”; do not assume it is a complete end-to-end cost measure. The released implementation lists OpenAI and Google API keys in its setup, so exact replication has a different access/cost profile from an open local run. Next: inspect the full prompting, data and token-count implementation before finalizing comparisons.

Full-paper and pinned-source review confirms that AutoForm is a strong baseline to instantiate directly: its multi-agent prompt invites code, pseudocode, JSON, tables, logical operators, and equations rather than prescribing one format. Its HotpotQA split-context experiment is explicitly communication-dependent and shows model-pair-dependent outcomes. Therefore this project must distinguish a format-selection prompt from a fixed encoding and evaluate adherence as well as task quality. A local small-model run that ignores a format instruction is evidence about instruction adherence, not about the representational efficiency of the requested format. See the [implementation and baseline audit](AUTOFORM_BASELINE_AUDIT.md).

### Chen et al. (2025), *Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System*, Findings of ACL

[ACL Anthology](https://aclanthology.org/2025.findings-acl.601/) · [official code](https://github.com/thunlp/Optima) · [paper PDF](https://aclanthology.org/2025.findings-acl.601.pdf)

Optima is a strong learned-interaction baseline, not merely a message-format prompt. Its iterative generate/rank/select/train loop uses task reward, a normalized conversation-token penalty, and base-model language loss; it explores SFT, DPO, and their combination. The study includes complementary-context HotpotQA and 2WikiMultiHopQA as well as reasoning/debate tasks, trains one Llama model to play both roles, and reports transfer from HotpotQA to 2WikiMultiHopQA and TriviaQA. On the paper's 2WikiMultiHopQA transfer table, iSFT-DPO reports F1 54.5 with 70.4 tokens versus MAD F1 25.9 with 543.7 tokens; these are the paper's task-specific metrics, not a universal or byte-level cost comparison. Training took up to 12 hours on eight A100 GPUs for most tasks and about 24 hours for MATH, so setup cost and reuse horizon are material.

This result weakens any claim that the open problem is simply to make natural-language agent messages shorter: task-trained interaction behavior can change both success and communication volume, and the learned behavior can transfer across related datasets. It does not isolate a reusable wire language from task-specific agent-policy training, and its same-model two-role setup does not establish cross-model protocol transfer. A Tacit representation claim must therefore compare against a trained communication/policy baseline when feasible, or explicitly narrow its claim to zero/few-shot, frozen-model, heterogeneous, or deployment-constrained settings; count training/search and report the reuse horizon. A representation-only ablation should hold the learned disclosure/scheduling policy fixed.

### Gupta et al. (2026), *Learning Optimal Message Representations for Agentic Communication* (OPTiMACS), Findings of ACL

[ACL Anthology](https://aclanthology.org/2026.findings-acl.1441/) · [full paper](https://aclanthology.org/2026.findings-acl.1441.pdf) · [detailed audit](OPTIMACS_AUDIT_V0_1.md)

OPTiMACS learns a task-conditioned representation-selection policy from complete multi-agent trajectories. It uses an LLM task categorizer, an expanding inventory of formats, and a behavior policy mixing Q-value exploitation, LLM-proposed formats, and diversity exploration. On its reported datasets it improves task scores over vanilla/AutoForm in many settings and reduces message-token totals on GSM+, WikiHop, and HotPotQA, but increases NarrativeQA tokens by 19.3%. The paper explicitly does not target OOD generalization and leaves transfer to related datasets as future work. Its headline efficiency tables do not give a call-level ledger for task categorization/format discovery, serialized channel bytes, or total setup and receiver-prompt cost; the reported numbers must not be treated as a complete end-to-end frontier. No official implementation link was located in the ACL record or this audit.

This directly challenges any claim that Tacit is the first to learn adaptive message representations. It belongs in the learned-format baseline family, separately from a reusable protocol artifact. Tacit's remaining research question must be narrower and testable: do stable, compositional representations improve the complete frontier under held-out tasks, heterogeneous receivers, equal schedule/budget, and measured selection/setup costs? See the [OPTiMACS source audit](OPTIMACS_AUDIT_V0_1.md).

### Tang et al. (2025), *Augmenting Multi-Agent Communication with State Delta Trajectory* (SDE), EMNLP

[ACL Anthology](https://aclanthology.org/2025.emnlp-main.518/) · [paper](https://aclanthology.org/2025.emnlp-main.518.pdf) · [official code/data](https://github.com/LittleDinoC/StateDelta)

SDE transmits natural-language tokens together with token-aligned differences between adjacent hidden states, then injects those deltas into the receiving model's hidden state. The paper evaluates both asymmetric-information QA and information-symmetric debate/workflow tasks and reports larger benefits on complex reasoning. This is a serious hybrid-channel comparator: it can address reasoning traces that text may omit, but it requires internal-state access and adds a high-dimensional channel whose actual byte, storage, and receiver-compute costs must be counted. Its claims do not establish a portable discrete language, and should not be compared to text using token counts alone.

### Xia et al. (2024), *FOFO: A Benchmark to Evaluate LLMs' Format-Following Capability*, ACL

[ACL Anthology](https://aclanthology.org/2024.acl-long.40/)

FOFO reports that open-weight models lag closed models in complex format adherence, that adherence and content quality can vary independently, and that format proficiency differs by domain. This makes format adherence a necessary measurement axis in communication experiments. In DuoSum v0.4, the local Qwen3-1.7B used valid compact-key-value syntax in 10/10 messages and concise-NL syntax in 13/13, but only 9/10 and 9/13 respectively carried the sender's own value. One binary-arm message had binary-digit syntax but none of eight decoded to the sender's value; none of eight JSON messages parsed as JSON. Those arms cannot support claims about correct binary or JSON performance.

## Latent and hidden-state transfer

### Fu et al. (2026), *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* (C2C), ICLR 2026

[arXiv paper](https://arxiv.org/abs/2510.03215) · [official code](https://github.com/thu-nics/C2C) · [detailed audit](C2C_AUDIT_V0_1.md)

C2C projects a sharer's KV cache into a receiver's cache space and fuses it with a learned residual/gating adapter. The paper reports higher accuracy and lower latency than text handoff across multiple heterogeneous Qwen/Llama/Gemma pairs; its timing counts both model prefill paths and a 90 ms local cache-fusion step. The released Rosetta wrapper loads sharer, receiver, and projector in one process/device and defines no serialized network payload. Main benchmark agents prefill the same shared input, so the results show learned cross-model representation enrichment, not communication necessary to convey private facts. Charge full KV bytes by sequence/layers/dtype, transfer/framing and synchronization, both model passes, fuser training/storage, and test on a receiver-need partition. Context-length accuracy trends are not an equal-byte communication scaling law. See the [C2C audit](C2C_AUDIT_V0_1.md).

### Bao et al. (2026), *Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights* (TFlow)

[arXiv paper](https://arxiv.org/abs/2605.13839) · [official code and inference checkpoint](https://github.com/BWR-hhh/TFlow) · [detailed audit](TFLOW_AUDIT_V0_1.md)

TFlow converts sender hidden states into transient, receiver-specific LoRA perturbations rather than appending messages to the receiver's context. In a three-agent Qwen3-4B setup, it reports higher accuracy than a standalone receiver with fewer processed tokens, and 71–83% fewer tokens plus 2.3–4.6x lower latency than its text-based multi-agent comparator. The gap on HumanEval+ is larger; applying instance-specific LoRA perturbations also makes TFlow slower than the standalone receiver on four of five tasks, despite beating the token-heavy TextMAS baseline. Its channel is a tensor/weight update, so token count is not a bandwidth measure: serialized factors, sender activations, generator setup/compute, receiver patching cost, and reuse horizon must be counted. The demonstrated receiver-specific, shared-Qwen3 setup is not model-agnostic or API-only. The public repository includes inference but not training code and links a roughly 123 MB generator checkpoint.

This is a conditional systems baseline, not a text-language competitor under identical access assumptions. A static audit of the pinned evaluator found that TFlow token metadata counts only the receiver's generation, while TextMAS sums sender and receiver input/output tokens; TFlow's two full-prompt hidden-state forward passes are omitted. The reported token savings need corrected accounting, though this alone does not refute the accuracy or wall-time results. The demonstrated benchmark gives every role the same question and shares one in-process Qwen3-4B backbone, so it does not establish communication of private information or a portable remote payload. A causal comparison must include receiver-need tasks, true versus mismatched-instance and neutral perturbations, and portable serialized bytes as well as full inference cost. See the [implementation follow-up](TFLOW_AUDIT_V0_2.md). Do not claim that a discrete protocol beats the strongest latent/weight-space alternatives unless this comparator is run or excluded for an explicit deployment constraint.

### Ramesh & Li (2025), *Communicating Activations Between Language Model Agents*, ICML

[arXiv](https://arxiv.org/abs/2501.14082) · [TMLR/ICML proceedings PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ramesh25a/ramesh25a.pdf)

Transfers/intervenes on intermediate activations between frozen language models and reports up to 27% improvement over natural-language communication on its evaluated coordination and reasoning setups with lower compute. This is an important upper-end alternative to text: it retains non-linguistic representations and depends on model weights/activations, projection/alignment, layer compatibility, and access to the receiver forward pass. It is not a drop-in baseline for API-only or heterogeneous black-box agents. Audit payload bytes, projection/training cost, model pairing, and causal controls before using the reported gain as evidence that a new text protocol is needed.

### Zhang et al. (2024), *Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems* (AgentPrune)

[arXiv](https://arxiv.org/abs/2410.02506)

AgentPrune targets redundant message edges and turns in multi-agent topologies, reporting substantial token reductions with comparable task scores across several benchmarks. It changes communication topology and message selection, not the code used to express a selected message. This makes it a relevant policy/topology cost baseline and a warning that gains attributed to a language may instead come from eliminating unnecessary exchanges.

### Wang et al. (2025), *AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration*, ACL

[ACL Anthology](https://aclanthology.org/2025.acl-long.1170/) · [official code](https://github.com/wangzx1219/AgentDropout)

Learns to remove redundant agents and communication edges dynamically across rounds. The paper reports average reductions of 21.6% in prompt tokens and 18.4% in completion tokens against its selected baselines. This is a strong topology/policy control because it reduces both receiver-side rereading and sender-side generation; it does not establish that retained message content is encoded more compactly. Future format comparisons need to hold active agents, edge schedule, and rounds fixed, or include a matched AgentDropout arm and charge the selector's training/inference cost.

### Zou et al. (2025), *Latent Collaboration in Multi-Agent Systems* (LatentMAS)

[arXiv](https://arxiv.org/abs/2511.20639) · [code](https://github.com/Gen-Verse/LatentMAS)

Uses autoregressive latent thoughts and a shared latent working memory; the paper reports results across nine benchmarks, five Qwen3/Llama3 backbones, and sequential/hierarchical settings, including up to 14.6% higher accuracy, 70.8–83.7% fewer output tokens, and 4–4.3× faster inference. The implementation passes and trims `past_key_values` between agents, so “zero text tokens” is not zero communication: the experiment must count the transferred tensor/cache elements, dtype, serialization, device/network movement, and receiver-side compute. The paper's shared-memory and same-backbone paths are not equivalent to an independently deployed heterogeneous-agent wire protocol. This is a serious upper-bound systems baseline when internal access and compatible runtimes are available.

### Zheng et al. (2025), *Thought Communication in Multiagent Collaboration* (ThoughtComm), NeurIPS 2025 Spotlight

[NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/b2b502c3629beadda06311386d2c6f73-Paper-Conference.pdf) · [arXiv full text](https://arxiv.org/html/2510.20733v1)

Uses a sparsity-regularized autoencoder to infer shared/private latent dimensions and their agent dependency structure, then routes dimensions and injects a learned prefix. Its identifiability result is bounded by invertible generative-process, variation, and sparse-Jacobian assumptions; it does not guarantee semantic grounding or finite-sample recovery for arbitrary hidden states. The paper reports strong MATH/GSM8K debate results (e.g. Qwen3-1.7B MATH 93.0% vs 75.8% Multiagent Finetuning), but each agent receives the same benchmark question, so necessary private-evidence transfer is not established. Its experiments use 500 training and 500 evaluation examples and a single node with eight H100s. No serialized payload-byte or complete cross-process cost frontier is reported. Treat as a serious trained latent collaboration baseline when internals and training are available; count communicated coordinates/bytes, both endpoints, adapters, and setup. See the [source audit](THOUGHTCOMM_AUDIT_V0_1.md).

### Yu et al. (2026), *Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems* (DiffMAS), arXiv preprint

[arXiv](https://arxiv.org/abs/2604.21794)

Trains task-specific LoRA adapters over a sequential multi-agent latent KV trace and reports gains over text, training-free latent, and C2C baselines across reasoning, coding, and commonsense benchmarks. It is a strong learned-latent upper bound for compatible internal-access deployments. All roles see the same task instance; there is no receiver-need/private-fact benchmark. The AIME tables appear to use 30 items (3.3-point increments) and provide no uncertainty intervals, while the system is trained on task-family examples. The paper also needs clarification on whether gradients update only the final role or the upstream communication encoders. Its trace grows linearly in stages × blocks × dimensions before framing, so latent continuity is not a bandwidth saving by itself. See the [audit](DIFFMAS_AUDIT_V0_1.md).

### Du et al. (2026), *Enabling Agents to Communicate Entirely in Latent Space* (Interlat), ACL 2026

[ACL Anthology](https://aclanthology.org/2026.acl-long.1248/) · [official code](https://github.com/XiaoDu-flying/Interlat)

Transmits temporally aligned last-layer hidden states and introduces learned compression; the receiver uses a learned adapter and supervised training. The paper reports ALFWorld gains over its text condition, including a separate three-agent topology experiment, while MATH overall trails CoT(full) and only the hardest level is better. Its cross-family Qwen-to-LLaMA result lacks a matched cross-family text condition. The reported ~24x figure is message-generation latency, not serialized wire-byte or full-system cost. Treat Interlat as a serious, conditional learned latent-channel baseline, not a portable language for arbitrary/API-only models. Count binary payload bytes, adapter/compressor training and storage, both endpoints' inference, and setup amortization; report cross-family separately. See the [Interlat audit](INTERLAT_AUDIT_V0_1.md).

### Chen et al. (2026), *CondenseFlow: Scalable Latent Space Collaboration via Semantic Compression for Multi-Agent Systems*, Findings of ACL 2026

[ACL Anthology](https://aclanthology.org/2026.findings-acl.669/) · [code](https://github.com/xxy33/condenseflow)

The Latent Thought Condenser uses learned semantic probes to compress KV caches to a fixed-size representation. The paper reports >99% lower KV-cache memory, about 20% lower inference latency than dense transfer, and a 1.7-point average gain over text methods over seven benchmarks/six models. Its constant-size claim concerns the latent payload as context grows; it does not make payload bytes, training/setup, or receiver compute free. The paper assumes equal KV dimensions, describes heterogeneous transfer as future work, and reports 50,000 training steps (about four A100-80G hours); a separate audit finds a quantifier gap between query-specific top-K selections in the proof and the claimed bound for one shared aggregation matrix. The public implementation audit additionally finds that the documented CondenseFlow path reads a cache attribute that is never set in the repository, and the published tests do not exercise that end-to-end path. This makes the linked artifact not reproduced; it does not invalidate the paper's reported empirical results. See the [theory audit](CONDENSEFLOW_AUDIT_V0_1.md), [implementation audit](CONDENSEFLOW_AUDIT_V0_2.md), and [reproducible counterexample](../experiments/condenseflow_audit_v0_1/counterexample.py). CondenseFlow remains a high-priority learned latent-compression baseline when both agents expose compatible internals, not evidence that a text protocol is unnecessary in black-box deployments.

### Zhang & Emu (2026), *Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM*

[arXiv](https://arxiv.org/abs/2607.26773)

This audit replaces the receiver-bound latent payload with controlled alternatives and separates message presence, example-specific content, and additional value from another agent. Its Qwen3-4B/8B results show that aggregate task effects can mix positive and negative components and vary by model/task. This is a direct methodological warning: latent payload capacity, probe readability, or end-task accuracy alone do not establish successful communication. For every channel, include same-shape true-message, other-example/mismatched-message, zero/neutral-message, and no-message controls where task semantics permit; report receiver sensitivity and whether example-specific information provides marginal utility. The paper's reported values are task- and model-specific, not a universal latent-channel verdict.

## Causal influence and communication selection

### Jaques et al. (2019), *Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning*

[ICML 2019 paper (PMLR)](https://proceedings.mlr.press/v97/jaques19a.html) · [arXiv](https://arxiv.org/abs/1810.08647)

This pre-LLM MARL work defines social influence through counterfactual changes to another agent's action distribution and uses influence as an intrinsic reward. In sequential social dilemmas, the paper reports learned explicit-message protocols and correlates influence received through messages with environmental reward. Its discussion cautions that the utility interpretation may rely on repeated interaction: a speaker that sends unreliable information can lose influence over time, a pressure absent in one-shot settings. This establishes causal action influence as a prior concept and separates it from guaranteed task benefit; it does not study frozen LLM agents, natural-language codecs, heterogeneous model transfer, or a communication-cost frontier.

### Ding et al. (2020), *Learning Individually Inferred Communication for Multi-Agent Cooperation* (I2C), NeurIPS

[NeurIPS paper](https://papers.nips.cc/paper_files/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html) · [arXiv full text](https://arxiv.org/abs/2006.06455) · [official code](https://github.com/PKU-AI-Edge/I2C)

I2C infers whether one agent's action changes another agent's policy, learns which neighbor to contact, and regularizes the receiver policy to use messages. It reports lower communication overhead and improved results in cooperative navigation, predator-prey, and traffic-junction MARL tasks. This is a strong communication-policy precedent, but it is trained CTDE with a joint action-value function; it selects sender/recipient links and does not compare textual message representations. It motivates a matched message-selection/sparsification control and confirms the factorization rule: hold topology/policy fixed for codec comparisons, and hold encoding fixed for policy comparisons.

### Zhao et al. (2026), *Are Agents Listening to Each Other? Measuring What Drives Agent Actions in LLM-Based Multi-Agent Systems*

[Wiley article](https://onlinelibrary.wiley.com/doi/10.1111/exsy.70392) · [official code/data](https://github.com/oldblack45/listening-agents)

This black-box study perturbs one incoming message at a realized decision point while preserving pragmatic features, compares next-action distributions, subtracts a same-input sampling-noise floor, and classifies the whole incoming bundle as single-driver, mixed-driver, or no-driver. The paper abstract reports 793 complete operator-specific conditions from 426 retained decision points and says the no-driver share changes from 38.7% to 75.3% under a held-out, bundle-size-aware calibration. The repository README instead advertises 849 decision points and “~40%”; these are different report/version denominators and must not be merged into one estimate. The authors explicitly limit the estimand to local one-step action sensitivity, not long-horizon influence. Its fixed-co-context intervention and noise control are a useful design reference; its label rates are calibration-sensitive and its sensitivity score does not say whether the changed action improves the task. The repository's reproduction path expects an OpenAI-compatible endpoint and agent/judge model calls, so this project records the method but does not launch that reproduction under the current resource/API constraints.

Together, claim grounding and action attribution motivate a three-layer audit for future protocol comparisons: (1) semantic validity, sender evidence, receiver novelty, and task relevance; (2) recipient sensitivity/use under controlled message substitutions at fixed decision context; and (3) terminal task utility and the complete channel/inference cost. A message can pass layer 1 without affecting a decision, pass layer 2 while steering toward a worse action, or improve a local action without improving terminal utility. Calibrate sensitivity thresholds on held-out identity/noise controls, keep co-incoming messages fixed, and report the continuous action-distribution effect alongside any discrete regime label.

The project's novelty cannot be “causal influence matters” or “causal influence can reduce communication”: both are established in MARL. The narrower open question is whether directly communicating frozen or lightly adapted LLM endpoints can use a stable representation to improve task utility per complete communication/inference cost, with semantic transfer across held-out tasks and heterogeneous receivers, beyond strong format and message-selection baselines.

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

### Bai & Eisner (2026), *Accelerating Language Model Workflows with Prompt Choreography*, TACL

[ACL Anthology](https://aclanthology.org/2026.tacl-1.13/) · [official implementation](https://github.com/tjbai/choreo) · [detailed audit](PROMPT_CHOREOGRAPHY_AUDIT_V0_1.md)

Shares dynamically generated message KV encodings inside a compatible model runtime to avoid repeated prefill computation. In the paper's three main MATH workflows it improves per-step time-to-first-token by 2.0–6.2×, while end-to-end speedups are about 1.03×; some more prefill-bound Tree-of-Thought configurations reach 2.2×. Naive choreography can reduce task accuracy, fine-tuning can restore some of it, and context reuse creates information-blockage and leakage risks. This is a strong shared-state inference baseline when agents share a device/runtime and context is repeatedly re-encoded. It does not reduce the logical message's network bytes and is not a portable cross-model language. Keep its compute/cache frontier separate from a representation-and-wire-byte frontier; charge fine-tuning and cache memory when used.

For a serialized transfer comparison, see the model-free [Qwen3-1.7B raw KV-state size estimate](KV_STATE_TRANSFER_SIZE_BOUND_V0_1.md): assuming BF16 cache elements, one token's dense K/V tensors are 112 KiB before container overhead. This is an analytical volume estimate, not a benchmark result and not the bandwidth cost of same-device Prompt Choreography.

### Random access codes (RACs)

Random access codes are a classical communication-theory primitive: a sender compresses a bit vector into a bounded message, and a receiver with a private query index tries to recover the requested coordinate. Classical (n\to1) encoders, including shared-randomness variants, have been studied directly; see [Ambainis et al. (2008)](https://arxiv.org/abs/0810.2937). This prevents framing a hidden-index/coordinate-retrieval task as a new benchmark idea. Tacit's use is narrower: an exact finite control for validating information boundaries, query-distribution effects, and wire-budget accounting before using expensive or capability-limited LLMs. The project-local enumeration and its limits are recorded in [`private_query_v0_1`](../experiments/private_query_v0_1/README.md).

The same paper proves that majority encoding is an optimal deterministic classical (n\to1) RAC under uniform inputs and query averaging, with exact success (1/2+2^{-n}\binom{n-1}{\lfloor(n-1)/2\rfloor}\) and asymptotic advantage (1/\sqrt{2\pi n}\). This is a known result and should be used as the analytic baseline for the uniform one-bit scaling curve; the exact project-local enumeration up to n=4 is an implementation cross-check, not a new theorem. See [`private_query_v0_3`](../experiments/private_query_v0_3/README.md).

### Agent2Agent (A2A) Protocol v1

[Current official specification](https://a2a-protocol.org/latest/specification/) · [normative schema](https://github.com/a2aproject/A2A/blob/main/specification/a2a.proto)

An interoperability standard for independent/opaque agents: discovery, task lifecycle, messages, streaming, artifacts, security, and protocol bindings. Its canonical data model separates task messages from output artifacts and allows `Part` payloads to contain text, raw bytes, URLs, or structured JSON data. It is therefore a real deployment substrate and a strong transport/envelope baseline, but it does not by itself show which semantic representation lets the recipient reason most efficiently. Any eventual runtime should carry its semantic payload through A2A-compatible parts where useful and report inner payload cost separately from the complete serialized A2A envelope.

The [protocol boundary audit](AGENT_PROTOCOLS_SEMANTIC_BOUNDARY_AUDIT_V0_1.md) compares A2A 1.0.0 with MCP 2026-07-28, FIPA ACL, and KQML. It refines the framing: FIPA ACL does specify formal communicative-act semantics, and KQML has performative semantics/protocol work, so these are not merely transport envelopes. Their acts and lifecycle semantics still do not determine an efficient task-content code for a particular LLM pair and task distribution.

### Sander et al. (2026), *A Technical Taxonomy of LLM Agent Communication Protocols* (preprint)

[arXiv](https://arxiv.org/abs/2606.19135)

Classifies nine adopted open-source protocols across counterparty, payload, interaction state, discovery, and schema flexibility. The authors argue that a layered/federated stack is more plausible than one standard maximizing efficiency, portability, and versatility simultaneously. This is a descriptive taxonomy, not an empirical comparison of semantic fidelity or task/cost frontiers. It supports keeping Tacit's semantic coding question separate from transport interoperability and treating A2A/MCP as neighboring layers rather than competitors in the same metric.

### ProtocolBench / ProtocolRouter (2025 preprint)

[arXiv](https://arxiv.org/abs/2510.17149) · [project page](https://hongyidu.ai/projects/protocolbench)

Compares protocol choices on task success, latency, message/byte overhead, and failure robustness, and studies scenario-aware routing. Important adjacent benchmark; inspect overlap and reuse artifacts before creating another protocol benchmark.

### Zeng et al. (2025), *S²-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency*, NAACL

[ACL Anthology](https://aclanthology.org/2025.naacl-long.475/)

Uses sparsification to suppress ineffective exchanges in multi-agent debate. The paper reports up to 94.5% lower token cost than its MAD baseline with less than 2% performance degradation on its evaluated settings. This is a strong policy-level challenge: a communication language must be compared after message selection/sparsity is controlled, or gains could come from dropping low-value turns rather than expressing retained content more efficiently. Reproduce its task/model-specific baselines before treating the headline as a general efficiency bound.

The method uses grouping and a decision mechanism to selectively incorporate non-redundant responses that differ from an agent's current viewpoint. Its evaluations are reasoning/debate tasks (including arithmetic, GSM8K, MATH, MMLU, and GPQA), not complementary-private-evidence pooling. No official implementation link is provided on the ACL/ArXiv paper landing pages reviewed on 2026-09-29. A faithful reproduction would be a policy comparator, not a message-language baseline, and must test whether a novelty filter preserves unique evidence rather than only diverse opinions.

### Fan et al. (2026), *Cost-Effective Communication: An Auction-based Method for Language Agent Interaction*, AAAI

[AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40182)

DALA treats communication opportunities as a scarce resource and selects messages by predicted value density. This also changes who speaks and when, rather than inventing a new content code. It strengthens the need to separate channel policy (selection, scheduling, topology) from representation (encoding/decoding), and to include a policy-matched baseline in any language comparison. The reported benchmark scores are abstract-level claims here, not independently audited results.

The paper specifies a centralized auction whose actor proposes messages and whose critic estimates value density; an actor-critic system is trained using MAPPO, with communication cost in the reward. The official proceedings page reviewed here links the PDF but not a code repository. That trained scheduler is materially heavier than an inference-time message filter and is not a practical first local baseline for this repository's resource-constrained setup.

### Nguyen et al. (2026), *Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention* (DAR)

[arXiv paper](https://arxiv.org/abs/2603.20640) · [MIT implementation](https://github.com/DA2I2-SLM/DAR)

DAR retains a subset of existing agent responses to preserve disagreement and limit repeated broadcasting. Its public code exposes basic MAD, top-k uncertainty filtering, and a critical-message filtering method, with a Hugging Face backend alongside vLLM. The documented benchmark set covers arithmetic/GSM8K, MMLU subsets, HH-RLHF, and CommonSenseQA; the quick-start recommends an H100 for a short run, and the HF example batches 16. This makes it a useful inspectable policy reference, but not direct evidence on asymmetric-information tasks or a low-load drop-in. Any adaptation must count selector/model inference cost and verify that retention does not discard the only agent holding a task-critical private fact.

### Wang et al. (2020), *Learning Efficient Multi-agent Communication: An Information Bottleneck Approach*, ICML

[PMLR proceedings](https://proceedings.mlr.press/v119/wang20i)

IMAC jointly learns compact messages and a scheduler under limited-bandwidth multi-agent reinforcement learning. Its information-bottleneck framing is relevant to the theory: message utility is task-conditioned, and connection/scheduling cost is part of the resource budget. Its learned policies and embodied cooperative tasks are not a direct LLM baseline, but show that optimizing message content while fixing whether/when agents communicate leaves out a long-established part of the problem.

### Bae et al. (2026), *LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning* (LMAC), ICML

[arXiv full paper](https://arxiv.org/abs/2605.18077) · [project/code](https://saaangjun.github.io/LMAC/)

This is a close adjacent precedent for letting an LLM design a communication protocol. It generates executable code that maps local observations to messages, then refines that protocol using offline state-reconstruction feedback. The protocol is used by trained MARL agents; LLMs are not the online communicating endpoints. The paper reports GPT-4.1 protocol-design cost (70.4k tokens, estimated $0.227 per run) and auxiliary decoder training, and tests constrained message dimensions. Thus “LLM-designed protocol” is not itself a novelty claim for Tacit. The open question here is narrower: whether heterogeneous or same-family LLM agents can directly use a stable protocol to exchange task-relevant semantics, at a better end-to-end success/cost frontier than optimized natural language and other baselines. LMAC's receiver state-reconstruction criterion also motivates task-conditioned semantic fidelity, while its offline decoder training and MARL stack are too heavy to be the first resource-bounded local experiment.

### Wang et al. (2024), *Reasoning in Token Economies: Budget-Aware Evaluation of LLM Reasoning Strategies*, EMNLP

[ACL Anthology](https://aclanthology.org/2024.emnlp-main.1112/)

Compares reasoning methods under matched query, token, and monetary budgets. The authors report that chain-of-thought self-consistency can outperform more elaborate methods when given comparable compute, and that some multi-agent debate strategies worsen as budget grows. This is a direct warning against attributing quality gains to collaboration or language without matching total inference resources; report both channel cost and full inference budget, including single-agent compute-matched controls.

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

### Dongre & Hakkani-Tür (2026), *Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue*, SIGDIAL

[ACL Anthology record and paper](https://aclanthology.org/2026.sigdial-1.21/)

Extends PARTNR with two partially observed agents, complementary capabilities, and a dialogue channel. The tasks are designed so neither agent can finish alone: one agent can stage objects while only the other has the state-changing actions. The paper compares silent coordination, synchronous/costed messaging, and asynchronous/free messaging, and defines observation convergence, belief convergence, grounded alignment, information novelty, and belief-sensitive messaging. In its paired Sonnet slate, synchronous dialogue reduced action conflict by 92.6 percentage points while success fell from 0.706 to 0.529; the asynchronous/free condition still had lower success (0.412) despite removing message action-step cost. Across three models, dialogue lowered task success in the tested setup. The paper attributes the discrepancy partly to hallucinated entity references; its handle-based audit classified 60–69% of mentions as neither observed by sender nor receiver.

This is a strong negative prior and a candidate task design, not evidence about message-format superiority. The paper explicitly treats dialogue as a policy over shared belief state and says lexical choice and information packaging are outside its scope; it is therefore complementary to, rather than a direct comparison against, a new language. It motivates measuring grounded novelty and receiver belief alongside conflict and success, with dialogue/no-dialogue controls and matched architecture. It also warns that “more alignment” proxies and fewer conflicts can move opposite to actual task completion. The base [PARTNR repository](https://github.com/facebookresearch/partnr-planner) is MIT licensed and documents local Hugging Face Llama-3.1-8B baselines, but our search found no public artifact for this paper's dialogue extension; its experiments use hosted Claude/Mistral models. Reproduction is not yet a near-term local path: verify extension code/data availability and simulator resource needs before adoption. The Anthology abstract says 40–83 percentage points, while the PDF says 41–93 and its paired Sonnet table gives 92.6 points; preserve this source/version discrepancy and scope the paired-table value to its exact condition.

### Anjum et al. (2026), *ProMCP: Profiling Token Flows and Latency Costs in MCP-Based LLM Agents*, Findings of ACL

[ACL Anthology record, paper, and official code](https://aclanthology.org/2026.findings-acl.1967/)

Introduces end-to-end instrumentation that correlates timestamps, payloads, and token accounting across MCP Host–Client–Server boundaries, and decomposes workflows into six stages. The reported study spans 20 servers and 169 tools. Its abstract reports that customized-client deployments spend 56–72% of tokens and 60–67% of latency in planning/schema injection, while an off-the-shelf client spends over 85% of latency in final synthesis; tool execution itself is a small share. These are MCP workflow measurements, not an LLM-to-LLM semantic-code comparison, and they should not be imported as expected effect sizes for Tacit.

The transferable lesson is methodological: measure stage-attributed end-to-end cost, repeated context/schema, planning, payload, retries, and final synthesis rather than treating the wire string as the whole system. This complements `tlu.costs.v3`, which records per-episode inputs and serialized payloads but does not yet prescribe a per-call stage timeline. Use ProMCP as an instrumentation reference if a future runtime spans MCP components; do not add MCP tool-schema overhead to a bare two-agent codec experiment unless that runtime is part of the declared system boundary.

### Zhang et al. (2026), *Silo-Bench: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems*

[ACL 2026 paper](https://aclanthology.org/2026.acl-long.1354/) · [arXiv](https://arxiv.org/abs/2603.01045) · [paper code and benchmark](https://github.com/jwyjohn/acl26-silo-bench)

30 generated algorithmic tasks divide private data into agent-local shards; the suite varies task communication complexity and agent count (2–100) and includes peer-to-peer, broadcast, and shared-file-system baselines. Its reported communication-reasoning gap localizes failures to integration of distributed state, not merely information exchange. The upstream II-11 Prefix Sum generator is Unlicense, and the local P2P runner works with an OpenAI-compatible local endpoint. Our DuoSum pilot deliberately left this task family's integration behavior untested. A first v0.7 adaptation exposed both role reversal and an upstream XML parser that mutates message contents; the corrected v0.8 replication holds the model/simulator fixed, preserves the wire string, and uses fresh tasks.

The v0.8 and v0.9 controls reproduce this integration failure at small scale with Qwen3-4B: an oracle can solve the task through the same engine, but the model fails both sender-side subtotal formation and receiver-side use of a correct subtotal. This is consistent with Silo-Bench's diagnosis, but it is not an independent validation of the paper's scale claims or an estimate of model-family behavior.

### Hasan & BusiReddyGari (2026), *DPBench: Large Language Models Struggle with Simultaneous Coordination*

[arXiv](https://arxiv.org/abs/2602.13255) · [code and benchmark](https://github.com/najmulhasan-code/dpbench)

Studies Dining-Philosophers resource contention under sequential and simultaneous decisions. The paper reports that simultaneous choices can deadlock even with communication, while pre-commitment / external coordination changes outcomes. This is not a message-compression comparison, but it reinforces a design distinction for this project: distinguish what the message says from when agents act, what they commit to, and what the runtime enforces. The current PrefixSum harness is sequential and role-explicit, so DPBench's concurrency results do not directly explain our current failures.

### Mao, Yang, and Zhang (2025), *Gadgetless Lifting Beats Round Elimination: Improved Lower Bounds for Pointer Chasing*

[ITCS 2025 paper](https://drops.dagstuhl.de/storage/00lipics/lipics-vol325-itcs2025/html/LIPIcs.ITCS.2025.75/LIPIcs.ITCS.2025.75.html) · CC-BY 4.0

The paper defines two-party `k`-step pointer chasing over independently uniform functions and a parity output. Its Theorem 2 gives an `Omega(n/k + k)` bit lower bound for `(k-1)`-round deterministic protocols with Alice first and success at least `2/3` under that input distribution; Corollary 3 states the corresponding result for randomized protocols with error at most `1/3`. The direct `k`-round pointer-relay protocol costs `O(k log n)` bits. This is a promising source of a round-depth control beyond one-way `INDEX_m`, but the asymptotic theorem is not an LLM-token bound and small-model tasks may not enter its informative regime. The project design and staged eligibility criteria are in [`INTERACTIVE_POINTER_CHASING_DESIGN.md`](INTERACTIVE_POINTER_CHASING_DESIGN.md).

### Eisenstein et al. (2026), *MT-PingEval: Evaluating Multi-Turn Collaboration with Private Information Games*

[arXiv paper and HTML](https://arxiv.org/abs/2602.24188) · [full text](https://arxiv.org/html/2602.24188)

Uses five collaborative game families with private information (chess chronology, COVR image QA, MD3 image matching, Tangram, and a table-intersection name game). The v2 evaluation fixes 128 whitespace-counted tokens per player and partitions that budget evenly over 2/4/8/16 turns; the authors explicitly distinguish this communication budget from model reasoning tokens, which are not charged. Their results show that more turns often fail to help: shrinking per-turn capacity can cause premature stopping or unused rounds, and apparent scaling on the name game can reflect guess-and-check behavior. The paper itself notes a whitespace/subword counting mismatch with model-native tokenizers. These are useful findings and a strong candidate source for interaction-policy questions, but its published “isotoken” protocol is not a complete matched-cost comparison for Tacit: it neither equalizes total inference compute nor tests flexible turn allocation, and an equal total cap does not guarantee that agents actually use the available turns.

**Reuse decision:** treat MT-PingEval as a high-priority task source, not a drop-in format leaderboard. Its v2 arXiv paper is CC-BY 4.0 and provides prompts in the appendix, but no official code or benchmark-data link was located in the paper metadata/full text during this audit; underlying COVR/image assets also have their own terms. The paper's name-game results show why raw multi-turn accuracy is insufficient: a random guess-and-check baseline explains much of the improvement as turn limits grow. A local one-way synthetic calibration now exists at [`experiments/private_match_v0_1/`](../experiments/private_match_v0_1/README.md); it is an independently generated task-family adaptation, not a reproduction or reuse of their generated records. For a later multi-turn extension, require correct-information, deranged/mismatched-information, no-message, guess-and-check, and centralized-oracle controls; show that receiver need and any adaptive-query gain survive at matched delivered bytes. Use the receiver's native tokenizer and enforce caps at runtime; report reasoning/inference compute separately, and log unused turns, messages, and stop reasons. The paper's CC-BY license does not by itself grant rights to unrelated source datasets.

### Park et al. (2026), *PAC-BENCH: Evaluating Multi-Agent Collaboration under Privacy Constraints*, Findings of ACL

[ACL paper](https://aclanthology.org/2026.findings-acl.1552/) · [official code](https://github.com/PAC-Bench/PAC-Bench) · [scenario dataset](https://huggingface.co/datasets/PAC-Bench/PAC-Bench) · [audit](PAC_BENCH_AUDIT_V0_1.md)

Introduces two-agent collaboration scenarios with private memories and explicit disclosure constraints. The paper reports a 100-scenario evaluated set; its public Hugging Face collection lists 1,476 rows. It identifies early privacy violations, over-abstraction, and privacy-induced hallucination. This adds a useful privacy-constrained deployment axis, but it does not compare message representations under matched channel budgets. Task and privacy evaluation rely partly on LLM judges (with human checks on a subset); the public workflow uses a multi-agent tool stack and defaults to Docker with 20 containers. The dataset card states CC-BY-4.0, while the observed repository README does not state a code license. Treat it as a secondary external stress test, not a deterministic primary protocol benchmark, and audit each asset's license separately.

### Nath et al. (2026), *CRAFT: Grounded Multi-Agent Coordination Under Partial Information*

[arXiv](https://arxiv.org/abs/2603.25268) · [code](https://github.com/csu-signal/CRAFT)

Introduces complementary private wall views for constructing a shared 3D object, with directors communicating to a builder who chooses among oracle-verified moves. The public repository is MIT licensed and includes local-model support and 20 evaluation structures. The sender set collectively sees information no individual has, and the benchmark distinguishes grounding, belief modeling, pragmatic sufficiency, and task progress. However, the oracle supplies up to five valid, locally forward-progressing moves. In our v0.24 audit, both a zero-message Qwen Builder and a deterministic policy that only selected from these candidates completed the simple instance; the deterministic policy used no model calls. This falsifies communication necessity for that instance under the oracle-assisted setup. It does not show communication is unnecessary without the oracle, and CRAFT should not support a protocol ranking until the candidate-list information boundary is changed or independently justified. Separately, the runner samples three directors with replacement and overwrites duplicate-ID responses in the Builder transcript while retaining every generated post in Directors' shared history. A full audit of the Apache-2.0 public trace dataset finds this in 4,652/5,930 aligned turns across 15 models: 5,349 of 17,790 posts (30.1%) are collapsed before reaching the Builder transcript. Reconcile this against the paper's evaluation revision before reuse. See the [implementation audit](CRAFT_IMPLEMENTATION_AUDIT_V0_1.md), [full trace audit](CRAFT_TRACE_DATASET_AUDIT_V0_2.md), and [v0.24 necessity report](CRAFT_COMMUNICATION_NECESSITY_V0_24.md).

### Li, Naito, and Shirado (2026), *Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs* (HiddenBench)

[ICML 2026 paper](https://arxiv.org/abs/2505.11556) · [MIT code and 65-task benchmark](https://github.com/Yassellee/HiddenBench_ICML)

Uses verified hidden-profile tasks where shared facts favor a wrong answer and unique facts must be pooled. The paper reports 30.1% average group accuracy under hidden information versus 80.7% individual accuracy with complete information across 65 tasks. Its existing Exchange/Decide structured protocol is a strong baseline: two exchange rounds of 1–2 relevant facts plus a reason to reject the current favorite, followed by one decide pass; the paper reports substantial gains on 18 tasks and three frontier models. We audited the pinned MIT implementation and the three-task verification set locally; the audit and proposed local-capability gate are in [HiddenBench audit v0.1](HIDDENBENCH_AUDIT_V0_1.md). This benchmark is validated for distributed decision information, but does not by itself establish long-horizon protocol efficiency, scaling, or transfer.

### Fukushima, Xiong, and Moradi Pari (2026), *The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation*

[arXiv](https://arxiv.org/abs/2609.11489)

Uses Hanabi as a known-answer environment to separate literal communicated content from convention-dependent interpretation. Their analysis of roughly 101,000 play actions across human-human, AI-AI, and human-AI logs reports very different convention gaps across pair types. This is a reminder that a wire message's literal content may not determine what a receiver infers; future Tacit evaluations should separately measure message semantics, receiver belief, and task outcome, especially in cross-model or cross-play conditions.

### Kriuk & Ng (2025), *Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression*

[arXiv](https://arxiv.org/abs/2512.17914)

Proposes a hybrid cache-plus-text-facts transfer: 70% layer selection, adaptive KV quantization, YAKE/NER fact extraction, and statistical calibration. The paper reports compression ratios and heuristic relevance/coherence scores on three QA sets, but its main results table does not provide exact task accuracy/F1 or numeric comparisons to its listed full-text, uncompressed-KV, and uniform-quantization baselines. The cache dtype and compression denominator, whether extracted facts count in bytes, and shape-compatible mapping across different layer/head dimensions are under-specified. Treat it as an unverified candidate, not as evidence of superiority; see the [source audit](Q_KVCOMM_AUDIT_V0_1.md).

### Shi et al. (2026), *KVComm: Enabling Efficient LLM Communication through Selective KV Sharing*, ICLR 2026

[OpenReview](https://openreview.net/forum?id=F7rUng23nw) · [paper](https://arxiv.org/abs/2510.03346) · [official code](https://github.com/Zephyroam/KVComm)

Shares selected KV pairs across model calls using an attention-importance layer selector. Unlike same-prompt debate baselines, its task formulation gives a sender private context C and receiver query Q, making this an important task-grounded latent communication comparator. It reports F1/ROUGE-L across nine same-model or same-base model pairs and reductions against direct input-merging in FLOPs/memory, but no serialized wire-byte frontier. Its pinned code also retains layer-0 KV and one anchor token in each unselected layer; the public default reuses its single calibration example in the evaluation iterator. Charge actual serialized cache bytes and calibration/setup, and separate these code-path observations from the paper's claims. See the [KVComm audit](KVCOMM_AUDIT_V0_1.md). Keep it distinct from similarly named KVCOMM, which reuses overlapping-context caches to reduce prefill work rather than transmit task-relevant private content.

### Shi et al. (2025), *KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems*

[arXiv](https://arxiv.org/abs/2510.12872) · [official code](https://github.com/FastMAS/KVCOMM)

Reuses and aligns KV-cache segments for overlapping shared text across dependent agents, reporting reduced time-to-first-token on shared-prefix multi-agent workloads. This is a serving/prefill optimization baseline: it can reduce the compute cost of rereading common context but does not itself encode or carry the sender's private task-relevant contribution. Include it in end-to-end cost accounting where shared context is large, while keeping it separate from semantic message-channel comparisons.

### Cheng et al. (2026), *When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs*

[arXiv](https://arxiv.org/abs/2608.04893)

Audits whether gains from KV relays depend on the correct example-specific cache by comparing true, mismatched, zeroed, and randomized payloads, and separates receiver-need from receiver-no-need regimes. The authors report strong task-relevant transfer in a calibrated private-information setting, but little or no pairing effect on several standard benchmarks, and recommend mismatched-cache controls before attributing gains to transmitted latent thoughts. This gives Tacit a concrete causal falsification design: equal-shaped, equal-budget relevant versus deranged messages, with single-agent/oracle controls and receiver need explicitly measured.

### Wenzel (2026), *Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text*

[arXiv](https://arxiv.org/abs/2607.14103)

Reports that SAE-sparse latents preserve probe-readable features at much lower representational size than text, yet its task-level cross-lingual concept results do not beat text; the author concludes that much of the lost latent feature content may be surface-form information rather than task-relevant semantics. This is a direct negative result against equating probe recoverability or compression with useful communication. Our primary endpoint must remain held-out downstream task utility under receiver-need controls.

### Sevestre & Dupoux (2025), *Frequency & Compositionality in Emergent Communication*

[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1387/)

In referential games, finds that limited exposure—not frequency itself—can induce compositional structure. If a learned LLM protocol is evaluated, report exposure/training-data frequency and held-out primitives; an apparently systematic code may reflect sparse examples rather than a general compositional bias.

### Avsian & Heck (2024), *SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models*

[project/paper links](https://adaravsian.github.io/sneak-website/)

Evaluates selective information sharing under asymmetric knowledge. Potentially useful when protocol efficiency interacts with privacy; not a direct compression baseline. Keep as a later robustness/privacy extension.

## Emergent communication, compositionality, and iterated learning

Lewis signaling / referential games are useful controlled environments for asking whether a sender's signal changes a receiver's action. But successful coordination inside one paired population is a much weaker claim than a reusable language. An arbitrary holistic code can solve a finite shared game; it need not transfer to held-out meanings, a fresh receiver, a new task, or a noisy channel.

- [EGG](https://github.com/facebookresearch/EGG) is a reusable toolkit for discrete and continuous channels, sender/receiver games, and population learning. It contains reference-game, anti-efficient encoding, compositionality/generalization, and language-bottleneck examples. The public repository was archived on 2026-08-10, so treat it as a valuable historical implementation and task taxonomy rather than an actively maintained platform. Its own README describes game success and channel training, not a general-purpose cross-task LLM protocol.
- Chaabouni et al., [*Anti-efficient encoding in emergent communication*](https://papers.neurips.cc/paper/8859-anti-efficient-encoding-in-emergent-communication.pdf) (NeurIPS 2019), show that a successful learned signaling code need not obey a communication-efficiency pattern: in their skewed-frequency game, frequent meanings acquired longer messages, and explicit length pressure changed the result. Message length is therefore an objective / receiver-bias outcome, not a property guaranteed by emergence.
- Mu & Goodman, [*Emergent Communication of Generalizations*](https://proceedings.neurips.cc/paper/2021/hash/9597353e41e6957b5e7aa79214fcb256-Abstract.html) (NeurIPS 2021), argue that single-object shared-context games encourage overfitting. Set-reference and concept games over sets of objects improve held-out concept generalization and systematicity in their setting. This supports evaluating compositional transfer over recombinations of primitives, not only random held-out instances from the same game.
- Kobrock et al., [*Agents generalize to novel levels of abstraction by using adaptive linguistic strategies*](https://aclanthology.org/2025.findings-acl.455/) (Findings ACL 2025), evaluate zero-shot transfer in a concept-level reference game. They report different strategies by transfer direction: compositional messages when moving from generic to specific concepts, and reuse of training messages when moving from specific to generic concepts. This is direct prior work on receiver-side generalization and cautions against treating one fixed compositional code as the only meaningful success criterion.
- Rita et al., [*Emergent Communication: Generalization and Overfitting in Lewis Games*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract-Conference.html) (NeurIPS 2022), analytically separate co-adaptation from information loss and show that overfitting to co-adaptation can undermine structure; controlling that source improves compositionality and generalization. A communication score measured on agents co-adapted to one another must not stand in for receiver-independent semantics.
- Kouwenhoven et al., [*Searching for Structure: Investigating Emergent Communication with Large Language Models*](https://aclanthology.org/2025.coling-main.667/) (COLING 2025), are a direct LLM precedent. Fifteen seeded simulations used instruction-tuned Llama 3 70B with greedy decoding; 15 of 27 attribute-combinations were used for communication and the full 27 were tested. Signals became more structured over four 30-interaction rounds, and success was about 70–75% against 25% chance. This is real evidence that LLM in-context learners can adapt artificial vocabularies, not evidence that the resulting protocol beats natural language or minimizes communication cost. Their six 8-generation transmission chains improved learnability, but signals lengthened; TopSim did not significantly rise across generations, and some vocabularies collapsed to many-to-one/underspecified mappings. The authors note sensitivity to prompts, task instructions, tokenization and greedy decoding, and identify decoding feedback as one possible cause. This is a useful positive-and-negative control for our claims.

  **Evaluation boundary:** the held-out block generates signals for all 27 meanings and reports a structure correlation; it does not ask an independent receiver to select the right novel target. So held-out signal generation is not held-out receiver utility. The paper's methods say one target plus four distractors, while its results state 25% chance; a replication must resolve candidate count from its actual episode generator and derive chance from that count. It also does not give an equal-budget frontier against optimized text, JSON, a compositional symbolic code, or a holistic lookup code. See the [held-out utility audit](EMERGENT_OOD_UTILITY_AUDIT_V0_1.md) for the falsifiable receiver-side follow-up and limits.
- [Raviv & Arnon (2018)](https://doi.org/10.1016/j.cognition.2018.08.011) explicitly distinguish systematicity from compositionality in iterated-learning experiments and note the gap between adult findings and evidence from children. Iterated transmission can amplify learner biases, but does not by itself prove that the resulting code has a compositional semantics or that the same mechanism applies to LLM agents. Likewise, iterated learning, repeated interaction, and population reinforcement learning are distinct update dynamics and should not be conflated.

**Evidence standard for any emergent-language claim:** report exact task success separately from protocol structure; for held-out compositions, require an independent receiver to choose correctly among candidates rather than only generating a signal or scoring structural similarity; test genuinely novel task instances; decode with a second model family when feasible; measure semantic fidelity, description length/serialized bytes, ambiguity, robustness to corruption, and recovery; and run multiple independent seeds/chains. Compare against a holistic lookup code, optimized natural/structured text, and an oracle upper bound at the same channel budget. TopSim, n-gram reuse, and vocabulary learnability are diagnostics, not substitutes for compositional generalization or task utility. Where a negotiation or transmission phase is needed, charge its complete communication and inference cost and report the reuse horizon.

**Project decision:** keep emergent learning as a candidate route rather than the default design. The 2025 LLM study makes the route plausible, while its message growth, semantic collapse, and narrow referential setup show why emergence alone is not a value proposition. Do not train or search for a protocol until the fixed-model communication tasks establish a measurable capability region and the host's frozen resource gate permits inference.

The [generalization prior audit](EMERGENT_GENERALIZATION_PRIORS_AUDIT_V0_1.md) records why held-out receiver utility and compositional transfer are not novel by themselves. The small `emergent_ood_v0_1` split is only a deterministic validation fixture; it must not be presented as a new benchmark contribution.

## Channel errors, feedback, and conversational repair

Classical channel coding separates a physical transmission channel from the code and studies rate versus decoding error (Shannon's channel-coding theorem). That is relevant if the transport can actually corrupt or erase payloads. The model-free [private-query v0.5 control](../experiments/private_query_v0_5/README.md) adds an exact finite RAC frontier under a binary symmetric channel; it is explicitly not a text-understanding model. LLM systems have additional failure sources that a bit-flip model does not capture: a receiver can misinterpret an intact string, ignore a field, hallucinate a missing value, truncate context, or apply a convention it learned from another model. Report transport integrity and semantic fidelity separately.

Nikolaus, [*Emergent Communication with Conversational Repair*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/27a2b7a22f91245200ebe89e468a1c54-Abstract-Conference.html) (ICLR 2024), adds a receiver-to-sender feedback channel to a learned Lewis game. Sender messages can be replaced by a detectable special noise token; the receiver has a binary feedback alphabet and returns a token at each sender step. Three-seed experiments find that feedback improves held-out task accuracy under channel noise while lowering TopSim; the feedback symbols can acknowledge/reconstruct sender symbols and convey aspects of the receiver's candidate objects, so the agents co-construct contextualized meaning rather than just asking “repeat.” The natural-image GuessWhat setup showed a similar direction. The released [code](https://github.com/mitjanikolaus/emergent_communication) makes this a reproducible methodological precedent.

Its boundary matters: this is a small jointly trained RNN signaling game, not LLM-to-LLM language; the primary noise marker is explicitly detectable; the reverse feedback channel is uncorrupted; feedback capacity is one binary symbol per time step; and the paper does not compare under an equal *total bidirectional* message budget. Performance in its synthetic noise condition therefore motivates repair as a candidate policy but does not show that extra LLM turns pay for themselves. The paper also demonstrates that structural proxies such as TopSim can move opposite task generalization under noise.

**Tacit design implication:** separate at least (a) transport corruption/truncation, (b) receiver semantic decode error with an intact payload, and (c) model/context mismatch. Use exact payload replay and controlled corruption to isolate (a); use sender-intended meaning versus receiver-reconstructed fields for (b); vary receiver family, prompt, and shared context for (c). Compare no repair, fixed redundancy, receiver acknowledgement/clarification, and adaptive repair at matched *total* channel and inference budgets, charging the reverse message, repeated context, and model calls. A repair question is relevant only when it reduces task loss on information that can change the receiver's action. Detailed predictions are recorded in the [channel robustness audit](CHANNEL_ROBUSTNESS_AUDIT_V0_1.md).

## Communication-policy alternatives: S²-MAD and DALA

These methods challenge free-for-all messaging and must be included as policy/system baselines where applicable. They select agents, turns, and content volume; they do not isolate the efficiency of a fixed semantic payload's representation. S²-MAD (NAACL 2025) uses grouping, redundancy filtering, conditional participation, and early stopping; its token savings depend on task/model response similarity, and its regex filter can miss paraphrases. DALA (AAAI 2026) learns value-density bids and chooses full text, summaries, keywords, or silence under a centralized budget; it also uses MAPPO and task-specific optimization data. Treat both as strong alternatives to the interaction policy \(\sigma\), with possible format-selection \(\phi\) changes, rather than direct proof for a message language \(\rho\). A controlled codec comparison must replay the same schedule and semantic payload; a policy comparison must freeze the representation. Full findings and accounting boundaries are in the [policy-baseline audit](COMMUNICATION_POLICY_BASELINES_AUDIT_V0_1.md).

## Information-theoretic frame to develop

The appropriate abstraction is task-oriented rate-distortion / information bottleneck: sender state is relevant only insofar as it changes the receiver's decision and eventual task utility. For a sender with private source X and receiver with private side information Z, Wyner–Ziv source coding is a useful analogy because only the decoder sees Z; Orlitsky–Roche's function-computation setting is closer when the receiver needs a function of both inputs. Interactive information complexity is the corresponding lens for multi-round exchanges. These classical results have precise source, loss, and coding assumptions; none directly states an optimal number of LLM tokens. Modern semantic-communication literature also notes that distortion measures are task-specific rather than universal ([survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC10888479/)).

The project now distinguishes an ideal information-theoretic reference from an observable operational frontier. The former uses a fixed joint task distribution, explicit task loss, sender/receiver information boundary, and ideal block-code assumptions; the latter uses measured bytes, model tokens, runtime, and actual LLM behavior. A task should show a positive oracle value-of-information gap before a language comparison: if the best no-message receiver already matches the centralized oracle under the same task loss, then communication has no optimal task value there. A positive gap only shows opportunity, not that a specific protocol can capture it. See [`docs/THEORY.md`](../docs/THEORY.md), §2, for definitions and proof.

## Search gaps / next reading pass

- Full-paper/code audits are now recorded for AutoForm, LatentMAS, Interlat, CondenseFlow, and two causal audits of latent/KV relays; next, verify benchmark task dependence and reproduce at least one matched, serialized-byte latent-vs-text comparison when the fixed host resource gate permits model inference.
- Reproduce S²-MAD's sparsification and DALA's message-value selection only if code, models, and the frozen resource gate make it feasible; the literature/attribution audit is in [`COMMUNICATION_POLICY_BASELINES_AUDIT_V0_1.md`](COMMUNICATION_POLICY_BASELINES_AUDIT_V0_1.md). Match total model-token/query budget with single-agent self-consistency before claiming gains from a message representation.
- Extend the emergent-communication audit with a reproducibility check of multi-seed / OOD evaluation practice and, if a learned protocol becomes competitive, freeze an independent-receiver and held-out-composition test before training. Initial audit is in the section above.
- Review rate-distortion, information bottleneck, communication complexity, interactive compression, and semantic/task-oriented communications.
- Survey coding theory/error correction and protocol negotiation under noisy or adversarial channels.
- Inspect current MCP/A2A SDK serialization when a runtime is selected; the spec-level boundary audit for FIPA ACL, KQML, MCP 2026-07-28, and A2A 1.0.0 is in [`AGENT_PROTOCOLS_SEMANTIC_BOUNDARY_AUDIT_V0_1.md`](AGENT_PROTOCOLS_SEMANTIC_BOUNDARY_AUDIT_V0_1.md). Keep act/lifecycle interoperability distinct from task-content efficiency while measuring any interactions between them.
- Search mechanistic interpretability/representation alignment for implications of hidden-state transfer across models.
