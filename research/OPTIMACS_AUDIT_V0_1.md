# OPTiMACS literature audit

**Status:** full-paper and official ACL Anthology record reviewed on 2026-09-29. This is a source audit, not an independent reproduction.

## Identification and relevance

Gupta et al., *Learning Optimal Message Representations for Agentic Communication*, Findings of ACL 2026, DOI [10.18653/v1/2026.findings-acl.1441](https://aclanthology.org/2026.findings-acl.1441/), introduce **OPTiMACS** (Optimal Prompt Transformation in Multi-Agent Communication System). This is a close prior, not a peripheral format prompt: it explicitly learns a task-conditioned policy for choosing message representations from multi-agent reward trajectories. It directly rules out claiming that Tacit is the first system to learn adaptive message formats for LLM agents.

## Method as described

- The task-category state is inferred from the raw message, source/target agent descriptions, and a growing task taxonomy. The paper treats this inference as sufficiently concentrated to approximate the partially observed task state by a single category.
- The action is a message structure. The format inventory begins with common forms (including JSON, XML, YAML, code, lists, equations, tables, and KQML) and can expand through LLM-proposed formats.
- A behavior policy mixes learned Q-values, an LLM format proposer, and diversity exploration. The paper's multi-agent setup uses weights `0.5/0.25/0.25`, an exploration rate `0.1`, and an expansion phase followed by a convergence phase. Its policy learns from complete trajectories and task success rewards.
- The learned greedy policy selects a format for the categorized task at inference. The paper says it does not target out-of-distribution generalization; it optimizes trajectories sampled from a dataset and leaves transfer to related datasets as future work.
- Experiments include GSM+ collaborative problem solving and five-agent information-exchange setups on WikiHopQA, HotpotQA, and NarrativeQA. Reported model families include GPT-4o, o3, Phi-4-mini, Llama-3-8B, and Qwen2.5-Math-7B. Policy learning uses roughly 500–2,000 sampled data points, with up to 25 turns per rollout; the reported experimental machine has one A100.

## Reported results and evidence limits

The paper reports improvements over vanilla multi-agent and AutoForm baselines on its evaluated settings. For example, its GPT-4o aggregate accuracy is 59.3 for OPTiMACS versus 55.6 for vanilla; the corresponding o3 values are 69.6 and 66.2. In its fixed-format table, OPTiMACS reports 79.8 average tokens and 1.23 seconds, versus natural language at 87.8 tokens and 1.58 seconds. Across its token-efficiency table, the reported dataset-level token deltas are −8.7% on GSM+, −5.4% on WikiHop, −18.8% on HotPotQA, and **+19.3% on NarrativeQA**. These are paper-reported outcomes, not independently verified measurements.

The cost case is not uniformly positive: NarrativeQA uses more tokens, and several fixed formats have their own quality/cost trade-offs. More importantly for a cost frontier, the paper describes an LLM-based task-categorization step at inference and LLM-based structure proposal during learning, but its headline token tables do not provide a call-by-call ledger separating those costs from communicated messages, nor serialized channel bytes, decoder/setup amortization, or receiver-native tokenizer units. The paper reports average message tokens/time, so its efficiency numbers should not be silently treated as complete end-to-end cost.

The experiments are meaningful evidence that a learned format-selection policy can help on these task distributions. They do not establish OOD transfer, a model-agnostic protocol, semantic compositionality, robustness to channel corruption, or superiority at matched complete inference/channel budgets. The paper's own limitations call out transfer to related datasets and occasional token increases. Its appendix provides pseudocode and prompts. The official ACL Anthology record exposes the paper and checklist but no code/data link; an official implementation was not located in this audit.

## Consequence for Tacit's research thesis

1. Do not claim novelty for “dynamically discovering task-specific message formats” alone. OPTiMACS already studies that problem directly.
2. Treat learned representation selection as a separate baseline family from a reusable communication language. It changes a per-task/per-agent format policy; Tacit must still determine whether a stable, compositional protocol improves the total task-success/cost frontier and transfers across tasks or models.
3. A rigorous follow-up should include an OPTiMACS-style task-conditioned policy when feasible, as well as AutoForm and fixed formats. Freeze the schedule when testing representation, and freeze representation when testing schedule/policy.
4. Count task classification, format search/proposal, format transformation, learning trajectories, and decoder instructions. Report training/search cost and amortize it only over a declared reuse horizon. Keep communication bytes, each receiver's tokenizer tokens, all prompt/completion tokens, latency, and success separate.
5. Compare on held-out instances and tasks, and include heterogeneous receivers. The paper's own no-OOD-design stance makes transfer a decisive test rather than an assumed property.

## Sources

- [ACL Anthology paper page and official metadata](https://aclanthology.org/2026.findings-acl.1441/)
- [Full paper PDF](https://aclanthology.org/2026.findings-acl.1441.pdf)
- [Microsoft Research publication listing](https://www.microsoft.com/en-us/research/publication/learning-optimal-message-representations-for-agentic-communication/)
