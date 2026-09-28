# Communication-policy baselines audit v0.1

**Date:** 2026-09-29  
**Scope:** S²-MAD (NAACL 2025) and DALA (AAAI 2026) as controls for separating communication policy from message representation.  
**Execution:** literature review only. No repositories were cloned, dependencies installed, or models run.

## Executive finding

Both methods are important baselines because they improve the *schedule and selection* of communication. They do not establish that a new reusable language or encoder is more efficient for the same chosen semantic content.

- **S²-MAD** removes duplicate viewpoints, chooses when agents participate, and uses grouped exchange / early stopping in multi-agent debate. It is a sparsification and participation-policy baseline.
- **DALA** learns message value, speech bids, selection under a central token budget, and a tiered output policy (full text, summary, keywords, silence). It jointly changes who speaks, whether they speak, how much they say, and the content policy. It is a strong learned interaction-policy baseline, not a controlled codec comparison.

In Tacit's notation, either changes \(\sigma\) (interaction/content selection) and potentially \(\phi\) (format choice), while a representation claim about \(\rho,\delta\) requires holding selected content and schedule fixed or replaying the same semantic payloads.

## S²-MAD

**Source:** Zeng et al., [*S²-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency*](https://aclanthology.org/2025.naacl-long.475/), NAACL 2025. The paper compares against CoT, CoT self-consistency, standard MAD, sparse MAD, and GroupDebate across GSM8K, MATH, MMLU, GPQA, and Arithmetic, with GPT-3.5-turbo, GPT-4, and Llama-3.1-8B agents. It reports large total-token reductions against debate baselines (up to 94.5% against MAD) with generally comparable benchmark accuracy; CoT-SC(40) is included as a compute-heavy single-agent comparison. The reported Arithmetic comparisons stop when a single GPT-4 already reaches 100%.

The mechanism is not a message code: groups are formed; similarity/redundancy filtering selects which viewpoints agents receive; conditional participation skips debate when viewpoints align; summaries and early termination control cost. Main results use regex matching, and a separate embedding/cosine strategy uses `bert-base-uncased`. The authors report that the best threshold varies by dataset (for example, 0.1 on GSM8K versus about 0.4 on MATH), while their limitation section notes that regex misses paraphrastic redundancy. This makes redundancy detection itself a task- and representation-sensitive policy component.

The paper's asymptotic accounting charges question, output, and summary tokens and derives savings from participation probability. This is useful protocol-cost analysis, but it does not report end-to-end latency or compute for filtering, and it does not compare alternative encodings of an identical fixed semantic message. We found no code link in the ACL paper record or paper text during this audit; reproducibility therefore remains unverified here.

**Use in Tacit:** compare it as a policy baseline on multi-agent debate tasks. For attribution, run a fixed schedule/payload replay with each codec, and separately compare S²-MAD selection policy over a shared codec inventory. Count the embeddings/matcher, summaries, voting, prompts, and any additional calls at their real costs. Do not transfer its token savings into a claim about a language's information density.

## DALA

**Source:** Fan et al., [*Cost-Effective Communication: An Auction-based Method for Language Agent Interaction*](https://ojs.aaai.org/index.php/AAAI/article/view/40182), AAAI 2026, [AAAI paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40182/44143). DALA trains a value-conditioned auction policy. Agents produce candidate messages and bids, a centralized auction allocates a hard per-round budget, and selected agents use one of four content levels: full, summary, keywords, or silence. MAPPO optimizes communication behavior and task reward. Thus the method changes the message policy and often the content itself, in addition to selecting senders.

The paper uses GPT-4-1106-preview for all agents; allocates 2% optimization data for the learned bidding strategy on benchmarks without an established training split; reports final evaluation on the remaining 98%; and states experiments use NVIDIA A100 80GB. Its task construction says essential formulas/facts/code are distributed among agents so no individual has enough information. It compares against single-agent prompting/CoT/self-consistency and several MAS methods, reports seven benchmarks, and shows token reductions alongside quality gains. The main text reports 6.20M GSM8K tokens in one table while the abstract and later discussion give 6.25M; preserve that discrepancy rather than selecting one value as exact.

The main paper specifies policy training, message value networks, token budgets, and the LLM base model, but the paper record/PDF expose no official code repository in this audit. The main text does not give a complete amortized cost for MAPPO/search, centralized value estimation, prompt annotation, candidate generation, auctions, or format selection. Some behavioral labels (“critical” versus “non-critical” information) use an external `o3-2025-04-16` adjudicator. Consequently, reported inference-token frontiers are useful outcomes for the trained system, but they do not show the full cost of discovering/training that policy or a universal format benefit.

**Use in Tacit:** treat DALA as a strong task-trained system baseline where training, central coordination, and homogeneous GPT-4 access are allowed. Report train/search episodes and their full inference cost separately from held-out use; include per-episode actor/critic/auction calls, input/output tokens, and serialized bytes. To isolate representation, replay a matched set of DALA-selected semantic payloads through candidate codecs/decoders; to compare policies, freeze the representation and count DALA's training/setup plus runtime selection. State explicitly when centralized control or policy-training access is outside a deployment setting.

## Falsifiable comparison predictions

1. If a purported new language advantage is actually sparse scheduling, it should shrink when the message schedule and selected semantic payload are replayed identically across representations.
2. If a learned selection policy is only worth its cost after reuse, its quality/cost advantage should cross the baseline at a measurable horizon after charging policy training, task-specific data, selector/auction compute, and runtime calls.
3. If DALA's value-density mechanism transfers, a policy trained on one task subset should preserve task utility and budget behavior on held-out tasks and receiver setups; otherwise report it as a task/model-conditioned policy.
4. If S²-MAD saves tokens by pruning redundancy, its savings should be weakest when agents hold distinct private facts and strongest when they repeat the same evidence. Evaluate both regimes, and verify that pruning does not remove unique information.

These predictions concern policy attribution and transfer. They do not establish that either method is best, nor do they replace a direct matched-budget codec frontier.
