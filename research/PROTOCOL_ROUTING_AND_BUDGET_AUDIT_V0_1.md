# Protocol routing, message representation, and compute-budget audit v0.1

**Status:** literature synthesis; no code from the cited projects was executed and no model or remote inference API was used. This audit updates the experiment boundary, not a claim that one protocol is superior.

## Why this distinction matters

An observed team-level gain can come from at least four different interventions:

1. **Representation:** what information a message encodes and how it is serialized (natural language, structured text, symbols, or latent state).
2. **Routing:** which sender transmits to which receiver, and when.
3. **Interaction policy:** what agents choose to disclose, request, challenge, or do after receiving a message.
4. **Compute allocation:** number and size of calls, context exposure, model capability, and total inference-token budget.

Changing multiple layers at once can improve a system while leaving the value of a new *language* unidentified. Tacit's primary representation estimand should therefore hold task, role information, topology/schedule, model, decoding, output contract, and total inference budget fixed. Broader system comparisons can follow, but must report the changed layers and full costs.

## New challenge: adaptive routing can beat fixed topology

Tambwekar et al., [*Proxifield: Decentralized Multi-Agent Communication through Semantic Proximity*](https://arxiv.org/abs/2609.20889) (arXiv, 2026-09-16), proposes round-wise sparse routing using explicit addressees, declared information needs, plan alignment, and information complementarity. Its comparison includes independent agents, a star orchestrator, and a shared-context board. The paper reports five matched environment seeds and studies model scale, team size, and permanent failures. On HiddenBench, shared context is ahead at its smallest model setting; Proxifield catches up and passes it with the largest model, so the result itself predicts a capability interaction rather than a universal protocol ranking.

This is a serious **routing and interaction-policy** baseline, not a clean message-representation baseline. Agents produce messages, proposed actions, rationales, needs, and addressees; receivers reply with accept/reject/counter proposals; a failed unanimous accept can trigger another action-selection call. Each round uses 2N to 3N generative calls, plus batched semantic embeddings and periodic memory summarization. The paper says embeddings use `text-embedding-3-large` and model agents are accessed through OpenRouter; the largest experiments use Qwen3.5-397B-A17B. This is not locally reproducible within Tacit's current compute limits, and those additional calls/services must not be treated as free in a cost comparison.

The strongest transfer is methodological: compare a new message representation under the *same* routes and turns first. Then separately compare full end-to-end systems, allowing routing to adapt but charging for proposals, replies, embeddings, memory summaries, retries, and duplicated context. Reuse of HiddenBench as a task family does not make results directly comparable when models, prompts, interaction protocols, and scoring conditions differ.

## New challenge: normalize reasoning computation

Tran and Kiela, [*Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets*](https://arxiv.org/abs/2604.02460) (arXiv, 2026-04-02; v2 2026-04-11), argue from the Data Processing Inequality that with a fixed reasoning-token budget and perfect context utilization, a single agent is more information-efficient; their experiments across Qwen3, DeepSeek-R1-Distill-Llama, and Gemini 2.5 report single-agent systems matching or outperforming multi-agent systems on their multi-hop tasks under matched budgets. They also report budget-control artifacts for some API settings and benchmark artifacts.

This does not imply that agents never help: their stated prediction is that multi-agent systems become competitive when single-agent context utilization degrades or when additional compute is spent. It does mean that “quality per message token” is not enough. A communication result must distinguish at least:

- **Channel budget:** delivered payload bytes and receiver-tokenizer payload tokens, including framing, protocol instructions, and repeated history.
- **Inference budget:** every sender, receiver, router, summarizer, and retry input/output token, with model identity and call count.
- **Task-compute condition:** whether agents have disjoint private evidence, context limits, or a multi-step environment that makes distributed access useful.

For Tacit's core representation comparison, report both the channel-cost frontier and a matched-total-inference-token analysis. A protocol that sends fewer tokens but spends more calls or hidden prompt tokens has not established lower total inference cost.

## Consequences for Tacit's next experiments

1. **Do not label topology improvements as a language result.** Keep topology/schedule frozen for the primary representation contrast; put dynamic routing in a separate system track.
2. **Retain optimized NL and compact structured text as baselines.** Any learned or hand-designed representation must beat them at equal delivered-channel budgets and remain competitive after full inference cost is counted.
3. **Use an information-partitioned task with a verified no-message prior.** The capability gate must not select evaluation episodes; preserve every preregistered held-out task and report full-information controls separately.
4. **Report quality against two budgets.** Plot success against (a) delivered channel bytes/tokens and (b) complete inference input/output tokens, with latency separately. Do not collapse these into one scalar without a declared deployment utility function.
5. **Falsifiable prediction:** if an apparent compact-language advantage is primarily a routing or extra-compute effect, it will shrink or disappear when the routes, turns, and total inference-token budget are held fixed. If it remains and transfers to an independently trained receiver/model family, the representation hypothesis gains support.
6. **Capability fit:** the new large-model results motivate the baseline design but do not justify another local model run. Current host/resource gates and the user's concern about system lag remain in force; literature review and offline code are the feasible work until those conditions change.

## Interpretation

The new evidence narrows the research question. The candidate contribution cannot simply be “agents coordinate better using a new protocol.” Existing work already studies adaptive communication graphs, shared context, and task-conditioned protocol selection. Tacit must identify whether a stable, reusable, transferable *representation* moves the complete frontier after controlling for routing and compute, or else state a different contribution precisely. Current local experiments do not yet answer that question.
