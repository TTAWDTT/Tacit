# PACT and Proxifield: representation–routing confound audit

**Reviewed:** 2026-09-30  
**Sources:** [PACT paper](https://arxiv.org/abs/2606.05304), [PACT code](https://github.com/iNLP-Lab/PACT), [Proxifield paper](https://arxiv.org/abs/2609.20889)  
**Status:** primary papers and public PACT repository reviewed; no code imported, no benchmark run.

## Finding

Two recent systems expose a design hazard for Tacit's central question. PACT changes the *content admitted to shared agent history*: it tests full output, concise generation, conclusion-only, brief summary, artifact-only, then proposes compact action-state records. Its split-evidence tasks require agents to exchange complementary evidence, and its coding-harness results show that action-state filtering can improve or preserve success while reducing repeated context. Its authors also report that no one of the common fixed strategies is best everywhere. This is direct prior art against a broad claim that a newly named content format alone makes LLM communication efficient.

Proxifield changes the *communication graph*: it routes proposals using direct address, information needs, plan alignment, and information complementarity, then exchanges proposal/reply/commit messages. Its HiddenBench tasks distribute private evidence, so this is genuinely communication-dependent. Its reported agent-count and failure-resilience results concern coordination topology; it does not isolate a new semantic wire language. Its appendix reports API-cost estimates from black-box usage accounting, and the paper notes that Proxifield can cost roughly two to four times the original naive Discussion protocol on HiddenBench. These are system-level findings, not a direct matched representation frontier.

The layers are conceptually distinct but not automatically experimentally independent. Proxifield computes routing similarities from message, rationale, observation, and memory embeddings. Replacing the message representation can change which peers are selected. Conversely, changing topology changes who receives a representation and what context they can combine it with. A single end-to-end score cannot identify whether a gain came from message encoding, routing, or their interaction.

## Formal estimand

Let `c` denote content policy/representation, `r` routing policy, and `Y(c,r)` a preregistered task outcome on paired episodes. For a fixed routing policy, the representation effect is

`Δ_C(r) = E[Y(c₁,r) − Y(c₀,r)]`.

For a fixed content policy, the routing effect is

`Δ_R(c) = E[Y(c,r₁) − Y(c,r₀)]`.

The interaction is

`I = [E Y(c₁,r₁) − E Y(c₀,r₁)] − [E Y(c₁,r₀) − E Y(c₀,r₀)]`.

If `I ≠ 0`, reporting only an averaged “language effect” hides that the representation's utility depends on the routing policy. These contrasts apply separately to task success, semantic fidelity, delivered-message bytes, each recipient's tokenizer cost, model input/output tokens, calls, latency, and inference compute; do not collapse those quantities into a post-hoc scalar.

## Predictions that can fail

1. **Content effect under fixed routing.** At a fixed schedule and identical receiver-visible non-message context, PACT-style action-state filtering should reduce accumulated receiver input tokens on long-history tasks. It need not improve exact private-fact fidelity or held-out task success; the representation is useful only if downstream utility survives.
2. **Routing effect under fixed content.** On tasks where private evidence has sparse recipients, a semantic router may improve success or reduce irrelevant delivered context versus star/shared context. If routing metadata generation, embeddings, replies, commitment calls, and repeated context erase those gains, complete cost per solved task will not improve.
3. **Interaction.** A content representation that omits rationale, needs, or evidence references may reduce the semantic signals Proxifield uses to route. Therefore its value may fall under semantic routing relative to fixed star. This prediction is falsified if the paired interaction interval is practically equivalent to zero under the prespecified margin.
4. **Scaling.** Any claimed benefit must be traced as task size and agent count grow, while separately charging edges, calls, context replay, router embeddings, and setup. A sparse graph can reduce recipients while still increasing model-call or per-round overhead.

## Experimental design consequence

Use two stages, and only call the second a factorial experiment if the interface can be controlled:

### Stage A: identify each main effect

- **Representation arm:** freeze topology, recipients, turn schedule, model, prompts outside the payload slot, episode IDs, and receiver context. Compare tuned natural-language controls, PACT/action-state, structured text, code, and any Tacit candidate. Preserve a private evaluator trace to score exact fact/evidence fidelity independently of final task utility.
- **Routing arm:** freeze the message schema/content policy and compare no-message, star, shared context, and a Proxifield-compatible router. Charge all proposals, replies, optional commits, embedding calls/compute, memory summaries, retries, and delivered bytes. This answers a coordination question, not a language question.

### Stage B: estimate interaction (conditional)

Cross the representation and routing conditions on the same frozen episodes only if routing receives a separate, invariant metadata view (for example, the same natural-language intent/need fields in every arm) while the exchanged payload alone varies. Otherwise the experiment changes both the payload and the graph at once; record this as a coupled system comparison and do not interpret its contrast as a pure representation effect. A useful factorization is `private payload → representation → delivered message`, alongside `routing metadata → graph`; explicitly specify whether the representation is permitted to alter routing metadata.

Use paired episode-level contrasts, held-out task templates, and heterogeneous receivers as transfer tests. Report empirical intervals and cost-success frontiers, including setup amortized over declared reuse horizons. Count an unoptimized or hand-written English sentence only as a weak control: PACT and task-specific natural-language optimization are stronger controls already required by the experiment plan.

## Baseline applicability gate

These papers are important prior art, but they are not plug-in baselines for every Tacit fixture. PACT targets systems where agent outputs and tool results accumulate in shared history or where a downstream role needs an action/state handoff. Tacit's v0.4 controlled task is a single sender message for a held-out candidate-selection meaning, with no evolving public state or repeated-history replay. On that fixture, a generic PACT filter would change the task or remove irrelevant fields rather than compare equivalent representations. Use an explicitly defined action-state natural-language condition only when the task semantics contain an action/state update; otherwise compare task-appropriate natural-language, structured, and compositional encodings under the fixed payload slot.

Likewise, Proxifield is relevant when there are multiple eligible recipients, meaningful information needs, and repeated decisions about who should receive a message. A two-role one-shot task with an exogenously fixed receiver cannot identify a routing advantage. Do not expand the first benchmark merely to host an inapplicable baseline. For a later multi-round, multi-agent validation, reuse or adapt a public routing benchmark/platform and compare topology only after its task and cost contract have been audited.

Thus the design gate is two-part: (i) cite strong adjacent methods in the project thesis, and (ii) include them empirically only when their intervention is semantically valid for the task. Document exclusions before data access, with the mismatch stated explicitly. This avoids both ignoring prior work and creating a strawman comparison.

## Scope and feasibility

PACT covers action-state content selection and existing harness integration. Proxifield covers adaptive sparse routing and sequential coordination. Tacit should neither claim those ideas as novel nor use their system gains as evidence for a message language. A defensible remaining question is whether a reusable, compositional representation improves held-out receiver utility and complete cost **conditional on a fixed information-access/scheduling policy**, and whether any gain survives cross-model transfer and protocol onboarding.

Proxifield's reported studies use API-served models up to 397B parameters and batched semantic embeddings; its authors list limited domain coverage and a model-family anomaly as limitations. This makes its published result valuable prior evidence but not a promise of local reproducibility. The current Tacit resource gate remains authoritative; this audit performed no local inference or model loading.

## Sources and evidence limits

- PACT reports five common strategies across complementary split-evidence interaction and sequential pipelines, and an action-state protocol; the authors report SWE-bench Verified results with Qwen3-14B. These are paper-reported results, not independently reproduced here.
- Proxifield reports a dynamic routing graph based on four semantic/direct-address signals and evaluates Drone Search and Rescue plus HiddenBench. Its scaling/resilience claims are reported by the authors, not independently reproduced here.
- The audit uses the papers' published methods and cost descriptions; it does not establish that either baseline wins under Tacit's prompts, models, tasks, or accounting boundary.
