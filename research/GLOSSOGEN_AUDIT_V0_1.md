# GlossoGen audit and project-boundary update

**Reviewed:** 2026-09-29  
**Primary sources:** [Stengel-Eskin et al., *GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions*, arXiv:2609.01491](https://arxiv.org/abs/2609.01491) ([HTML](https://arxiv.org/html/2609.01491)); [official GlossoGen repository](https://github.com/agencyenterprise/GlossoGen)  
**Status:** paper and public platform reviewed; no code imported and no experiment run.

## Why this changes our prior

GlossoGen is a direct, recent answer to several broad claims in this project brief. Its SaveVeyru scenario has complementary private information, an action-grounded communication need, a character-metered channel, language change over repeated interactions, postmortem convention negotiation, and newcomer transmission. It reports emergent protocols with productive morphology and non-zero decoding/production of novel forms. In swap experiments, newcomers learn from message/action history without seeing the postmortem definitions; more history improves task success, and weaker open-weight models can sometimes learn protocols they could not originate. The authors also report that postmortem convention work and sufficient model capability are important for novel protocol emergence.

This means Tacit must not claim novelty for “LLMs inventing a language,” convention negotiation, productive forms, or learning a protocol from use. The [CLSR audit](CLSR_AUDIT_V0_1.md), [AutoForm audit](AUTOFORM_BASELINE_AUDIT.md), and [emergent-communication utility audit](EMERGENT_OOD_UTILITY_AUDIT_V0_1.md) remain relevant, but GlossoGen is now the closest systems-level comparator and an existing artifact worth testing or integrating with.

## What the paper establishes

- **A richer communication setting:** SaveVeyru uses role-specific observations and actions; one party observes symptoms while another holds a changing symptom-to-treatment map. A round can require repeated diagnosis, messages, and actions, so this is more than a one-shot referential game.
- **Language evolution under pressure:** the paper reports 120 proprietary-model runs across model, budget, and postmortem conditions, plus open-weight model screens. Tight character budgets together with a free postmortem discussion stage can produce less English-like, more successful codes for the tested proprietary agents. Open-weight agents in the tested configurations did not show the same emergence and also had lower task success.
- **Structure and transmission:** grammar induction and novel-form probes find productive patterns; swap experiments show that usage history can transfer a protocol to a newcomer. A newcomer may ask metalinguistic questions and repair misunderstandings, which makes interaction and negotiation part of the communication system.
- **A compression-related diagnostic:** the paper's two-part description length separates grammar cost from data cost under an induced grammar. On 45 runs, data coding length is negatively associated with success, while grammar complexity alone is not significantly associated. Grammar induction uses an LLM annotator, and the first usage round helps refine the grammar; later usage rounds are used for the reported held-out data coding length.
- **A usable research platform:** the public GlossoGen implementation records agent interactions and supports scenario-specific tools, channels, swapping, and post-hoc metrics. Rebuilding those platform features inside Tacit would be wasteful unless a measured gap requires it.

These are the paper's reported results, not an independent replication. The paper is an arXiv preprint at the time of this audit.

## What remains a different estimand

GlossoGen asks whether protocols can evolve and support performance in a multi-round, dynamic, action-grounded scenario. Tacit's v0.4 fixture asks a narrower controlled question: can a frozen representation let a sender communicate an unseen higher-order composition to a separately prompted receiver, under a fixed one-message schedule, and how does exact success trade against model-token, byte, setup, and latency cost across strong alternative representations?

The distinction is useful only if we hold it honestly. GlossoGen already tests novel-form decoding and newcomer transmission, so “held-out compositional utility” by itself is not sufficient novelty. The more specific open comparison is an exact, matched-budget receiver-utility frontier with a predeclared candidate distribution, strict end-task scoring, strong tuned natural-language/format/code controls, explicit protocol learning and amortized setup cost, and cross-model transfer under controlled prompt and call schedules. We should compare against or build an adapter for GlossoGen rather than present Tacit's runner as a replacement platform.

## Comparison caveats and opportunities

1. **Different cost units:** SaveVeyru charges one simulated second per message character. This gives the task a common pressure variable, but it is not tokenizer-native inference cost, UTF-8 transport bytes, complete serialized channel bytes, or compute/latency accounting. Tacit should retain these as separate axes rather than convert them into a single “compression” claim.
2. **Free negotiation is still a cost choice:** postmortem discussion does not spend the simulated in-round character budget. For a deployable protocol, induction, negotiation, examples, and transfer data have to be reported separately and amortized over a declared reuse horizon.
3. **Protocol creation and protocol learning differ:** GlossoGen shows that models may learn a convention from usage even when they do not invent one. Tacit's first experiment should distinguish (a) online co-adaptation, (b) a frozen explicit card, and (c) usage-only newcomer transfer; success in one cannot stand in for the others.
4. **Ground truth is unusually available:** the scenario exposes environment events and action outcomes for post-hoc language analysis. Tacit's evaluator-only ledgers provide similarly explicit meaning labels in a synthetic ontology, but the result must remain scoped to that task family and not imply natural scientific or coding-task transfer.
5. **Analysis tooling is not a baseline:** GlossoGen's induced grammar and description-length analysis provide a methodological reference, not evidence that a compact protocol beats a tuned English, structured-text, symbolic, or holistic code at equal complete cost.

## Platform feasibility check (2026-09-29)

The current official README and docs make a local reuse path technically plausible, but they also reveal concrete resource and estimand limits:

- The platform can be installed as a dependency from a pinned Git tag (the docs require v0.1.16 or later for the scenario entry-point contract) and accepts a `self-hosted` provider backed by any OpenAI-compatible chat-completions endpoint. Per-agent model overrides and agent-swap/fork flows are supported. This repository's interpreter is Python 3.13.5, which meets the documented Python ≥3.12 requirement. These are compatibility facts only; the package is not installed here.
- The paper's transfer intervention is directly expressible: develop a convention with the original team, replace one role, give the newcomer only the prior 0/1/5/10 rounds of usage history and action outcomes, hide the postmortem definitions, then measure the swapped team's next 11 rounds. This is a useful *external sequential-transfer* validation for H6; it cannot replace the v0.4 exact-scored, one-message representation comparison.
- Every agent turn is a model call that sends the accumulated conversation; cost therefore grows faster than round count. The current docs say there is no platform-level spend cap. A local adaptation would need frozen low values for `round_count`, `max_round_duration_seconds`, `max-agent-turns`, and `agent_max_tokens`, plus a runner-side hard request ceiling and per-result checkpoints before any service starts. Serial local inference also makes multi-agent rounds slow. A failed run must stop without relaxing the machine gate.
- The default debrief/postmortem messages are free against the simulated per-round character budget. They remain real model calls and channel traffic, so a complete-cost comparison must meter them separately and amortize them; the existing simulated character budget cannot be converted into tokenizer tokens, UTF-8 bytes, or compute. The paper's SaveVeyru outcome also uses a judge, so a local adaptation should first inspect whether a deterministic scoring scenario can preserve the same protocol-transfer estimand.

**Decision:** do not install or execute the platform as part of the current local turn. The already-installed v0.4 runner is the controlled task; a GlossoGen-compatible scenario/adapter is a conditional second-stage transfer experiment after the primary model capability and resource gates pass. The platform's official [README](https://github.com/agencyenterprise/GlossoGen), [simulation guide](https://agencyenterprise.github.io/GlossoGen/latest/running-simulations/), [swap guide](https://agencyenterprise.github.io/GlossoGen/latest/agent-swaps/), and [local-model guide](https://agencyenterprise.github.io/GlossoGen/latest/local-inference-vllm/) were reviewed on this date. No code was copied or installed.

## Updated falsification gate

Before implementing protocol invention or describing a Tacit language, we must answer these questions with a model-free or literature-supported design:

1. Can GlossoGen's public task/platform be used as an independent sequential transfer validation, while v0.4 remains the controlled exact-utility task? If not, what concrete limitation justifies a new platform?
2. Can each representation be compared on both a shared serialized-byte budget and recipient-native tokenizer costs, with all card discovery, negotiation, examples, and inference calls charged separately?
3. Does a frozen protocol beat **development-optimized** natural language, AutoForm-style format selection, valid structured text, compositional code, and a training-only holistic dictionary on held-out receiver utility, including an unseen receiver?
4. Does any gain survive usage-only protocol transfer without a free uncharged definition channel?

If the first three conditions cannot be met on locally available models/resources, the project should prioritize a benchmark/accounting/transfer artifact or report a negative result. Do not name a protocol a new language based on unfamiliar strings, high English perplexity, TopSim, or short messages alone.
