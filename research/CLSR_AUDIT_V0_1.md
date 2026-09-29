# CLSR and adjacent protocol prior audit v0.1

**Research date:** 2026-09-29  
**Scope:** direct comparison with recent machine-generated symbolic protocols, multi-order context consolidation, and typed action compression. This is a source audit, not a replication. No external repository was cloned, dependencies installed, benchmark data downloaded, or model run.

## Decision

CLSR is now the closest prior art to Tacit's original “let LLMs invent a machine language” premise. Its Communicative Language Symbolism Routing framework evolves reusable symbolic Language Symbolism Frameworks (LSFs), profiles their cost/accuracy and failure behavior, and selects, ensembles, or composes them at inference time. Therefore, **autonomous invention/evolution of a compact symbolic protocol is not a novelty claim for Tacit**.

The narrower open question is whether a frozen, compositional protocol improves communication between separately situated LLM endpoints that hold complementary private information, especially across receiver/model boundaries and under complete bidirectional communication and inference accounting. CLSR's published tasks primarily measure single-query reasoning accuracy against generated-completion-token cost; they do not establish this hidden-information transfer result. Treat CLSR as a mandatory direct conceptual baseline and methodological template for language lifecycle, while preserving the task distinction.

## CLSR: direct symbolic-protocol prior

Pei, Huang, and Wang, *When LLMs Develop Languages: Symbolic Communication for Efficient Multi-Agent Reasoning*, arXiv:2606.29354 (June 28, 2026; the authors' repository identifies the ICML 2026 publication). The authors define an LSF as a reusable compact lexicon, compositional syntax, and usage/validity constraints. LSFs are induced from training exemplars, evolved by correctness and token cost, profiled on validation tasks, frozen, then routed by an LLM. Inference can select one LSF, aggregate multiple LSF outputs, or compose them across rounds.

The paper reports seven reasoning benchmarks and four open-weight backbones. For Qwen3-8B, examples include MMLU-Pro 60.4 accuracy at 96 average generated tokens versus Raw CoT 60.2 at 276; GPQA-Main 47.7 at 228 versus 49.1 at 1085; and MATH500 86.8 at 257 versus 87.2 at 878. These are the paper's results, not Tacit replications. They make CLSR a serious challenge to any claim that natural-language reasoning traces are already an adequate machine protocol.

### Comparability and cost boundaries

- **Task/estimand:** the headline experiment optimizes a model's reasoning transcript for answering a benchmark query. The paper's formal setup has a query, ground-truth answer, LSF pool, router, and final answer. It is not an instance of two agents with disjoint private views that must exchange a payload to solve a joint task. A Tacit result must test that additional communication estimand directly.
- **Model population:** paper comparisons use the same named backbone for LSF generation/inference strata, with additional generator-to-inference transfer and larger-backbone analyses. This is not proof of robust cross-family sender/receiver transfer. Its appendix reports a generator/inference model swap; Tacit still needs a receiver-held-out or heterogeneous cross-play evaluation.
- **Primary cost:** the main metric counts all online **generated completion tokens**, including router, intermediate, and aggregation outputs. It omits prompt/input tokens from that primary frontier. The appendix acknowledges that router/profile/LSF inputs can be roughly 0.5–2k tokens and introduces a cache-aware token-equivalent diagnostic; its coefficients depend on serving infrastructure and assume reusable-prefix behavior.
- **Protocol setup:** LSF synthesis/evolution and profiling consume model calls and training exemplars. The paper generates initial populations using 200–2,000 training exemplars, then evaluates evolved pools; it reports several days of offline LSF generation/evolution on 8 RTX 4090s. The authors frame this as a one-time cost that can be amortized over many downstream queries, but the headline online frontier does not include it at a declared reuse horizon. The appendix separately gives a cache-aware input-token diagnostic. Tacit's setup ledger and vector break-even analysis should be retained when using an evolved protocol.
- **Strongest method contribution:** CLSR jointly changes representation, protocol selection, number of calls, and reasoning depth. It is a strong whole-system baseline. A representation-only causal claim requires either holding routing/call schedule fixed or reporting a separate complete-system comparison.
- **Reproducibility:** the official `LSF_MDia` repository describes deterministic offline fixtures and an adapter path, but distinguishes those from reproducing a named paper table, which requires exact model revisions, task manifests, evaluators, seeds, and release artifacts. The public repository is MIT-licensed. This audit inspected source text only; it does not claim that the paper's tables were independently reproduced.

### Tacit consequences

1. Update the thesis: compact symbol invention, evolutionary refinement, LSF cards, and cost/accuracy routing are prior art.
2. Do not compare a hand-written symbol table only against English. Include concise NL, valid structured formats, AutoForm, and a CLSR-style frozen reusable dialect when model execution and setup accounting become feasible.
3. Evaluate the dialect on paired hidden-information tasks with disjoint sender/receiver views; require no-message and full-information controls, exact semantic reconstruction, and held-out receivers/models. Keep seed/task instances separate from dialect induction and routing validation.
4. Report separate frontiers for (a) online inference cost and (b) amortized total cost including dialect creation/profile/routing setup. Count input and output tokens per tokenizer, serialized bytes, calls, service/wall time, and any cache assumptions.
5. Because local inference is resource-gated, do not attempt a CLSR reproduction now. First make a model-free design mapping from its lifecycle (induce → validate → freeze → cross-play) to an existing exact communication task; only then propose the smallest eligible local batch.

## Adjacent but distinct baselines

### MOC: multi-order evidence/context consolidation

Guan et al., [MOC](https://arxiv.org/abs/2606.02359), exposes multi-hop evidence and merges similar messages under a context-length constraint. Its contribution mixes message topology, recipient context construction, and generative distillation; it is not a fixed semantic codec. The paper evaluates seven-agent systems with Gemma-2-27B, Qwen2.5-32B, and DeepSeek-V3.2-685B, and reports a 20-agent input-token reduction from 1.338M to 1.249M against 1.369M for vanilla MAS. It reports 79.7 seconds of distillation on a single seven-agent MMLU example at one setting, using Gemma2-9B alongside Gemma2-27B agents on RTX 4090 hardware. This is a topology/context-policy comparator for future long-context, multi-hop studies, not a feasible first local baseline and not an isolated language-format result.

### AACP: typed action/coordination packets

Mackay's [AACP v1.4 Internet-Draft](https://www.ietf.org/archive/id/draft-mackay-aacp-03.html) defines deterministic, pipe-delimited task/domain/action packets for repetitive business workflow instructions, with a rule library and LLM fallback. Its author reports about 23% lower coordination-message tokenization than verbose English and larger total workflow savings in four framework integrations. These are useful practical claims to verify, but the draft is an individual informational Internet-Draft, not an IETF standard, and the reported results are author-reported. AACP encodes typed action intents using shared domain vocabularies; it is not evidence that an agent can compress arbitrary private task facts or transfer a general compositional reasoning language. Include it if Tacit studies delegation/workflow instructions, with registry/setup and complete prompt accounting.

## Primary sources

- Pei et al. (2026), [paper](https://arxiv.org/abs/2606.29354), [full text](https://arxiv.org/html/2606.29354), [official code](https://github.com/pzqpzq/LSF_MDia).
- Guan et al. (2026), [MOC paper](https://arxiv.org/abs/2606.02359), [official code](https://github.com/yao-guan/MOC).
- Mackay (2026), [AACP Internet-Draft v1.4](https://www.ietf.org/archive/id/draft-mackay-aacp-03.html), [IETF Datatracker status](https://datatracker.ietf.org/doc/draft-mackay-aacp/).

## Limits

This review used paper text and repository documentation. It did not inspect every supplementary run artifact, verify reported aggregates, execute code, or independently audit licensing beyond the public repository's stated MIT license. The reported numbers above must remain attributed to their authors.
