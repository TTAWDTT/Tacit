# Emergent-language generalization prior audit v0.1

## Question

Does a held-out-composition task or an independent receiver test constitute a
new research contribution for Tacit? The literature says no. Both are established
evaluation ideas. The remaining question is narrower: what evidence exists for
LLM endpoints, under realistic communication and inference costs, using a
protocol that transfers across receivers or model families?

## Prior work

### Mu & Goodman (NeurIPS 2021)

[*Emergent Communication of Generalizations*](https://proceedings.neurips.cc/paper/2021/hash/9597353e41e6957b5e7aa79214fcb256-Abstract.html)
argues that single-object reference games invite context-specific shortcuts.
It introduces set-reference and concept games in which an agent communicates
about sets of objects, with optional distinct contexts. On ShapeWorld, concept
games include primitives and conjunction/disjunction concepts, with 20% of
concepts held out; the paper evaluates whether agents generalize beyond the
trained concepts. This is much closer to receiver-side semantic generalization
than a signal-generation-only diagnostic.

The paper's set-reference games use groups of positive and negative examples,
and the concept games let teacher and student see separate examples. These
designs test abstraction and generalization more deeply than a single
four-choice referential episode over three named attributes. The small Tacit
split must not be presented as a replacement for them.

### Kobrock et al. (Findings ACL 2025)

[*Agents generalize to novel levels of abstraction by using adaptive linguistic
strategies*](https://aclanthology.org/2025.findings-acl.455/) studies a
concept-level reference game. It reports asymmetric zero-shot transfer: agents
use compositional strategies when generalizing from generic to specific
concepts, and reuse multiple training messages when moving from specific to
generic concepts. This cautions against requiring one fixed compositional code
as the only successful form of generalization.

### Chaabouni et al. (ACL 2020)

[*Compositionality and Generalization in Emergent Languages*](https://aclanthology.org/2020.acl-main.407/)
reports that generalization and measured compositionality are not correlated
in their setting, while compositionality helps new learners acquire a protocol,
including learners with different architectures. Therefore measure semantic
transfer and structural composition separately; do not use one as a proxy for
the other.

## Consequence for Tacit

The newly added [`emergent_ood_v0_1`](../experiments/emergent_ood_v0_1/README.md)
fixture is a small deterministic split control, not a novel benchmark. It
isolates third-order combinations by training all unary values and pairwise
combinations, but does not instantiate a sender/receiver, reproduce the richer
concept/set tasks, or test an LLM. Keep it only as a transparent unit fixture
for split and scorer validation.

Do not claim that held-out receiver utility or compositional generalization is
unexplored. A defensible Tacit question, pending a broader search, is whether
direct LLM endpoints can discover/use a stable representation that improves a
complete communication/inference frontier and transfers to an independent or
heterogeneous receiver, compared with the established concept-level, text,
structured, adaptive-format, holistic-code, and latent alternatives. This is a
candidate gap, not yet a novelty claim. Benchmark selection should first test
whether existing concept/set tasks or current multi-agent task suites can
answer it without a redundant synthetic game.

## Falsifiable implications

1. A receiver's exact held-out task success must be measured directly; signal
   generation, TopSim, and message-level correlations are diagnostics only.
2. Generalization should be measured both within a protocol-trained receiver
   and across independently initialized / heterogeneous receivers. Transfer can
   improve with composition, but a compositionality score alone does not predict
   all forms of generalization.
3. Compare multiple levels of abstraction and candidate difficulty. A protocol
   that transfers from broad to specific meanings may fail in the reverse
   direction.
4. Report total protocol-learning or prompt-selection cost, message bytes,
   native receiver tokens, decoding/inference cost, and reuse horizon. Existing
   concept-game results do not automatically establish an LLM deployment-cost
   advantage.

## Sources

- Mu & Goodman, NeurIPS 2021: [paper](https://proceedings.neurips.cc/paper/2021/hash/9597353e41e6957b5e7aa79214fcb256-Abstract.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/9597353e41e6957b5e7aa79214fcb256-Paper.pdf).
- Kobrock et al., Findings ACL 2025: [paper](https://aclanthology.org/2025.findings-acl.455/).
- Chaabouni et al., ACL 2020: [paper](https://aclanthology.org/2020.acl-main.407/).
