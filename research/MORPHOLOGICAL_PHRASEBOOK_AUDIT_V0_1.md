# Morphological phrasebook audit v0.1

**Status:** primary-source literature audit; no code was imported and no model was run.

## Source and scope

Boldt and Mortensen, [“Communicating in Emergent Language with an Induced Morphological Phrasebook”](https://aclanthology.org/2026.acl-long.1389/) (ACL 2026), test whether a form–meaning inventory induced from a learned emergent language can be used by explicit senders and receivers. Their public [implementation](https://github.com/brendon-boldt/morphological-phrasebook) builds on EGG and CSAR and is MIT-licensed.

This is adjacent methodology, not an LLM communication result. Its core environment is a jointly trained neural signaling/reconstruction game: an encoder sees a discrete tuple, sends up to eight symbols, and a receiver reconstructs it. The main example uses four attributes with four values. Online agents are jointly optimized; offline agents learn from annotated message/meaning corpora; phrasebook agents use rule-based algorithms over CSAR-induced morphemes. The paper reports roughly 1,200 seeds in its primary comparison and 100 runs for environment ablations.

## Findings relevant to protocol research

- In situ phrasebook sender/receiver behavior made the induced form–meaning inventory testable. Phrasebook agents often generalized well in the studied setup, although they underperformed the online neural pair and sometimes struggled to fit training data.
- Ablations show that repetition and morpheme order can carry meaning. A bag of unique symbols would miss these properties.
- The authors propose *morpheme bijectivity*: prevalence-weighted normalized pointwise mutual information (NPMI) over induced form–meaning pairs. On their stratified environment-ablation sample, it predicts phrasebook-agent accuracy better than corpus TopSim (reported R² = 0.85 versus 0.57); bag-of-symbols disentanglement is lower (R² = 0.29). This measures compatibility with their induced phrasebook agents, not a universal language-quality score.

The authors identify material limits: the phrasebook algorithms are heuristic; qualitative analysis of neural strategies is limited; the empirical environments are narrow; and morpheme bijectivity/CSAR currently assume discrete decomposable observations. The studied agents are not LLMs, and the paper does not measure a communication/inference cost frontier or complementary private knowledge.

## Consequences for Tacit

1. Keep task performance and independent receiver use primary. Structural proxies (TopSim, token/meaning mutual information, syntax scores) may diagnose a protocol but cannot substitute for held-out task success or semantic fidelity.
2. For a protocol with induced primitives, test the proposed encoder/decoder directly on unseen primitive combinations. Ablate order and repetition only when the protocol representation makes those mechanisms meaningful; do not assume a bag-of-symbols view is adequate.
3. A shuffled meaning/message-pair control is essential for usage-only onboarding: it preserves the message multiset and exemplar meanings while destroying the association. Above-chance transfer with correct pairings but not shuffled pairings is stronger evidence that the receiver learned the convention. It is still not, alone, proof of compositionality; the same receiver must generalize to held-out compositions and be scored on end-task utility.
4. Tacit's existing v0.4 onboarding design already includes a shuffled-association control and candidate-disjoint held-out meanings. This paper supports preserving that control and separating it from the fixed-card decoder condition. It does not license changing the primary outcome to NPMI or claiming an LLM result.

## Reproduction decision

Do not vendor or run the paper's EGG/CSAR stack as the next project step. It studies jointly trained RNN agents and discrete tuple reconstruction, whereas Tacit's open question concerns pretrained LLM endpoints, private-information utility, heterogeneous receivers, and full message plus inference cost. The public repository's environment is also managed with micromamba/conda and nested source repositories; a faithful reproduction would be a separate historical baseline, not a prerequisite for the current LLM experiment. Revisit it only if Tacit obtains an emergent discrete protocol corpus whose primitive meanings can be independently annotated.
