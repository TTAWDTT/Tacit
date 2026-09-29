# Cheap talk, hidden-state transfer, and coordination value v0.1

**Status:** source audit and theory update; no model or external service was run.

## Source and result

Madmoun & Lahlou, [*Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches* (EACL 2026)](https://aclanthology.org/2026.eacl-short.23/) evaluates a repeated four-player Stag Hunt with a one-word, non-binding broadcast before each action. The paper's appendix specifies three pilot rounds and says agents may send any one word. In its heterogeneous four-model group, cooperation rises from 0% without communication to 48.3% with it. The authors report that `stag` becomes a focal signal in 73% of messages during cooperative rounds. In same-family coalitions, the no-message cooperation rate is already 52.2%; cheap talk primarily reduces the risk and variance of coordination failure rather than increasing the cooperation rate.

The same paper shows why an action-rate metric is not a utility metric. In its standard 1.6× public-goods-with-punishment setting, communication increases contribution from 48% to 71% while average payoff falls from 184.4 to 127.5. At 4× incentives, communication increases contributions from 55% to 100% and average payoff from 457.9 to 480.0. These are the authors' reported outcomes, not a Tacit replication. The open-model cohort uses four hosted instruction-tuned models at temperature 0.7; the Stag Hunt is a common-knowledge coordination game and does not distribute complementary private task facts.

## Interpretation and limits

The Stag Hunt result is evidence that LLMs can converge on a short action-referencing focal signal in a task with multiple coordination equilibria. It is a strong prior against treating every gain from enabling messages as semantic knowledge transfer. The signal `stag` communicates or coordinates intended action under shared payoff knowledge; it does not show that a compact code transmits a sender-only observation to an independently informed receiver.

The public prompt appendix makes the interaction mechanism concrete: each player first emits one arbitrary word, then selects an action after seeing all players' words. This changes what each player knows about the others' intended actions and adds a communication stage. The study reports behavioral outcomes and message content analysis, but it is not a matched comparison of natural language, JSON, symbolic code, or tokenizer-normalized payloads. Its positive cooperation result does not generalize across incentive structures.

## Consequence for Tacit

For exact hidden-state tasks such as Emergent OOD, preserve a receiver-local decision estimand: message meaning must be tied to the sender's private target, and no-message/full-information controls remain necessary. The target-meaning/message shuffle used for onboarding artifacts is not an episode-level counterfactual receiver test. Add a hash-bound, cost-matched episode-level message-association derangement before claiming that the receiver used transmitted semantics. Construct it so a sender message from another episode cannot directly name one of the destination's candidate meanings; if a perfect compatible derangement is unavailable, report overlap and use a prespecified conditional null rather than treating the shuffle as fully irrelevant.

For joint-action settings with shared payoff and coordination externalities, include a separately labeled cheap-talk/intent signal and report task welfare, individual payoff, coordination rate, message cost, and information transfer as distinct outcomes. A rise in cooperation can coexist with a fall in welfare. Do not use that task to rank content codecs unless private-information access, policy, and utility are controlled.

## Formal result and falsifiable test

Theory §18 proves a narrow Bayes decision statement: if a message is conditionally independent of the hidden task state given the receiver's local information, an ideal receiver's Bayes risk cannot improve. Team utility with cross-agent action externalities is outside that proposition; a shared message can coordinate actions without changing the receiver's posterior about the world. Prediction P15 states the matched-permutation test and its required compatibility condition.

The next implementation step is a receiver-only replay interface that consumes frozen sender traces, constructs and hashes a valid state-to-message derangement, and charges each replay call under the same cost schema. It must preserve the same receiver task rows and model settings. It must not issue endpoint calls unless a fresh passing machine preflight, receiver-capability gate, and explicit run ceiling all pass.
