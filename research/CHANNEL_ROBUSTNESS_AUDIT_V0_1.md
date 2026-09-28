# Channel robustness and conversational repair audit v0.1

**Date:** 2026-09-29  
**Scope:** distinguish classical transport-channel coding from model-mediated semantic errors; review feedback/repair as a candidate LLM communication policy.  
**Execution:** paper and official code inspection only. The code was not cloned or run; no model or dataset was downloaded.

## Research question

When a receiver fails, does the failure come from a damaged payload, from misunderstanding a delivered payload, or from a mismatch in the receiver's model/context? A checksum or error-correcting code addresses only some transport failures. A clarification round addresses some receiver uncertainty but costs tokens, calls, and latency. A useful comparison has to identify the failure channel and price both directions.

## Classical channel coding versus LLM-agent handoff

Shannon's [channel-coding theorem](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x) states, under a specified stochastic channel model, that codes can make block error arbitrarily small at rates below channel capacity as block length grows. It does not specify which task information should be sent, how an LLM interprets the string, or whether the redundancy improves a downstream decision. For project experiments, record at least three layers separately:

1. **Transport integrity:** delivered bytes, truncation, dropped messages, or injected byte/token corruption.
2. **Semantic fidelity:** whether the receiver recovers the sender's intended task-relevant facts/relations from an intact payload.
3. **Task utility:** whether the final receiver action is correct under the fixed task scorer.

In the usual tool/API message path, transport preserves the submitted string; the dominant uncertainty can therefore be semantic decoding or task execution rather than bit corruption. Do not add classic ECC overhead to an experiment unless a noisy transport is part of the deployment condition. Conversely, do not call an intact but misread message a transport error.

## Conversational repair precedent

Nikolaus, [*Emergent Communication with Conversational Repair*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/27a2b7a22f91245200ebe89e468a1c54-Abstract-Conference.html) (ICLR 2024), compares a unidirectional signaling game to a learned feedback variant. Its public [implementation](https://github.com/mitjanikolaus/emergent_communication) includes toy and GuessWhat game runners. In the basic game, the sender and receiver are GRUs trained jointly; 90% of meaning combinations are train and 10% held out; three random seeds are reported. A noise process replaces sender message tokens with a special noise token at probability \(p_{noise}\), which makes corruption visible to the receiver. A second noise implementation permutes tokens and is harder. The reverse channel has a binary alphabet and sends feedback at each sender step.

The paper finds that feedback improves test accuracy under noise, with the advantage growing through much of the tested noise range, while TopSim falls. It reports similar direction on GuessWhat's natural-image reference task. The authors' message analysis suggests feedback does more than request repetition: it correlates with previous sender messages and receiver-side candidate objects, enabling acknowledgement and joint/contextual construction of meaning. This makes feedback an interaction policy with potential information flow, not merely a fixed error-correcting bit.

**Limitations for Tacit:** trained RNN agents are not LLM endpoints; error tokens are explicit; reverse feedback is noise-free; both parties co-adapt; and channel budget comparisons do not hold total forward-plus-reverse traffic fixed. The paper validates a positive mechanism in its simulation, but not that repair wins after matching communication and inference budgets in LLM systems.

## Falsifiable predictions

1. **Transport-noise prediction:** under injected byte/token loss or corruption, redundancy or a checksum-like structured representation should improve semantic recovery only if it detects/corrects the chosen corruption model. It should provide little or no task benefit on an integrity-preserving transport after its bytes and decode cost are counted.
2. **Semantic-repair prediction:** receiver clarification should help most when the intact message leaves multiple plausible interpretations that imply different task actions. If ambiguity concerns only task-irrelevant details, clarification cost will not buy task utility.
3. **Budget prediction:** any repair advantage must survive matched total bytes, all prompt/completion tokens, calls, and a fixed round cap against a one-pass representation with matched redundancy. Measure quality across budgets rather than comparing unrestricted repair to a capped one-shot message.
4. **Model-mismatch prediction:** a protocol with stable semantics should show lower semantic reconstruction error across receiver models than a co-adapted private code; adaptive repair may recover some utility but its extra reverse traffic and latency should be visible.

## Proposed measurement protocol

On a task that first passes no-message/centralized-oracle communication-need checks, replay each sender-intended semantic payload under:

- intact transport + same receiver;
- deterministic corruption/truncation variants + same receiver;
- intact transport + a different receiver family or prompt/context;
- no repair, fixed redundancy, and receiver-initiated clarification.

Log raw sent/delivered bytes, whether transport changed the bytes, receiver-decoded semantic fields, final task result, and the full per-agent inference ledger. Pair seeds and episodes. Count every clarification token in the same total channel cap; count repeated history in inference tokens and measure wall time. TopSim or syntax validity alone cannot establish recovery.

**Current decision:** keep adaptive feedback/repair as a later policy and robustness baseline. First stabilize a task and receiver condition where message semantics and end-task scoring are reliable. If such a study is run, compare at equal total bidirectional budget and explicitly state which noise model the protocol addresses.
