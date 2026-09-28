# PrefixSum receiver failure audit v0.1

This is a post-hoc audit of the frozen Qwen3-14B v0.16 run. It reads the published per-episode ledger and the pinned synthetic task files; it does not load a model, alter the run, or expose raw model reasoning.

## Episode decomposition

| Segment length | Episodes | Sender exact | Correct payload received | Receiver exact | Receiver returned local-only prefix | Wrong numeric vector | Expression string |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 8 | 8 | 1 | 2 | 3 | 2 |
| 3 | 8 | 8 | 8 | 1 | 0 | 6 | 1 |
| 4 | 8 | 8 | 8 | 0 | 1 | 7 | 0 |
| **Total** | **24** | **24** | **24** | **2** | **3** | **16** | **3** |

The mutually exclusive receiver-answer categories are evaluated against each task's ground-truth segment and the inclusive prefix computed from Agent 1's private shard alone. The sender is exact in every episode. Message receipt is not merely a successful tool event: the simulator trace contains the expected sender subtotal in all 24 cases. The receiver nevertheless returns the local-only prefix in three cases, an incorrect numeric vector in sixteen, and an unevaluated arithmetic expression in three.

The separate direct combined-receiver control has another failure split:

| Segment length | Exact | Parseable but numerically wrong | Unparseable |
|---:|---:|---:|---:|
| 2 | 2/8 | 1/8 | 5/8 |
| 3 | 1/8 | 3/8 | 4/8 |
| 4 | 4/8 | 4/8 | 0/8 |
| **Total** | **7/24** | **8/24** | **9/24** |

So direct combined failures are not purely output-format errors: eight are valid parsed answers with wrong values. Conversely, simulator receipt does not guarantee the receiver's final tool submission is a correct numeric value. These are distinct measured failure modes, although this study does not identify their causal relationship.

## Interpretation and falsifiable follow-up

The trace isolates the failure after payload delivery, but does not identify a single cause. The independent direct controls show local-prefix and offset-vector arithmetic at 24/24 each, while a direct combined-receiver call is only 7/24 exact. This pattern is consistent with a composition, prompt/context, or action-to-answer integration bottleneck. It does not show that the `s=<subtotal>` payload is intrinsically ambiguous, and it does not show that another encoding would fix the failure.

The next discriminating study, if local resource conditions later permit model inference, should keep model, examples, total prompt budget, and answer contract fixed while contrasting: (1) current simulator reception, (2) the same message injected as an explicit typed offset field, and (3) a no-message/local-prefix control. It should predeclare event-level outcomes for correct receipt, correct offset use, valid numeric output, and exact final submission. The typed-field arm changes receiver instructions as well as representation unless those instructions are token-matched, so a representation claim requires a matched-prompt control. Do not run this until a fresh frozen resource gate passes; this audit itself authorizes no model calls or format ranking.

This is a post-hoc decomposition of one reused task family and one greedy sample per case, not new generalization evidence.
