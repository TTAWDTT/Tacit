# v0.17 stopped after parser failure

The pre-registered local CRAFT feasibility run was stopped after its first completed turn (4 model calls; one medium structure; run seed 317). This is an implementation-failure report, not a task-performance result.

All three Directors generated ordinary bracketed natural-language responses. The pinned CRAFT parser's plain-text fallback removes every `[...]` span as though it were an instruction echo, so all three parsed `public_message` values became `No message provided`. The Builder transcript filter consequently delivered zero Director messages. The Builder still selected an oracle candidate and the environment reported progress `0.037`; this is not evidence of communication success.

During oracle candidate generation, the upstream code also attempted to print a `↔` character to a Windows GBK console and logged a Unicode encoding error. It did not prevent the first Builder move, but it is a reproducibility defect for this host.

The output log is retained locally under ignored `.cache/pilot_v0_17/runner.log`; it contains only one completed turn and no whole-game JSON because the process was interrupted after the confirmed zero-delivery failure. We did not continue spending inference on rounds that could not deliver messages. No protocol or task-success conclusions are drawn. A new run requires a new frozen adapter that preserves bracketed model output and forces UTF-8 console I/O.
