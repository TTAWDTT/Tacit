# Multi-turn communication-budget source note (2026-09-29)

## Scope

Checked the primary benchmark records and artifact pages while reviewing the new fixed-schedule SDK. The question was whether a multi-turn communication runtime should merely report bytes after the run or enforce a shared aggregate channel budget during delivery.

## Findings

- Eisenstein et al., [MT-PingEval](https://arxiv.org/abs/2602.24188), defines an isotoken evaluation that keeps a per-player communication-token cap fixed while varying the number of turns. The paper reports that extra turns often do not improve task success and may expose dialogue-planning failures. Tacit's existing audit records the additional caveat that its budget is whitespace-counted, not a full inference-compute or model-native-token budget. This supports hard budget enforcement while keeping inference costs as a separate ledger.
- Zhang et al., [SILO-BENCH](https://aclanthology.org/2026.acl-long.1354/), evaluates exact-scored distributed tasks with success, partial correctness, generated-token consumption, and communication density. Its scope is coordination topology and scale, not an isolated representation comparison. Its [official repository](https://github.com/jwyjohn/acl26-silo-bench) describes multiple communication infrastructures and reports generated token consumption separately from inter-agent interaction intensity.
- Yang et al., [MAS-BENCH](https://aclanthology.org/2026.findings-acl.1698/), adds explicit distributed-sorting communication constraints and diagnoses shared-state, convention, and termination failures. It is further evidence that interaction reliability is distinct from message encoding; it does not replace a fixed-schedule, byte-accounted codec study.

## Design consequence for Tacit

The multi-turn SDK now optionally enforces an aggregate **application-wire byte** limit over the exact JSON payload, envelope, length prefix, and acknowledgment it measures. A message that cannot fit is not delivered. If its generating model call already occurred, that call remains in `model_call_records()` while it contributes no transmission record or channel bytes. If no possible message can fit, the SDK stops before making another model call. The cap excludes TCP/IP headers and does not cap model-native tokens, prompt tokens, or inference compute; these dimensions remain separately reported.

This is a measurement/runtime improvement, not a benchmark result or superiority claim. The literature does not make fixed-budget evaluation novel; a Tacit result still has to isolate representation under matched task, information access, schedule, model calls, and decoder, with strong baselines and transfer tests.
