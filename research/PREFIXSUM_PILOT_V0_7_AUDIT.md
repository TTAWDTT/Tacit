# PrefixSum v0.7 implementation audit (not a valid protocol comparison)

## Status

The v0.7 run was stopped after 59 of 84 task-condition cells. Those traces are retained locally in ignored `.cache/pilot_v0_7/`. No aggregate outcome report is published because the upstream message parser changed the actual wire representation for three conditions, and an overlarge parsed integer crashed logging. Treat all v0.7 task-success and payload comparisons as invalid for protocol ranking.

The frozen experiment was intended to test Prefix Sum with Qwen3-4B Q4_K_M/llama.cpp across seven message conditions. During trace inspection, two independent implementation problems made it impossible to interpret as planned.

## Wire-content coercion

At the pinned upstream Silo-Bench revision `e74127782ed1c42fff474249961f022c063d76f2`, `src/utils/parsing.py` applies `_convert_value` to every XML parameter. It tries integer conversion and JSON decoding without considering the tool or parameter name. The `send_message` tool subsequently calls `str(content)`.

This changes message payloads:

- An all-digit binary string such as `101` becomes the integer 101, then is sent as decimal text `101`.
- A valid JSON object such as `{"s": 123}` becomes a Python dictionary and is sent through Python's `str(dict)`, producing single quotes and invalid JSON.
- A JSON list for the full-shard arm becomes a Python list and is serialized with Python representation instead of preserving the original JSON text.
- In one binary run, an extremely long digit string parsed as a large Python integer; upstream `ujson` raised `OverflowError` while logging the tool call. This terminated the runner before it wrote the aggregate CSV.

These behaviors violate the frozen intervention semantics, so JSON, binary, and full-shard message measurements cannot be interpreted as those representations.

## Coordination failure

Trace review also found repeated role reversal: the common prompt asked Agent 0 to send the needed offset to Agent 1, while Agent 1 should receive; instead, Agent 1 often sent its own subtotal to Agent 0. Agent 0 then submitted locally, and Agent 1 never obtained the prior-segment offset. On longer segments, Agent 0 also made cumulative-sum errors. Since neither the sender/receiver role nor the message path was reliably executed, task success cannot diagnose the representation conditions.

## Correction

The initial 59 rows are retained as a failed implementation audit, not merged into a corrected dataset. The next protocol uses fresh task seeds, a project-local tool adapter that preserves `send_message.content` as the exact raw string, and role-specific task prompts that require Agent 0 to send before submitting and Agent 1 to receive before computing. It will freeze those changes publicly before any further model call. The message parser and scorer will separately record content syntax, sender-value fidelity, direction, and final list accuracy.

This failure reinforces an engineering requirement for a real communication SDK: payload typing and serialization must be explicit end-to-end. A representation is not evaluated if the transport silently coerces it.
