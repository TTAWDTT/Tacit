# Preregistration: complete the shipped sum-format tokenizer comparison (P26)

**Status:** frozen before counting. Model-free tokenizer and transport accounting only.

## Question

The corrected P24 audit included decimal, JSON, binary text, and no-message, while the shipped SDK exposes five communicating formats. What are the full prompt/output content-token and application-byte costs of the remaining labeled and fixed-sentence formats on the same task vectors?

## Frozen method

- Reuse the exact P24 Qwen3-4B tokenizer revision/hash and SDK accounting boundary.
- Reuse `run_sum_episode`'s `_role_instructions` constructor for both formats, exact public/private contexts, exact task, final-answer instruction, and runtime user JSON.
- Enumerate all 4^m input vectors for m=2,3,4. For labeled, transmit exactly `v=N`; for sentence, exactly `My private integer is N.`. Receiver completion is the exact sum under ideal faithful communication.
- Use `LocalTCPMessageChannel.measure` with the shipped protocol IDs for exact application-byte counts. Report prompt inputs, completions, total known content tokens, calls, bytes, and 100% oracle success separately.
- Compare with the already published P24 decimal, JSON, and binary rows by same `m`. Do not rerun or revise those conditions.

## Interpretation

This fills coverage of shipped fixed formats only. It does not optimize English prompts, measure model adherence, or establish a generally strong natural-language baseline. No prompt will be called “optimized” without a frozen development-only search budget and independent validation. Chat templates, provider-added tokens, inference cost, and latency remain outside the count.
