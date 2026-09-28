# `INDEX_m`: a communication-complexity control for LLM agents

## Why add this task family

The recent HiddenBench runs have separated local reasoning ability from information sharing, but the Qwen3-14B run is expensive and the three-round natural baseline remained at zero on one hidden task. PrefixSum exposed arithmetic and receiver-operation failures before a language comparison became eligible. CRAFT exposed a different issue: privileged oracle-candidate actions can make no-message completion possible. A useful next task should provide an exact communication requirement, a cheap single-agent capability control, a known no-message baseline, and a tunable difficulty parameter.

`INDEX_m` is a narrow control task for those purposes. It is not proposed as a replacement for real coordination benchmarks and does not prove that a code transfers to scientific reasoning, planning, or natural language tasks.

## Task definition

For each episode:

- The sender sees a private uniformly random bit vector `x ∈ {0,1}^m`.
- The receiver sees an independently sampled private index `i ∈ {1,…,m}`.
- The receiver must return exactly the bit `x_i`.
- Neither agent receives the other agent's private input through shared context or tools.

The answer is always one bit. Varying `m` changes how much of the sender's private state is potentially relevant and how much information a one-way protocol must expose. Use fresh bit vectors and indices for held-out seeds; never map episode IDs directly to answers or codes.

## Formal guarantees

For zero-error deterministic one-way communication, sender-to-receiver messages require at least `m` bits in the worst case. If two vectors shared one message, any coordinate where they differ would give the receiver the same observation but demand different outputs. Sending all `m` bits achieves this bound.

With a receiver query, send the index using `⌈log₂m⌉` bits and return the selected bit using one bit. The combined communication is at most `⌈log₂m⌉+1` bits before protocol framing. The receiver can achieve only `1/2` expected accuracy without communication when `x` is uniform and independent of `i`. A single agent given both inputs is the capability control and should be exact.

The derivation and its limits are in the [formalization](PROBLEM_FORMALIZATION.md#a-communication-complexity-control-task).

## Experimental sequence

### A. Validate the task, without ranking formats

For each candidate `m`, verify by construction that the sender lacks `i`, the receiver lacks `x`, and the shared prompt contains neither input. Check the answer key independently. Measure the single-agent full-information control and a no-message receiver control; a protocol study is ineligible if the model fails the former or beats chance on the latter beyond expected sampling variation without communication.

An oracle sender/receiver pass through the same message transport and scorer is a mechanical end-to-end control. This catches task, transport, and scoring defects before LLM calls.

### B. Compare policies

At fixed `m`, compare sending the full vector in one direction with a receiver query followed by a one-bit answer. Both must count all messages, turns, prompts, retries, and generated tokens. This is an interaction-policy result; it cannot establish that a shorthand is a better language.

### C. Compare representations inside fixed policies

Freeze semantic payloads and decoder access, then compare:

- strong optimized natural-language shorthand;
- structured text such as JSON or a compact table;
- a small typed compositional code with explicit symbols for query, answer, index, and bit.

First test exact typed values, then held-out vector lengths, indices, and random bit patterns. Require the decoder to generalize across combinations rather than memorize episode IDs. Include invalid syntax, value substitution, message deletion, and truncation conditions to measure recovery and error detection.

Report task success against total cost: both-direction model tokens, bytes, model-generated tokens, wall/service time, and any shared grammar or calibration cost. Use a task-specific tokenizer plus bytes, because the theoretical bit bounds cannot be equated to LLM token counts.

### D. Cross-model and scaling checks

If a representation wins at fixed `m`, test a held-out receiver model and a different model family before calling the code model-agnostic. Increase `m` only at lengths where the single-agent capability gate still passes. Compare empirical success/cost curves against the exact one-way lower bound and interactive upper construction; do not fit a scaling claim from one or two lengths.

## Falsifiers and limits

- If the optimized natural-language baseline reaches the same frontier as the typed code, there may be no practical language artifact to ship.
- If the code saves output tokens but lowers exact decoding, increases prompt/setup cost, or adds enough retries/latency, it is not an efficiency win.
- If full-information controls fail, the result diagnoses model/vector-reading capability, not communication.
- This task intentionally removes complex reasoning. A success here establishes a valid communication channel, not broad agent competence.
- A fair one-way lower bound counts information in the message only when sender and receiver share no unaccounted task-specific side channel. Grammar, codebook, and decoder prompts are shared state and their acquisition/distribution cost must be counted in deployment claims.
