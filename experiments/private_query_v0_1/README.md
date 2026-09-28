# Private-query benchmark v0.1

## Purpose

This is a small, exact communication-complexity control for a receiver that privately chooses which fact it needs. A sender sees an (n)-bit vector (X); a receiver sees a uniformly random coordinate (I), unknown to the sender, and must return (X_I). The sender may transmit one bit over a noiseless one-way channel.

This task checks whether a fixed, task-oblivious representation can preserve information for a family of receiver queries. It does not test language-model ability or establish that any human-readable or learned language is superior.

The task is a classical **random access code (RAC)**, a well-established communication primitive studied in communication complexity and quantum information. In particular, prior work analyzes classical (n\to1) codes and shared-randomness variants; this repository does not claim novelty for the task or primitive ([Ambainis et al., *Quantum Random Access Codes with Shared Randomness*](https://arxiv.org/abs/0810.2937)). The project-specific use is as an exact, cheap gate for checking task-distribution and sender/receiver information assumptions before spending inference budget on LLM protocol comparisons.

## Exact baseline conditions

- **No message:** receiver success is 1/2.
- **One-bit task-oblivious code:** the script enumerates every deterministic encoder and gives each message/query pair its Bayes-optimal decoder.
- **One-bit query-conditioned oracle:** if the sender is explicitly given (I), it sends (X_I) and succeeds perfectly. This condition changes the information boundary. If (I) must be communicated, its reverse-channel cost must be counted.
- **Full-source oracle:** send all (n) bits and succeed perfectly.

For any task-oblivious one-bit code, the Fourier/Parseval bound in [`docs/THEORY.md`](../../docs/THEORY.md#9-private-query-communication-benchmark) gives success at most (1/2+1/(2\sqrt n)). The bound need not be tight at small (n); the exhaustive report identifies exact finite optima for (n\le4).

## Reproduce

Run from the repository root:

```powershell
python experiments/private_query_v0_1/enumerate_codes.py --max-n 4 --output experiments/private_query_v0_1/results.json
```

The script uses only the Python standard library, makes no model calls, and enumerates (2^{2^n}) truth tables for each (n). The committed `results.json` is the reference output. Regenerating it is deterministic.

## Scope and next experiment

This is a model-free reference task, not LLM evidence. A later LLM study should first verify that both roles can reliably encode and decode the elementary bit vectors. If they can, compare a task-oblivious learned code against query-conditioned and full-source controls under matched *total* forward and reverse budgets, including any query disclosure, prompt, schema, setup, and decoding cost. Vary source width and query distribution, and test unseen query mixtures before making generalization claims.
