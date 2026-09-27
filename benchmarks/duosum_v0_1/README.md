# DuoSum benchmark v0.1

DuoSum is a deliberately small communication-necessity benchmark for two agents. Each agent sees one positive private integer. Both agents must return the exact sum. Since the other value is positive and private, neither local observation determines the answer. This isolates semantic handoff and coordination from open-domain reasoning; it is a calibration benchmark, not a substitute for planning, coding, scientific, or long-horizon tasks.

The generator creates four input widths (4, 8, 12, and 16 bits) with four deterministic replicates per width. Replicate 0 is the calibration split; replicates 1–3 are held out. The JSON task files contain both inputs and gold outputs for reproducibility. The Silo-Bench engine places only each agent's own value in its model context.

## Generate and audit

```powershell
python benchmarks/duosum_v0_1/generate.py
```

The generator writes task JSON and a SHA-256 manifest under `tasks/`. It asserts that each agent's local value differs from the gold sum. For an experiment, the runner also checks the manifest, per-file hashes, pinned engine commit, and pinned model weights.

## Theory connection

For two agents with private inputs from a domain of size `M`, both required to output their exact sum, a deterministic binary point-to-point protocol needs at least `2 log2(M)` bits in the worst case. If `M` is a power of two, sending one fixed-width input from each agent attains the bound. See [`docs/THEORY.md`](../../docs/THEORY.md) for assumptions, proof, and the distinction between wire bits and model tokens.
