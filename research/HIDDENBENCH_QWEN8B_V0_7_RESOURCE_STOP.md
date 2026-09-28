# HiddenBench Qwen3-8B capability screen v0.7: automatic resource stop

**Status:** incomplete; no scoreable votes and no capability conclusion.

The preregistered run started on 2026-09-28 with Qwen3-8B Q4_K_M, 99 GPU layers, one 8,192-token slot, reasoning enabled, and the official model-card thinking sampler. Its launch gate passed at 16.5% host CPU and 39% GPU utilization. The automated runtime guard stopped the runner and llama.cpp after two 15-second samples showed GPU utilization at **97%** and **95%**. Host CPU was only 20.5% and 20.2% in those samples. GPU memory was 6,527/8,188 MiB. After shutdown, no `llama-server` or runner process remained and GPU memory returned to about 699 MiB.

The server log records one completed model request (612 prompt tokens, 1,021 generated tokens, 23.55 seconds service time) and a second request launched when the guard fired. The runner was terminated before its four-vote result could be persisted. Because vote-level outputs are not durably checkpointed, the completed response is not counted as a scored vote. No accuracy or capability conclusion is available.

## Interpretation

The low host CPU confirms that full GPU offload shifts load away from the CPU but can still occupy the laptop GPU heavily. The resource guard worked as designed. This pilot does not establish that the model is too large or too slow; it establishes only that 99-layer placement violated the preregistered GPU stop rule on this machine.

The next attempt is preregistered separately in `experiments/hiddenbench_v0_8/` with the same cached model, task, seed, sampling, and vote gate, but 20 GPU layers. It is still only a four-vote full-information screen. The threshold stays at 4/4; if the lower-GPU configuration fails the gate or trips a resource stop, stop the HiddenBench local-model line rather than weakening the gate.
