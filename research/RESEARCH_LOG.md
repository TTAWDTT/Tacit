# Research log

## 2026-09-27 — Initial scoping and public thesis

- Confirmed the project directory was not yet a Git repository and had no project files.
- Surveyed primary sources across alternative-format LLM communication, latent collaboration, emergent language, protocol benchmarks, topology, and current A2A specification.
- Key correction: non-natural format selection was already directly tested by Chen et al. (Findings EMNLP 2024); token reduction alone is not a novel contribution. Recent latent communication work also means latent baselines cannot be omitted.
- Initial thesis: communication representation should be evaluated as task/receiver/channel-conditioned decision-making, against task success and total cost; no universal new language is assumed.
- Created the first benchmark and baseline plan as hypotheses/design, not reported results.
- Machine feasibility noted: RTX 4060 8 GiB VRAM, ~32 GiB RAM. Plan local experiments around small models and symbolic tasks.
- Next: inspect full papers and code for the closest four baselines; map existing communication-dependent benchmark tasks before designing a new benchmark; revise thesis when results warrant it.
