# Research log

## 2026-09-27 — Initial scoping and public thesis

- Confirmed the project directory was not yet a Git repository and had no project files.
- Surveyed primary sources across alternative-format LLM communication, latent collaboration, emergent language, protocol benchmarks, topology, and current A2A specification.
- Key correction: AutoForm (Chen et al., Findings EMNLP 2024) directly tested alternative-format communication, including an evidence-split HotpotQA setting. Token reduction and private-information necessity are therefore not novelty claims. Its quality/cost response differs by model pair (GPT-4 improves while GPT-3.5 declines on that split), motivating explicit quality-cost curves and current open receiver replications.
- Found existing communication-dependent suites: Silo-Bench (30 algorithmic tasks, 2–100 agents, several topologies) and MT-PingEval (private-information games, fixed token budget divided over turns). Use them before considering new benchmark creation.
- ProtocolBench measures transport/protocol system costs with bytes, latency, quality, and failure recovery, but its scenarios center document QA/tool workflows, queues, safety, and failure storms. Treat it as the instrumentation comparator; do not conflate transport overhead with semantic format efficiency.
- AutoForm's public implementation requires OpenAI and Google API credentials. The inspected local workstation has Python 3.13, PyTorch 2.6 CUDA 12.4, Transformers 5.2, 8 GiB RTX 4060 / ~32 GiB RAM, and no standard Ollama/llama.cpp executable or exposed model API keys. A local HF model with a compatible small runtime is feasible but must be validated before promising an execution path.
- Initial thesis: communication representation should be evaluated as task/receiver/channel-conditioned decision-making, against task success and total cost; no universal new language is assumed.
- Created the first benchmark and baseline plan as hypotheses/design, not reported results.
- Machine feasibility noted: RTX 4060 8 GiB VRAM, ~32 GiB RAM. Plan local experiments around small models and symbolic tasks.
- Next: inspect official benchmark schemas/metric implementation and AutoForm prompts/token accounting; decide a locally feasible model/server; publish an exact protocol-agnostic benchmark adapter spec before running model comparisons.
