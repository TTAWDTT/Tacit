# PAC-BENCH audit v0.1

**Audit date:** 2026-09-29
**Status:** ACL 2026 paper, official repository, and dataset-card review; no artifacts downloaded or executed.

## What it contributes

Park et al., *PAC-BENCH: Evaluating Multi-Agent Collaboration under Privacy Constraints* (Findings of ACL 2026), study two privately owned agents with separate memories, a shared collaborative goal, and explicit disclosure constraints. The paper describes a 100-scenario evaluated set; the [official repository](https://github.com/PAC-Bench/PAC-Bench) also links a [Hugging Face dataset](https://huggingface.co/datasets/PAC-Bench/PAC-Bench) whose card lists 1,476 rows. The dataset card declares CC-BY-4.0. The paper reports that privacy constraints reduce joint task/privacy success and identifies early disclosure, overly conservative abstraction, and privacy-induced hallucination as recurring failure modes.

## Fit to this project

PAC-BENCH adds a valuable conditional deployment axis: a communication channel may need to transfer enough task-relevant information while preventing prohibited disclosure. Its generated scenarios and privacy constraints can stress-test a protocol after a reproducible task/cost comparison is established. This is not evidence that any representation is more bandwidth-efficient, and privacy protection is not currently the project's primary outcome.

## Limits for protocol research

- Task progress is judged with an LLM; privacy evaluation combines a keyword filter with an LLM rubric, and the paper human-checks a subset. These measures are useful for scenario-level research but are not a fully deterministic semantic-fidelity scorer for a representation frontier.
- Scenarios are generated and filtered through LLM calls. The provided simulator expects a multi-agent tool stack; the README's default workflow builds Docker services and starts 20 containers. This is not a lightweight local text-format test.
- The code is public, but the observed public repository README does not state a code license. The Hugging Face dataset card does state CC-BY-4.0; verify each artifact's terms before redistribution. Do not infer that the dataset license automatically covers repository code or upstream assets.
- The paper itself lists natural-language privacy constraints and LLM judging as limitations, and evaluates fixed constraints in two-agent, 20-turn scenarios. Dynamic privacy negotiation and larger groups remain out of scope.

## Reuse decision

Keep PAC-BENCH as a secondary, privacy-constrained external stress test, not a required baseline in the first representation frontier. If used, separately report task success, privacy compliance, and their conjunction; charge all model/judge/tool calls and serialized message costs; and retain human or rule-based verification for any privacy claim. For current machine-checkable work, prefer tasks with frozen inputs, exact answer scorers, and causal correct-versus-mismatched message controls. No PAC-BENCH scenarios or claims are reproduced by this audit.
