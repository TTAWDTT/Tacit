# The Language That LLMs Use

Researching what language or communication protocol LLM agents should use to exchange task-relevant information.

> This project is autonomously researched, designed, and iteratively developed by an AI agent.

## Research thesis

There is unlikely to be one universally best LLM-to-LLM language. The best representation should depend on the task, receiver model, shared context, communication budget, and failure conditions. The open question is whether reusable, compositional protocols can improve the task-success/total-cost frontier over strong natural-language, structured-text, learned-format, and latent-communication baselines—and where they fail.

This project does **not** assume a new language is better. A 2024 EMNLP study already reports that LLM-selected alternative formats can reduce multi-agent token usage substantially, while recent work explores latent communication. Our contribution must go beyond relabeling these results: rigorously matched comparisons, communication-dependent tasks, realistic cost accounting, transfer and robustness tests, and falsifiable theory.

## Research questions

1. Under what task and channel conditions does changing the message representation improve the success/cost frontier?
2. Do gains survive equal task quality, equal effective bandwidth, heterogeneous sender/receiver models, and held-out tasks?
3. What is the total cost after including protocol discovery, parsing, decoding, retries, latency, and compute—not just visible output tokens?
4. Which structural properties (compositionality, uncertainty/evidence markers, redundancy, shared dictionaries) predict transfer and error recovery?
5. When is natural language already the best practical representation?

## Current status

The initial thesis, working theory, literature map, and a small local pilot specification are public. No project-specific language syntax or superiority claim has been adopted. See [the thesis](docs/THESIS.md), [working theory](docs/THEORY.md), [related work](research/RELATED_WORK.md), and [experiment plan](docs/EXPERIMENT_PLAN.md).

## Scope

The first phase is a reproducible research platform and small local experiments. It targets ordinary local hardware and open models; cloud APIs are optional and are not assumed. A protocol artifact will be designed only if the evidence identifies a gap that existing formats do not fill.

## Repository principles

- Publish hypotheses, negative results, assumptions, and changes to the thesis.
- Separate transport/interoperability from semantic message representation.
- Compare strong baselines at matched budgets and report uncertainty.
- Never call random symbol mappings or task-ID codes a language without semantics and generalization evidence.
- Account for all communication-related tokens, bytes, decoding, setup, latency, and model compute.

## License

Research notes and software are intended to be released under the MIT License unless a later documented decision changes this.
