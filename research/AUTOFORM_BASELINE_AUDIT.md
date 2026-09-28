# AutoForm baseline audit

**Reviewed:** 2026-09-28  
**Primary sources:** [Chen et al., Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.623/) · [official paper PDF](https://aclanthology.org/2024.findings-emnlp.623.pdf) · [official implementation](https://github.com/thunlp/AutoForm)  
**Inspected code revision:** `8df94501c462e7f7b4708e5f0297fbdcf8e12ffa` (local research cache; not vendored into this project)

## What AutoForm establishes

AutoForm is the closest direct prior work to this project's question. It asks an LLM to select and use an alternative format for reasoning or agent communication. Its format choices include lists, equations, logical expressions, JSON, tables, pseudocode, and code. The method is prompt-level format selection: it does not define a learned vocabulary, a constrained grammar, or a separate decoder.

The paper's strongest communication control is HotpotQA with supporting facts split between two agents. In its separate-context table, GPT-4/GPT-4 changes from F1 0.65 and 151.0 reported tokens to F1 0.69 and 100.0 tokens (−33.8%). GPT-3.5/GPT-3.5 changes from F1 0.62 and 369.3 tokens to F1 0.53 and 286.6 tokens (−22.4%). Thus lower token count alone does not establish a better quality/cost frontier: the measured quality response depends on the model pair. The paper's headline 72.7% reduction is a different HotpotQA pairing/setting in Table 2, and must not be attributed to the separate-context table.

This work already establishes both that communication can be necessary under split evidence and that a model-selected format is a serious baseline. Neither is a novelty claim for Tacit. The open question is whether any representation can offer a robust advantage after controlling task success, total cost, receiver ability, format adherence, and generalization.

## What the released implementation does (and does not)

The inspected official HotpotQA runner shuffles non-empty context passages with a fixed seed and distributes them round-robin across two critics. The YAML's `cnt_agents: 3` is easy to misread: the loader instantiates `cnt_agents - 1` critic agents, so the actual pair matches the paper's two-agent description. Its AutoForm-named entrypoint defaults to a HotpotQA config whose prompt asks agents to choose a more efficient medium, gives examples of evidence sharing, invites structured data or code, and requests concise communication. The repository also contains explicit JSON and KQML prompt variants. The prompts allow free-form output; the code does not enforce a formal message grammar or decode a new language. The setup documentation requires OpenAI and Google API keys, so the published hosted workflow cannot be reproduced as-is with this project's local-only inference budget.

The code exposes API prompt/completion token counters, but the exact aggregation and scope of the paper table's `# Tokens` value still need tracing to the table-generation path. Until then, treat the published figure as the paper's reported token metric, not as verified total communication cost.

This is a useful distinction for evaluation: AutoForm measures an instruction that may cause a model to emit a shorter ad-hoc format. A fixed JSON/KV protocol measures adherence to an explicitly specified representation. They answer different questions and should appear as separate arms. If an AutoForm prompt causes a model to stay in natural language, that is an instruction-following outcome, not evidence that some hypothetical structured format failed.

## Implications for the next Tacit comparison

1. Keep AutoForm-style format selection in the baseline set alongside carefully optimized natural-language messages, schema-constrained JSON, compact key-value text, and no-communication/oracle controls. Do not label an unparsed prompt-selected output as a new language.
2. Use tasks with verified private information and a model/task gate showing that the receiver can solve the task when given sufficient evidence. The v0.8 full-information HiddenBench screen passes only for one task, seed, model, and four votes; it does not pass the hidden-information or protocol-comparison gate.
3. Pair every communication condition on the same task, seed, model revision, decoding settings, and runtime. Preserve raw generated messages and score (a) successful delivery, (b) semantic evidence transfer, (c) answer quality, (d) adherence/parseability, and (e) end-to-end cost separately.
4. Report a success-versus-cost frontier, including prompt/input tokens, generated tokens, wire bytes, rounds, latency, and available compute measures. Do not infer savings from message-body token counts alone. Compare both at fixed cost and at matched quality; retain per-task paired outcomes and uncertainty.
5. Include a falsifiable budget prediction: if a representation mainly removes redundant message content without losing task-critical evidence, its advantage should be largest at tight communication budgets, while its success should converge toward strong baselines as the channel becomes unconstrained. Measure evidence omissions and repair turns to test the mechanism.

## Current decision

AutoForm is a required baseline and prior result, not a reason to create a new symbolic protocol now. Its public numbers motivate a local, matched replication, but the local Qwen3-8B screen's host-CPU peak (91.9%) makes a multi-condition run unjustified until a smaller request budget passes an idle-machine preflight. Continue offline design and benchmark auditing first. No new model inference was run for this audit.
