# CRAFT natural-language baseline v0.22

v0.22 establishes a multi-structure baseline before any alternative message protocol is designed. It runs the pinned CRAFT natural-language prompts and task/scorer on one simple, one medium, and one complex structure with Qwen3-8B Q4_K_M.

The runner applies a fixed adapter: sample all three distinct Director roles without replacement, preserve simultaneous-round visibility, remove only `<message>`/`<think>` wrappers and one outer square-bracket envelope during parsing, and set UTF-8 console I/O. The Builder receives each role's exact extracted public prose; no rephrasing, compression, or content rewriting occurs. The same parsing rules are applied in every future condition.

Settings: 8 turns per structure; oracle-assisted Builder with at most five candidate moves; no tool calls; one model and one seed. The limit is 96 completions (3 structures × 8 turns × 3 Directors plus Builder). This is a small reference baseline, not a published benchmark claim.

Run with the v0.21 server already loaded (Qwen3-8B Q4_K_M, 99 GPU layers, one slot, context 4096, batch 1024, micro-batch 256):

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_22/run_craft_baseline.py
```

The runner checks the pinned model and one-slot/context endpoint before generation. Raw CRAFT traces, including private reasoning, remain under ignored `.cache/pilot_v0_22/`; publish only the sanitized metrics and public messages.
