# CRAFT parser-adapter diagnostic v0.18

This paired diagnostic reuses the first medium structure and schedule seed from v0.17. The v0.17 raw responses demonstrated that Qwen3 emits its natural-language message in square brackets, while the upstream parser strips all bracketed text. This adapter changes only that parser fallback: non-instruction bracketed spans are unwrapped and preserved as the public message. It also sets child-process console encoding to UTF-8. CRAFT prompts, task, model, move scorer, and oracle setup remain fixed.

It runs the same structure for two turns (at most 8 calls). Because Director generation is stochastic, this is a paired diagnostic rather than an exact replay. The output is not a language comparison. Per-turn message counts and task progress determine whether the channel can now be exercised; any new parsing or execution failure is reported as such.

Start the existing pinned llama.cpp server and run:

```powershell
$env:PYTHONPATH = '.cache/python-packages'
python experiments/craft_v0_18/run_craft_parser_diagnostic.py
```

Raw traces remain in ignored `.cache/pilot_v0_18/`. Public reports must not include private chain-of-thought or full prompts.
