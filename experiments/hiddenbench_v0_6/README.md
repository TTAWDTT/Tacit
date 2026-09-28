# HiddenBench phase-policy feasibility pilot v0.6

## Observed run status

The 2026-09-28 run was interrupted after `natural_3` completed and `exchange_decide` had begun. `natural_3` scored 0.00 initial/final individual and majority accuracy on this one task and seed. Host CPU peaked at 89.3%; the local server was stopped and `reveal_all_3` did not run. See [`../../research/HIDDENBENCH_PHASE_PILOT_V0_6_INTERRUPTED.md`](../../research/HIDDENBENCH_PHASE_PILOT_V0_6_INTERRUPTED.md) and its [sanitized JSON aggregate](../../research/HIDDENBENCH_PHASE_PILOT_V0_6_INTERRUPTED.json). These data do not support a protocol comparison. The preregistered run must not be restarted unchanged; a future run needs a separately documented lower-cost configuration and a capability gate.

This preregistered pilot tests an implementation of the published HiddenBench Exchange/Decide baseline and its Reveal-All diagnostic. It uses one hidden-profile verification task, one seed, and three-round matched conditions. The pilot is designed to validate the phase-aware runner and measure resource use; one task cannot establish comparative effectiveness.

The implementation reuses the pinned HiddenBench task and prompt data, agent instantiation, vote validation, and official scorer. The local adapter adds explicit round-phase prompts, which the upstream CLI's single static `--extra-prompt` cannot express. Reveal-All appends each sender's actual visible facts verbatim to their first-round message; no hidden fact is given to the wrong agent.

## Frozen configuration

See [`preregistration.json`](preregistration.json) before running. The local model remains Qwen3-14B Q4_K_M, but the server is limited to four generation threads and four batch/prompt threads. This is a new configuration and is not pooled with v0.5. The runner makes at most 60 nominal model calls (20 per condition); API retries can increase actual server requests and are recorded separately.

## Offline protocol tests

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path '.cache/research/HiddenBench_ICML/src').Path
python -m unittest discover -s experiments/hiddenbench_v0_6 -p 'test_*.py' -v
```

The fake model checks round scheduling, vote counts, presence of phase-specific instructions, and that Reveal-All appends only the sender's visible facts. Analyzer tests check per-condition server-log attribution and verify that sanitized outputs do not include private benchmark facts. These tests perform no model inference and use no model server.

## Local run

The raw benchmark outputs contain task facts and prompts. Keep them under ignored `.cache/`; only sanitized aggregates may be published.

Check the host-load gate without starting a model service:

```powershell
./experiments/hiddenbench_v0_6/run_local_pilot.ps1 -CheckHostLoadOnly
```

The PowerShell orchestrator requires baseline host load below 40% CPU and 70% GPU. It then starts llama.cpp hidden at below-normal process priority, with generation and batch threads capped at four. It checks the model alias and pinned preflight before inference, then always stops the model server when the runner exits:

```powershell
./experiments/hiddenbench_v0_6/run_local_pilot.ps1
```

The run writes one private result after each completed condition under `.cache/pilot_hiddenbench_v0_6/<timestamp>/`, then creates a sanitized score/cost report next to it while the local tokenizer endpoint is still available. It also records periodic server/host CPU, memory, and GPU samples. Raw JSON contains task facts and private prompts; only the sanitized report is eligible for publication. The orchestrator stops the server after the run or if the computer becomes uncomfortable to use:

```powershell
Stop-Process -Id $server.Id
```

The preregistration's stop condition takes precedence over completion. Press Ctrl+C if the computer becomes uncomfortable to use; report any interrupted condition as incomplete and do not score a partial transcript as a completed task. The orchestrator's `finally` block stops its model server on normal completion or an interrupted Python run.
