# HiddenBench Qwen3-8B partial-GPU capability screen v0.8

This four-vote screen changes only GPU layer placement from v0.7's 99 layers to 20. It retains the same Qwen3-8B checkpoint, HiddenBench full-information task, seed, official prompts, reasoning mode, and sampling parameters. The purpose is to test whether an 8B model can pass the full-information gate under a resource profile that avoids the previous GPU stop. It does not compare communication protocols.

The preregistration is frozen in [`preregistration.json`](preregistration.json). Raw vote records remain in ignored `.cache/`; publish only a sanitized aggregate.

The launcher uses the same strict idle-load gate as v0.7: CPU baseline below 35% and GPU utilization below 60%. It automatically stops the runner and server after CPU reaches 75% for three consecutive samples or GPU reaches 90% for two consecutive samples. The model server is always stopped in `finally`.

From the repository root:

```powershell
./experiments/hiddenbench_v0_8/run_local_capability.ps1 -PrepareOnly
./experiments/hiddenbench_v0_8/run_local_capability.ps1
```

`-PrepareOnly` verifies pinned local artifacts and the current idle-load gate without starting a model. If the four full-information votes are not all correct, or the runtime stop fires, this task/model line is not eligible for a protocol ranking.
