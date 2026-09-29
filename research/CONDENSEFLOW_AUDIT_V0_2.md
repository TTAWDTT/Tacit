# CondenseFlow audit v0.2: public implementation reproducibility

**Status:** static audit of the authors' public code at commit
`5540c279e18fb1cbf1d2b66178970ffee8e4893e` (shallow-cloned inside this
project's ignored `.cache/research/` directory). No source was executed, no
model weights were downloaded, and no training or inference was run.

This follow-up checks whether the published Quick Start and claimed LTC
communication path can be reproduced from the linked implementation. It does
not re-evaluate or refute the paper's reported benchmark results.

## Blocking cache-return path in the documented CondenseFlow mode

In the pinned [`ltc_wrapper.py`](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/src/models/ltc_wrapper.py#L117-L160),
`generate_with_latent_input` stores the result of Hugging Face `model.generate`
in local variable `outputs`. If `return_kv_cache` is true, it calls
`extract_and_compress_cache()` without passing the cache returned by generation.
That method reads `self.model._last_past_key_values` and raises
`RuntimeError` when the attribute is absent.

The cloned repository contains no write to `_last_past_key_values`; a
repository-wide search found only the read in `extract_and_compress_cache`.
`load_model` loads an ordinary `AutoModelForCausalLM` and installs no hook or
monkey patch. The normal CondenseFlow agent calls this wrapper with
`return_kv_cache=True` for its outgoing message, and the documented
`StandardPipeline` defaults to `communication_mode="condenseflow"`. Thus the
documented standard path reaches the missing-cache error on ordinary
Transformers models unless an unmentioned external patch is applied. The
Dense-Latent branch separately returns `outputs.past_key_values` directly.

## Test coverage does not exercise that path

The public [`test_ltc.py`](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/tests/test_ltc.py)
checks module construction, tensor shapes, compression-ratio arithmetic, and
attention-weight normalization; it does not instantiate `LTCWrapper` or call
`generate_with_latent_input`. The standard-pipeline test is a structural stub
whose pipeline construction and assertions are commented out. There is no
end-to-end test that verifies a generated KV cache is compressed, returned,
fed to the next agent, and used to produce the final answer.

## The proved operator differs from the implemented operator

The v0.1 theory audit identified the query-dependent selection gap in the
attention bound. The implementation adds a separate discrepancy: in
[`ltc.py`](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/src/models/ltc.py),
each compressed value is `LayerNorm(A V)`, while the theorem defines
`V_tilde = A V`. Layer normalization is generally neither the identity nor a
row-stochastic convex combination and can change value-vector norms. Therefore
the stated `Vmax`/total-variation argument does not directly bound the
implemented operator. This is another theory-to-code gap; it says nothing by
itself about measured task accuracy.

## Training-path accounting caveat

The public training entrypoint calls `trainer.train(train_dataset)` without an
evaluation dataset. Although the trainer accepts `eval_dataset` and has a
best-evaluation-checkpoint branch, that branch is not exercised by the
documented training script. The config's 50,000 steps and the paper's reported
approximately four A100-80G GPU-hours are published settings/results, not an
independently measured end-to-end setup cost in this audit. Any comparison
must charge the reported or newly measured probe training, checkpoint
distribution, and receiver/model inference separately.

## Reproduction decision

Treat the paper as a serious conditional latent baseline and its reported
results as published evidence, but label its linked public artifact
**not-reproduced** until the cache-return path and an end-to-end test are
resolved. A future local replication needs to pin the code/model/library
versions; verify the returned `past_key_values` path without relying on
undocumented model attributes; test one full multi-agent exchange; and report
serialized tensor bytes, LayerNorm's effect on the stated guarantee, receiver
compute, and setup amortization. Do not silently repair the upstream code in
this project's baseline record or compare an unexecuted implementation as if
it were a measured result.

## Sources

- Chen et al. (2026), [ACL Anthology paper and appendix](https://aclanthology.org/2026.findings-acl.669/).
- Authors' [pinned wrapper](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/src/models/ltc_wrapper.py)
  and [pinned standard agent](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/src/agents/base_agent.py).
- Authors' [pinned LTC module](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/src/models/ltc.py),
  [unit tests](https://github.com/xxy33/condenseflow/tree/5540c279e18fb1cbf1d2b66178970ffee8e4893e/tests),
  and [training entrypoint](https://github.com/xxy33/condenseflow/blob/5540c279e18fb1cbf1d2b66178970ffee8e4893e/scripts/train_ltc.py).
