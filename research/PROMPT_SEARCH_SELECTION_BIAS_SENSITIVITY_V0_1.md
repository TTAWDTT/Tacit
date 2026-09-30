# Prompt-search selection bias: an iid sensitivity calculation (v0.1)

**Status:** theoretical sensitivity only. This is not an empirical estimate of the v0.4 optimizer's bias, a power analysis, or evidence that any protocol wins.

## Question

When several English prompt candidates are scored on the same training block and the highest-scoring candidate is retained, how much apparent accuracy improvement can selection alone create under a simple null model?

## Calculation

Assume each of `M` candidates has `n` independent Bernoulli outcomes with the same success probability `p`, so its score count is `X_i ~ Binomial(n, p)`. If the development process reports the maximum score, its expected optimism is

`E[max_i X_i] / n - p = (1/n) sum_{k=1..n} (1 - F(k-1)^M) - p`,

where `F` is the binomial CDF. The exact standard-library implementation is [`prompt_search_selection_bias.py`](prompt_search_selection_bias.py). Reproduce the table with:

```powershell
python research/prompt_search_selection_bias.py
```

| iid outcomes per candidate (`n`) | Null success rate (`p`) | candidates (`M`) | expected selected-score optimism |
|---:|---:|---:|---:|
| 4 | 0.25 | 4 | 22.4 pp |
| 4 | 0.50 | 4 | 24.9 pp |
| 4 | 0.75 | 4 | 19.4 pp |
| 12 | 0.25 | 4 | 13.0 pp |
| 12 | 0.50 | 4 | 14.7 pp |
| 12 | 0.75 | 4 | 12.2 pp |
| 12 | 0.50 | 8 | 20.2 pp |
| 64 | 0.25 | 4 | 5.6 pp |
| 64 | 0.50 | 4 | 6.4 pp |
| 64 | 0.75 | 4 | 5.5 pp |
| 64 | 0.50 | 8 | 8.9 pp |

“pp” means percentage points. These values are expected development-set optimism conditional on this toy model, not confidence intervals and not test-set bias after a valid untouched evaluation.

## Why this does not estimate v0.4 bias

The default v0.4 training block has 16 candidate sets and four target episodes per set (64 episode rows), but the rows share candidate-set structure and are not 64 independent draws from a common task population. The search also runs a second proposal round using first-round training feedback; therefore its eight total candidates are adaptive and are not iid candidates. The `n=64, M=8` row is intentionally omitted from the compact table because plugging those values into the formula would invite a false interpretation. The iid calculation can understate or overstate the actual bias depending on within-set dependence and how adaptation reuses errors.

The search currently evaluates each of four candidates over the full 16-set training block in each of at most two rounds. That is `2 rounds × 4 candidates × 16 sets × 4 episodes × 2 agent calls = 1,024` task-generation/evaluation calls, plus at most two proposal calls. This is the full-coverage upper envelope, not a required run size or a recommendation to execute it now. It is unsuitable while the last recorded local resource preflight fails. Any smaller development search must record its fixed set coverage and cost; it still cannot replace independent split seeds for confirmatory inference.

## Decision and falsifiable implication

Treat iterative English prompt search as development only. Freeze candidates using validation after training-only search is complete, then evaluate once on untouched test tasks, and use independent split seeds as the inferential unit. Report the full search cost and all tried candidates. A falsifiable toy-model prediction is that, at fixed training sample size, the selected candidate's train-to-held-out score gap grows as the candidate count grows; at fixed candidate count, that gap shrinks as training sample size grows. Failure to observe these directions would weaken this simple selection-risk prediction, though clustered split-level uncertainty still governs the final claim.

The calculation makes one operational point: four candidates on a small block can create a large apparent gain under a null. It gives no support to a new communication language and does not supply the unresolved split-level sample size.
