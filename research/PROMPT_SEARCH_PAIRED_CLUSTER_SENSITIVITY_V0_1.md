# Paired cluster sensitivity for prompt-search selection (v0.1)

**Status:** model-based sensitivity analysis only. The ICC values below are assumptions, not estimates from Tacit outcomes. This does not estimate the optimizer's actual bias, establish a protocol advantage, or choose a confirmatory sample size.

## Why add a paired model?

The [iid sensitivity](PROMPT_SEARCH_SELECTION_BIAS_SENSITIVITY_V0_1.md) treated each candidate's 64 training outcomes as independent. In v0.4, all candidate prompts are evaluated on the same candidate-set episodes. A hard set can lower every candidate's score at once; this shared difficulty is removed from candidate-to-candidate contrasts. Standard cluster design-effect formulas correctly warn that a single arm's marginal accuracy has less independent information when outcomes within clusters correlate, but that formula alone does not give the variance of paired candidate differences. Cluster-trial methods emphasize the need to match correlation assumptions to the analysis and outcome ([Eldridge et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4521133/)).

Adaptive reuse raises a separate issue: choosing new prompts after seeing training scores is expected, while repeatedly adapting to a holdout can undermine its generalization role. Dwork et al. formalize adaptive holdout reuse and show why access to previous holdout results matters ([paper](https://papers.nips.cc/paper/5993-generalization-in-adaptive-data-analysis-and-holdout-reuse.pdf)). Tacit's safeguards therefore keep iterative feedback train-only, freeze candidates before final test, and reserve test for one evaluation.

## Paired beta-binomial random-intercept null

Let `j=1,...,J` index candidate-set clusters, `r=1,...,m` episodes per cluster, and `c=1,...,M` prompt candidates. Assume every candidate has the same population success rate `p`. A latent cluster difficulty `q_j` is shared by all candidates, with `q_j ~ Beta(pκ, (1-p)κ)`. Conditional on `q_j`, each `Y_cjr ~ Bernoulli(q_j)` independently. The within-candidate intracluster correlation is `ρ = 1/(κ+1)`. Every candidate is scored on the same `J×m` task rows.

For the accuracy `S_c` of one candidate over `N=Jm` rows:

`Var(S_c) = p(1-p)/N × [1 + (m-1)ρ]`.

For two distinct candidates, the shared latent cluster difficulty gives `Cov(S_c,S_c') = ρp(1-p)/J`. Therefore:

`Var(S_c - S_c') = 2p(1-p)(1-ρ)/N`.

This is a paired-comparison result under the stated random-intercept model: common cluster difficulty cancels from the difference. It predicts decreasing candidate-ranking noise as `ρ` rises when the residual outcomes are conditionally independent. It does **not** imply that clustering is harmless for accuracy estimation, nor does it cover candidate-specific cluster interactions, correlated generation failures, adaptive proposals, heterogeneous cluster sizes, or fixed task suites with no sampling interpretation.

## Monte Carlo sensitivity at v0.4's nominal geometry

The standard-library simulator uses `J=16`, `m=4`, `p=0.50`, 10,000 replicates, and seed 37. It reports the expected selected training accuracy minus `p`; `MCSE` is the simulation standard error of that estimate. The 4-candidate and 8-candidate arms are exchangeable candidates under this null. The 8-candidate result is only a non-adaptive sensitivity case: it does not model the second adaptive proposal round.

| Assumed ICC `ρ` | candidates `M` | expected selected-score optimism | MCSE |
|---:|---:|---:|---:|
| 0.00 | 4 | 6.42 pp | 0.04 pp |
| 0.10 | 4 | 6.11 pp | 0.06 pp |
| 0.25 | 4 | 5.58 pp | 0.07 pp |
| 0.50 | 4 | 4.41 pp | 0.09 pp |
| 0.00 | 8 | 8.86 pp | 0.04 pp |
| 0.10 | 8 | 8.33 pp | 0.05 pp |
| 0.25 | 8 | 7.57 pp | 0.07 pp |
| 0.50 | 8 | 6.06 pp | 0.09 pp |

The `ρ=0` results reproduce the iid scenario within Monte Carlo error. Under this shared-difficulty null, the selected-score optimism declines as ICC increases because common difficulty moves candidate scores together and candidate-specific residual noise shrinks. This is a model implication, not an estimate of how Tacit's tasks behave.

Reproduce the simulation (roughly a few seconds on a typical desktop CPU):

```powershell
python research/prompt_search_selection_bias.py --n 64 --p 0.50 --candidates 4 8 --paired-cluster-sensitivity --replicates 10000 --seed 37
```

## Experimental consequence

For future paired comparisons, retain the same episode IDs and candidate-set ordering across all candidates; report per-set paired differences and their distribution, not only separate candidate accuracies. The training optimizer can use the paired training outcomes to propose revisions, but its selected training score stays development-only. Freeze the full search output once on validation, then evaluate the frozen comparison on untouched test data. Independent split seeds remain the highest-level replication unit in the existing clustered-analysis audit; four episodes per candidate set do not become four independent splits.

The next model-backed study should estimate within-set and across-split variation from a resource-approved, preregistered pilot before selecting a confirmatory number of split seeds. Do not plug assumed ICC values from this sensitivity grid into a power calculation. If observed paired candidate differences fail to show lower variance with higher within-set shared difficulty, the random-intercept mechanism is falsified for this task/model population and a richer structure is needed.
