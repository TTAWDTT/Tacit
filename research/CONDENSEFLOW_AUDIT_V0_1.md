# CondenseFlow audit v0.1

**Status:** source-grounded literature and theory audit; no model was loaded and
no CondenseFlow experiment was reproduced.

## Why it matters here

CondenseFlow is a strong learned latent-communication baseline, not a text
codec. It compresses same-architecture agents' layerwise KV caches to a fixed
number of learned probe summaries and reuses them as a KV prefix at the next
agent. The ACL 2026 Findings paper reports seven benchmarks and six models,
more than 99% lower KV-cache memory than dense transfer, about 20% lower
latency in a 20-round stress test, and a small average accuracy gap to dense
transfer. These results make a latent arm necessary for any broad claim about
the best LLM-to-LLM channel when both agents expose compatible internal states.

The result has a narrower deployment envelope than an API-level language:
the paper explicitly assumes equal KV dimensions, calls out heterogeneous
agents as future work, and requires training an LTC module. Its appendix
reports 50,000 training steps and approximately four A100-80G GPU-hours. A
reproduction must include checkpoint setup and amortization, tensor dtype and
serialized bytes, agent schedule, transfer/device movement, receiver-side
compute, and privacy/auditability. “KV-cache memory” and “tokens saved” are not
substitutes for a measured serialized channel cost.

## Formal claim audit

The paper defines `rho` as the worst query's attention mass on its best `K`
positions, then states a bound on the Frobenius error for a compressed KV
cache formed by one row-stochastic aggregation matrix `A`. In the proof, the
selection matrix `A*` is constructed from `S*(q)`, the best positions for each
query. That construction is query-dependent. The final step then takes one
fixed matrix `A` across all queries and invokes the query-wise construction as
an upper bound on `min_A`.

Those quantifiers do not follow: a high-attention subset can differ between
queries, and a single transmitted cache cannot choose a different selection
matrix after seeing each receiver query. The paper's actual LTC aggregation is
more constrained still (`A = softmax(Qc K^T / sqrt(d))`); the proof optimizes
over all row-stochastic matrices without showing that its query-specific
selector belongs to that parameterized family.

### Reproducible two-query counterexample

Use one-dimensional keys and values `K = V = [1, -1]`, two receiver queries
`q = [5, -5]`, and compression length `Kc = 1`. For either query, its best
single original position carries `rho = sigmoid(10) ≈ 0.9999546` of the
attention mass. With `Vmax = 1` and two queries, the stated bound's right-hand
side is about `0.000128`.

Any single row-stochastic compressor `[p, 1-p]` produces one compressed key
and value. Attention over that one-position cache returns the same scalar for
both queries. The exact original outputs are approximately `[0.999909,
-0.999909]`; the minimum possible Frobenius error of any one fixed compressor
is therefore `sqrt(2) * tanh(5) ≈ 1.414`. The discrepancy is over four orders of
magnitude. Each query separately has a near-perfect top-one position, but
those positions are different. The executable calculation is
[`counterexample.py`](../experiments/condenseflow_audit_v0_1/counterexample.py).

This counterexample challenges the stated guarantee for a **single fixed
compressor**. It does not show that the empirical learned compressor performs
poorly, nor does it invalidate the standard query-specific top-`K` attention
truncation bound. A valid shared-compressor theorem would need a common
query-independent selection/aggregation condition (for example, a bound on
the mass captured by one subset for every query), or an explicit query-aware
communication protocol that charges the selection/query exchange. The
multi-round `R * delta` statement also bounds output errors only if each
round's downstream transformation is non-expansive in the chosen norm; a
general transformer can amplify upstream perturbations, so an appropriate
Lipschitz factor or empirical-only qualification is needed.

### A valid query-shared sufficient condition

For a fixed, prespecified set of receiver queries `Q`, define

`rho_shared = max_{S subset [T], |S|=K} min_{q in Q} sum_{j in S} alpha_j(q)`.

This reverses the paper's per-query selection order: choose one support `S`
first, then evaluate its least attention mass over every query. Let `A_S` be
the single hard selector for `S`, reused for all queries. For any `q`, let
`beta_q = sum_{j in S} alpha_j(q)`. The compressed attention distribution is
the original distribution renormalized on `S`, so its total-variation
distance from the original is `1 - beta_q`; equivalently, the `L1` distance is
`2(1 - beta_q)`. With `||v_j||_2 <= Vmax`, this gives

`||O - O_tilde(A_S)||_F <= 2 (1 - rho_shared) Vmax sqrt(|Q|)`.

The proof applies one and the same support to every query. Since a hard
selector is a row-stochastic matrix, the same upper bound applies to the
minimum over the full set of row-stochastic matrices. Extending it to the
implemented LTC requires the extra condition that its parameterized
probe-induced aggregation can realize or approach `A_S`; the paper does not
establish that condition. `Q` must also be fixed independently of the realized
query at communication time, or the cost of selecting and disclosing query
specific support must be counted.

In the two-query counterexample, `rho_shared = 1 - sigmoid(10)`, about
`0.0000454`; the corrected bound is loose (about `2.828`) but valid. This is
the desired diagnostic behavior: incompatible salient supports erase the
guarantee instead of promising near-zero error. The executable reproduces both
the invalid per-query claim and this shared-support sufficient bound.

## Falsifiable predictions for a future latent arm

1. Construct receiver query sets whose per-query top-`K` supports are disjoint.
   At fixed `K`, query-wise attention concentration can remain high while the
   best single shared compressor's error rises. Measure both quantities; this
   directly tests whether per-query salience predicts a reusable message.
2. Compare query-specific top-`K` (with query disclosure/round-trip cost), one
   shared top-`K`, fixed learned LTC, and the exact dense cache. Do not present
   the query-aware oracle as a free baseline.
3. Across increasing rounds, separately measure local attention-output error,
   downstream logit/task error, and serialized bytes. If model layers amplify
   small cache errors, output perturbation will grow according to measured
   layer sensitivity rather than the additive `R * delta` rule alone.
4. Count training/setup and amortize it across explicit reuse horizons. The
   result can only support a homogeneous internal-state deployment claim until
   heterogeneous KV dimensions and model-family transfer are demonstrated.

## Research decision

Retain CondenseFlow as a required conditional upper-bound baseline in the
experiment plan. Do not rank it against black-box text protocols without
separating their access assumptions and full costs. The formal gap strengthens
the project's emphasis on fixed-message semantics and task-grounded transfer:
high attention concentration alone is not evidence that one reusable message
preserves all receiver-relevant queries.

## Sources

- Chen et al. (2026), [ACL Anthology paper and appendix](https://aclanthology.org/2026.findings-acl.669/).
- Authors' [public implementation](https://github.com/xxy33/condenseflow).
