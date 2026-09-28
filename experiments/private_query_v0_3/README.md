# Exact one-bit private-query scaling v0.3

## Result

The uniform private-query task is a classical (n\to1\) random access code. Its exact optimal average success is known:

\[
p(n)=\frac12+\frac{1}{2^n}\binom{n-1}{\lfloor(n-1)/2\rfloor}.
\]

The optimal deterministic sender computes the majority of the (n) source bits (with either fixed tie convention); the receiver returns that transmitted bit for every queried coordinate. The exact success curve is \(p(n)=1/2+1/\sqrt{2\pi n}+O(n^{-3/2})\), so the one-bit advantage over chance vanishes as the number of independently queryable facts grows. This is a known random-access-code theorem, not a new Tacit result ([Ambainis et al., 2009](https://arxiv.org/abs/0810.2937), §2.3).

This strengthens the project-local Parseval upper bound and recovers the v0.1 exhaustive optima for n=1–4. The script computes exact rational values for n=1–128 and powers of two through the requested scale, with default maximum n=4096. It uses only Python's standard library and makes no model calls.

## Reproduce

From the repository root:

```powershell
python experiments/private_query_v0_3/exact_one_bit_scaling.py --max-n 4096 --output experiments/private_query_v0_3/results.json
```

`results.json` reports exact fractions as well as decimal success and asymptotic-approximation diagnostics. The reference law assumes a uniform source and uniform query, a one-bit classical channel, and no query disclosure to the sender. It says nothing about nonuniform priors, more than one bit, LLM capability, tokenization, or operational codebook cost; those remain separate questions in v0.2 and future experiments.
