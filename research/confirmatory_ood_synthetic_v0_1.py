"""Standalone synthetic verification, not an experimental scorer or power claim.

No dependencies, network, model calls, or experimental inputs. Default output is
deterministic JSON; --self-test runs the verification suite. Assumed two-point
split distributions are integrated exactly over their binomial sufficient count
(up to floating-point arithmetic), rather than estimated by Monte Carlo.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import unittest


ALPHA = 0.05 / 2
MARGIN = 0.05


def interval(mean, n, lower, upper, alpha=ALPHA):
    """Two-sided Hoeffding interval for independent bounded split observations."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer number of splits")
    if not all(math.isfinite(v) for v in (mean, lower, upper, alpha)):
        raise ValueError("finite values required")
    if not lower < upper or not lower <= mean <= upper or not 0 < alpha < 1:
        raise ValueError("invalid range, mean, or alpha")
    radius = (upper - lower) * math.sqrt(math.log(2 / alpha) / (2 * n))
    return max(lower, mean - radius), min(upper, mean + radius)


def split_contrasts(rows, expected):
    """Synthetic rows (split, episode, arm, cap, binary success); explicit manifest.

    Enforce balanced complete coverage; never make a complete-case selection.
    This intentionally does not parse real Tacit ledgers or replace any scorer.
    """
    if not expected or any(not ids for ids in expected.values()):
        raise ValueError("nonempty expected split/episode manifest required")
    counts = {len(ids) for ids in expected.values()}
    if len(counts) != 1:
        raise ValueError("unequal split coverage violates this design")
    wanted = {
        (split, episode, arm, cap)
        for split, episodes in expected.items()
        for episode in episodes
        for arm in ("A", "B")
        for cap in ("tight", "middle", "loose")
    }
    found = {}
    for split, episode, arm, cap, success in rows:
        key = split, episode, arm, cap
        if key not in wanted or key in found or success not in (0, 1):
            raise ValueError("unexpected/duplicate row or nonbinary outcome")
        found[key] = success
    if found.keys() != wanted:
        raise ValueError("missing registered outcomes")
    effects = []
    for split in sorted(expected):
        ids = expected[split]
        differences = {
            cap: math.fsum(found[split, e, "A", cap] - found[split, e, "B", cap]
                           for e in ids) / len(ids)
            for cap in ("tight", "loose")
        }
        effects.append((differences["tight"], differences["tight"] - differences["loose"]))
    return effects


def binomial_weights(n, p):
    if not 0 <= p <= 1:
        raise ValueError("probability out of range")
    if p in (0, 1):
        return [float(k == n * p) for k in range(n + 1)]
    weights = [math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                        - math.lgamma(n - k + 1) + k * math.log(p)
                        + (n - k) * math.log1p(-p)) for k in range(n + 1)]
    total = math.fsum(weights)
    if not math.isclose(total, 1, abs_tol=1e-9):
        raise ArithmeticError("unstable binomial mass")
    return [w / total for w in weights]


def exact_scenario(n, p, low, high):
    """Assume D_t=X, D_l=-X, hence J=2X; split draws X are iid.

    Feasible arm rates: at tight cap A=(1+X)/2, B=(1-X)/2;
    at loose cap reverse the rates. No assumption of monotonic model utility.
    """
    # Preserve the exact decimal null boundary (.05), avoiding accidental
    # classification as an alternative due to cancellation at p=.525.
    prob = Fraction(str(p))
    truth_d = float(Fraction(str(low)) * (1 - prob) + Fraction(str(high)) * prob)
    truth_j = 2 * truth_d
    masses = binomial_weights(n, p)
    events = {name: [] for name in (
        "noncoverage_D", "noncoverage_J", "simultaneous_noncoverage",
        "practical_rejection_D", "practical_rejection_J", "any_true_null_rejection",
    )}
    for k, mass in enumerate(masses):
        mean_d = low + (high - low) * k / n
        ci_d = interval(mean_d, n, -1, 1)
        ci_j = interval(2 * mean_d, n, -2, 2)
        miss_d = not ci_d[0] <= truth_d <= ci_d[1]
        miss_j = not ci_j[0] <= truth_j <= ci_j[1]
        reject_d = ci_d[0] > MARGIN
        reject_j = ci_j[0] > MARGIN
        flags = (miss_d, miss_j, miss_d or miss_j, reject_d, reject_j,
                 (truth_d <= MARGIN and reject_d) or (truth_j <= MARGIN and reject_j))
        for name, flag in zip(events, flags):
            if flag:
                events[name].append(mass)
    return {
        "independent_splits": n,
        "assumed_mean_D": truth_d,
        "assumed_mean_J": truth_j,
        **{name: math.fsum(values) for name, values in events.items()},
    }


def make_fixture(repeats=1):
    expected = {s: set(range(4 * repeats)) for s in range(3)}
    rows = []
    for s, episodes in expected.items():
        for e in sorted(episodes):
            for cap in ("tight", "middle", "loose"):
                for arm in ("A", "B"):
                    score = int(e % 4 < (s + 1 if arm == "A" and cap == "tight" else 1))
                    rows.append((s, e, arm, cap, score))
    return rows, expected


def report():
    scenarios = {
        "zero_null_max_variance": (.5, -1.0, 1.0),
        "H1_margin_boundary": (.525, -1.0, 1.0),
        "H2_margin_boundary": (.5125, -1.0, 1.0),
        "assumed_positive_effect": (.6, -1.0, 1.0),
        "low_variance_positive": (.5, .125, .25),
        "skewed_zero_mean": (.2, -.25, 1.0),
        "constant_zero": (0.0, 0.0, 0.0),
        "constant_upper_boundary": (1.0, 1.0, 1.0),
    }
    counts = (4, 12, 20, 100, 1000)
    return {
        "schema": "tacit.confirmatory-ood-synthetic.v0.1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Synthetic assumed distributions only; no model data or sample-size adequacy claim",
        "method": "exact binomial enumeration, floating-point arithmetic, no Monte Carlo",
        "alpha_each": ALPHA,
        "family_alpha": .05,
        "practical_margin": MARGIN,
        "dependence": "D_t=X, D_l=-X, J=2X; contrasts perfectly dependent; iid splits",
        "scenarios": {
            name: {"probability_high": p, "support_X": [low, high],
                   "results": [exact_scenario(n, p, low, high) for n in counts]}
            for name, (p, low, high) in scenarios.items()
        },
        "precision_only_not_power": {
            "target_unclipped_radius": .05,
            "D_required_splits": math.ceil(2 * math.log(2 / ALPHA) / .05**2),
            "J_required_splits": math.ceil(8 * math.log(2 / ALPHA) / .05**2),
            "warning": "Radius is not rejection probability; these are not chosen sample sizes",
        },
    }


class SyntheticChecks(unittest.TestCase):
    def test_formula_and_interaction_scale(self):
        radius = math.sqrt(2 * math.log(80) / 100)
        self.assertAlmostEqual(interval(0, 100, -1, 1)[1], radius)
        self.assertAlmostEqual(interval(0, 100, -2, 2)[1], 2 * radius)
        self.assertGreater(interval(0, 100, -1, 1)[1],
                           interval(0, 100, -1, 1, alpha=.05)[1])

    def test_boundaries_and_degenerate_data(self):
        self.assertEqual(interval(0, 1, -1, 1), (-1, 1))
        self.assertEqual(interval(1, 20, -1, 1)[1], 1)
        self.assertEqual(interval(-1, 20, -1, 1)[0], -1)
        # Constant observations do not imply zero population uncertainty.
        self.assertGreater(interval(0, 20, -1, 1)[1], 0)

    def test_invalid_intervals(self):
        for mean, n, low, high, alpha in (
            (0, 0, -1, 1, .025), (0, 1.5, -1, 1, .025),
            (0, True, -1, 1, .025), (float("nan"), 2, -1, 1, .025),
            (0, 2, 1, -1, .025), (2, 2, -1, 1, .025),
            (0, 2, -1, 1, 0), (0, 2, -1, 1, 1),
        ):
            with self.subTest(args=(mean, n, low, high, alpha)):
                with self.assertRaises(ValueError):
                    interval(mean, n, low, high, alpha)

    def test_pairing_and_no_episode_pseudoreplication(self):
        effects = split_contrasts(*make_fixture())
        self.assertEqual(effects, [(0, 0), (.25, .25), (.5, .5)])
        repeated = split_contrasts(*make_fixture(repeats=20))
        self.assertEqual(effects, repeated)
        self.assertEqual(len(effects), 3)
        mean = math.fsum(d for d, _ in effects) / len(effects)
        self.assertEqual(interval(mean, len(effects), -1, 1),
                         interval(mean, len(repeated), -1, 1))

    def test_reject_duplicate_missing_unknown_and_nonbinary(self):
        rows, expected = make_fixture()
        corruptions = [rows + [rows[0]], rows[1:],
                       rows + [(99, 0, "A", "tight", 1)],
                       [(*rows[0][:-1], .5)] + rows[1:],
                       [r for r in rows if r[0] != 0]]
        for damaged in corruptions:
            with self.assertRaises(ValueError):
                split_contrasts(damaged, expected)
        with self.assertRaises(ValueError):
            split_contrasts(rows, {0: {0}, 1: {0, 1}})

    def test_binomial_mass_and_known_case(self):
        self.assertEqual(binomial_weights(2, 0), [1, 0, 0])
        self.assertEqual(binomial_weights(2, 1), [0, 0, 1])
        for got, wanted in zip(binomial_weights(2, .5), [.25, .5, .25]):
            self.assertAlmostEqual(got, wanted)
        self.assertAlmostEqual(math.fsum(binomial_weights(1000, .6)), 1)

    def test_exact_coverage_and_family_error_scenarios(self):
        for scenario in report()["scenarios"].values():
            for result in scenario["results"]:
                self.assertLessEqual(result["noncoverage_D"], ALPHA + 1e-12)
                self.assertLessEqual(result["noncoverage_J"], ALPHA + 1e-12)
                self.assertLessEqual(result["simultaneous_noncoverage"], .05 + 1e-12)
                self.assertLessEqual(result["any_true_null_rejection"], .05 + 1e-12)

    def test_practical_null_boundaries(self):
        h1 = exact_scenario(100, .525, -1, 1)
        h2 = exact_scenario(100, .5125, -1, 1)
        self.assertEqual(h1["assumed_mean_D"], MARGIN)
        self.assertEqual(h2["assumed_mean_J"], MARGIN)
        self.assertEqual(h1["any_true_null_rejection"], h1["practical_rejection_D"])

    def test_insufficient_clusters_and_positive_sanity(self):
        self.assertEqual(exact_scenario(4, .6, -1, 1)["practical_rejection_D"], 0)
        self.assertGreater(exact_scenario(1000, .6, -1, 1)["practical_rejection_D"], .9)
        self.assertEqual(exact_scenario(20, .5, .125, .25)["practical_rejection_D"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        print(json.dumps(report(), indent=2, sort_keys=True, allow_nan=False))
