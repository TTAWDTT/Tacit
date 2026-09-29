from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any


SCHEMA = "tlu.holistic-protocol-onboarding.v1"


def holistic_onboarding_accuracy(*, meaning_count: int, distinct_examples: int) -> Fraction:
    """Bayes accuracy for one uniform test meaning under a random unknown bijection.

    A protocol assigns each of M meanings a unique symbol, but a newcomer has
    no prior alignment to the convention. It observes n distinct meaning-symbol
    examples and then decodes one uniformly sampled meaning. The unknown
    bijection is uniform over all M! possibilities and the newcomer is Bayes
    optimal. The last unobserved pair is inferable by elimination.
    """
    if isinstance(meaning_count, bool) or not isinstance(meaning_count, int) or meaning_count < 1:
        raise ValueError("meaning_count must be a positive integer")
    if isinstance(distinct_examples, bool) or not isinstance(distinct_examples, int):
        raise ValueError("distinct_examples must be an integer")
    if not 0 <= distinct_examples < meaning_count:
        raise ValueError("distinct_examples must be in 0..meaning_count-1")
    return Fraction(distinct_examples + 1, meaning_count)


def disjoint_holdout_candidate_accuracy(*, candidate_count: int) -> Fraction:
    """Bayes accuracy when every candidate is unseen under a random bijection.

    The receiver sees the candidate meanings and a transmitted code symbol;
    calibration examples cover meanings outside the candidate set. A uniform
    random unknown bijection makes the symbol independent of which candidate
    is the target, so this equals the no-message reference.
    """
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    return Fraction(1, candidate_count)


def build_report(*, meaning_count: int) -> dict[str, Any]:
    if isinstance(meaning_count, bool) or not isinstance(meaning_count, int) or meaning_count < 1:
        raise ValueError("meaning_count must be a positive integer")
    rows = []
    for examples in range(meaning_count):
        accuracy = holistic_onboarding_accuracy(
            meaning_count=meaning_count,
            distinct_examples=examples,
        )
        rows.append({
            "distinct_examples": examples,
            "expected_exact_accuracy": float(accuracy),
            "exact_fraction": f"{accuracy.numerator}/{accuracy.denominator}",
        })
    return {
        "schema": SCHEMA,
        "reference": "bayes_accuracy_for_uniform_random_unknown_holistic_bijection",
        "is_language_or_llm_result": False,
        "meaning_count": meaning_count,
        "assumptions": [
            "the protocol is a fixed bijection between M meanings and M symbols",
            "the newcomer has no prior information about the meaning-symbol bijection",
            "the bijection is uniformly random among M! possibilities",
            "calibration reveals n distinct, correct meaning-symbol pairs without noise",
            "the test meaning is uniform over all M meanings and is observed by the sender",
            "the receiver is Bayes optimal and observes the transmitted symbol",
            "calibration cost, test-message cost, and decoder compute are not included",
        ],
        "frontier": rows,
        "disjoint_holdout_reference": {
            "description": "when all candidate meanings are unseen in calibration, an arbitrary holistic bijection conveys no information about which candidate is the target",
            "expected_accuracy_for_candidate_count_k": "1/k",
        },
        "limits": [
            "a structured or pretrained receiver may have informative priors and outperform this reference",
            "the result concerns a one-to-one holistic lookup code, not compositional protocols",
            "candidate-set side information and multi-round interaction are not modeled",
            "the curve is a coding/identifiability reference, not evidence about LLM onboarding",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanings", type=int, required=True)
    args = parser.parse_args()
    try:
        report = build_report(meaning_count=args.meanings)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
