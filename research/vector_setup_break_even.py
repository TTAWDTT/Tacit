"""Exact componentwise break-even analysis for one-time protocol setup costs.

Inputs use integers, ``Fraction`` objects, or rational strings such as ``"3/8"``.
Floating point values are rejected so horizon decisions do not depend on
rounding near an integer boundary.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping


def _fraction(value: int | str | Fraction, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact non-negative rational")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, str):
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{name} must be an exact rational string") from exc
    else:
        raise ValueError(f"{name} must be an int, Fraction, or rational string")
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _format(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _ceil_ratio(numerator: Fraction, denominator: Fraction) -> int:
    quotient = numerator / denominator
    return (quotient.numerator + quotient.denominator - 1) // quotient.denominator


def vector_setup_break_even(
    *,
    setup_cost: Mapping[str, int | str | Fraction],
    baseline_per_episode: Mapping[str, int | str | Fraction],
    candidate_per_episode: Mapping[str, int | str | Fraction],
) -> dict[str, object]:
    """Return the smallest positive episode horizon for componentwise parity.

    Every key is a separate cost unit (for example ``wire_bytes`` or a named
    tokenizer). A finite horizon exists iff no dimension has negative
    per-episode savings and every zero-savings dimension has zero setup cost.
    The returned horizon is a cost result only; task utility must be assessed
    separately before claiming Pareto dominance.
    """
    keys = set(setup_cost)
    if not keys or set(baseline_per_episode) != keys or set(candidate_per_episode) != keys:
        raise ValueError("all three cost vectors must have the same non-empty dimensions")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("cost dimension names must be non-empty strings")

    details: dict[str, dict[str, object]] = {}
    blocking: list[str] = []
    horizons: list[int] = []
    for dimension in sorted(keys):
        setup = _fraction(setup_cost[dimension], name=f"setup_cost[{dimension}]")
        baseline = _fraction(
            baseline_per_episode[dimension], name=f"baseline_per_episode[{dimension}]"
        )
        candidate = _fraction(
            candidate_per_episode[dimension], name=f"candidate_per_episode[{dimension}]"
        )
        savings = baseline - candidate
        if savings < 0:
            status = "candidate_costs_more_per_episode"
            horizon = None
            blocking.append(dimension)
        elif savings == 0 and setup > 0:
            status = "positive_setup_without_marginal_savings"
            horizon = None
            blocking.append(dimension)
        else:
            status = "finite"
            horizon = max(1, _ceil_ratio(setup, savings)) if savings > 0 else 1
            horizons.append(horizon)
        details[dimension] = {
            "setup_cost": _format(setup),
            "baseline_per_episode": _format(baseline),
            "candidate_per_episode": _format(candidate),
            "baseline_minus_candidate": _format(savings),
            "horizon_episodes": horizon,
            "status": status,
        }

    finite = not blocking
    return {
        "schema_version": "tlu.vector-setup-break-even.v1",
        "finite_componentwise_break_even": finite,
        "horizon_episodes": max(horizons) if finite else None,
        "blocking_dimensions": blocking,
        "dimensions": details,
        "assumption": "Stationary per-episode costs and a positive integer reuse horizon; utility, uncertainty, and deployment-specific exchange rates are outside this cost-only result.",
    }
