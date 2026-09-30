"""Print hypothetical cost arithmetic; no LLM or task-performance evidence."""
import json
from .reference import Cost, break_even


def main():
    a, b = Cost('illustrative_units', 100, 20, 6), Cost('illustrative_units', 0, 0, 10)
    print(json.dumps({'evidence': 'invented costs, no model results',
        'reuse': [{'n': n, 'adaptive': str(a.average(n)), 'baseline': str(b.average(n))}
                  for n in (1, 10, 30, 100, 1000)],
        'break_even': break_even(a, b),
        'repeated_context_counterexample': break_even(Cost('illustrative_units', 100, 20, 11), b)}, indent=2))


if __name__ == '__main__':
    main()
