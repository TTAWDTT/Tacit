"""Offline research primitives. No model client, task runner, or test-file reader."""
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def nonnegative(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a cost')
    result = Fraction(str(value))
    if result < 0:
        raise ValueError('negative cost')
    return result


@dataclass(frozen=True)
class Cost:
    """One declared axis only; unknown is None, never imputed zero."""
    unit: str
    discovery: object
    deployment: object
    per_use: object

    def __post_init__(self):
        if not self.unit:
            raise ValueError('unit required (token axes must include tokenizer/role)')
        for name in ('discovery', 'deployment', 'per_use'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, nonnegative(value))

    def average(self, n):
        if type(n) is not int or n < 1:
            raise ValueError('reuse must be a positive integer')
        if None in (self.discovery, self.deployment, self.per_use):
            return None
        return (self.discovery + self.deployment) / n + self.per_use


def break_even(a, b):
    """Integer interval where a costs <= b. Cost-only; no accuracy claim."""
    if a.unit != b.unit:
        raise ValueError('incomparable axes')
    if a.average(1) is None or b.average(1) is None:
        return {'status': 'unknown', 'first': None, 'last': None}
    fixed = a.discovery + a.deployment - b.discovery - b.deployment
    saving = b.per_use - a.per_use
    if saving == 0:
        return {'status': 'all' if fixed <= 0 else 'never',
                'first': 1 if fixed <= 0 else None, 'last': None}
    if saving > 0:
        threshold = fixed / saving
        first = max(1, -(-threshold.numerator // threshold.denominator))
        return {'status': 'eventual', 'first': first, 'last': None}
    threshold = fixed / saving
    last = threshold.numerator // threshold.denominator
    return {'status': 'temporary' if last >= 1 else 'never',
            'first': 1 if last >= 1 else None, 'last': last if last >= 1 else None}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    card: str
    # Explicit primitive bijection, not candidate IDs or whole-test-tuple labels.
    mapping: tuple

    def __post_init__(self):
        if not self.name or not self.family or not isinstance(self.card, str):
            raise ValueError('candidate identity/card missing')
        if not isinstance(self.mapping, tuple) or not self.mapping:
            raise ValueError('immutable mapping required')
        if any(not isinstance(p, tuple) or len(p) != 2 or
               any(not isinstance(v, str) or not v for v in p) for p in self.mapping):
            raise ValueError('mapping must contain string pairs')
        if len({p[0] for p in self.mapping}) != len(self.mapping) or len({p[1] for p in self.mapping}) != len(self.mapping):
            raise ValueError('mapping must be a bijection')


@dataclass(frozen=True)
class Observation:
    candidate: str
    receiver: str
    stage: str
    episode: str
    strict_success: bool
    # Complete selected-axis evaluation cost, including invalid/rejected calls.
    cost: object


def select(candidates, observations, *, receiver, train_ids, validation_ids,
           unit, proposal_cost, deployment_cost, max_candidates):
    """Two-stage fixed-inventory selection: train audit, validation ranking.

    No adaptive proposals from validation. Caller must supply trusted split IDs;
    declarations alone cannot prove provenance. All inventory candidates are charged.
    """
    candidates, observations = tuple(candidates), tuple(observations)
    if not receiver:
        raise ValueError('receiver required')
    if not unit or type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('unit and positive inventory budget required')
    if not candidates or len(candidates) > max_candidates:
        raise ValueError('candidate budget')
    support = {p[0] for p in candidates[0].mapping}
    if any({p[0] for p in c.mapping} != support for c in candidates):
        raise ValueError('candidate meaning coverage differs')
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError('duplicate candidate')
    train_ids, validation_ids = set(train_ids), set(validation_ids)
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError('nonempty disjoint development supports required')
    expected = {(name, stage, eid) for name in names
                for stage, ids in [('train', train_ids), ('validation', validation_ids)] for eid in ids}
    seen = set()
    total = nonnegative(proposal_cost)
    scores = {name: 0 for name in names}
    costs = {name: Fraction(0) for name in names}
    for row in observations:
        key = row.candidate, row.stage, row.episode
        if row.receiver != receiver or key not in expected or key in seen or type(row.strict_success) is not bool:
            raise ValueError('unpaired, duplicate, wrong receiver, test, or malformed feedback')
        seen.add(key)
        cost = nonnegative(row.cost)
        total += cost
        if row.stage == 'validation':
            scores[row.candidate] += int(row.strict_success)
            costs[row.candidate] += cost
    if seen != expected:
        raise ValueError('missing candidate results, including failed candidates')
    # No validation-driven mutation. Stable identity breaks exact ties.
    chosen = min(names, key=lambda name: (-scores[name], costs[name], name))
    candidate = next(c for c in candidates if c.name == chosen)
    body = {'schema': 'tacit.receiver-adaptation.offline.v1',
            'receiver': receiver, 'candidate': asdict(candidate),
            'inventory_hash': digest([asdict(c) for c in sorted(candidates, key=lambda c: c.name)]),
            'feedback_hash': digest([dict(asdict(r), cost=str(nonnegative(r.cost))) for r in sorted(observations, key=lambda r: (r.candidate, r.stage, r.episode))]),
            'train_ids_hash': digest(sorted(train_ids)), 'validation_ids_hash': digest(sorted(validation_ids)),
            'selection_rule': 'validation strict successes descending, cost ascending, name ascending',
            'unit': unit, 'discovery_cost': str(total),
            'deployment_cost': str(nonnegative(deployment_cost)), 'max_candidates': max_candidates,
            'evidence': 'offline supplied observations; not model evidence'}
    return dict(body, freeze_sha256=digest(body))


def verify_freeze(artifact):
    body = dict(artifact)
    claimed = body.pop('freeze_sha256')
    if digest(body) != claimed:
        raise ValueError('freeze changed')
    return body


def shuffled_mapping(candidate):
    """Deterministic cyclic derangement: preserves labels and their byte multiset."""
    if len(candidate.mapping) < 2:
        raise ValueError('cannot derange one label')
    labels = [p[1] for p in candidate.mapping]
    return tuple((p[0], labels[(i + 1) % len(labels)]) for i, p in enumerate(candidate.mapping))
