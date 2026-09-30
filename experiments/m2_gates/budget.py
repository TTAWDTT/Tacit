"""Offline extension: vector feasibility and durable pre-dispatch reservation.

No model client. Bounds must come from a reviewed adapter; this is not a sandbox.
Conservative reservation is never refunded, even after a failed attempt.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import sqlite3
from types import MappingProxyType


class GateError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def amount(value):
    if value is None or isinstance(value, (bool, float)):
        raise GateError('unknown/boolean/float amount: use exact integer or decimal string')
    try:
        result = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise GateError('invalid amount') from exc
    if result < 0:
        raise GateError('negative amount')
    return result


def vector(values, axes):
    if not isinstance(values, Mapping) or set(values) != set(axes):
        raise GateError('missing, extra or incomparable cost axis')
    return {a: amount(values[a]) for a in axes}


def packed(values):
    return {a: str(v) for a, v in values.items()}


@dataclass(frozen=True)
class Charge:
    # context is explicit on EVERY attempt, never hidden in amortized setup.
    work: Mapping
    context: Mapping

    def __post_init__(self):
        # Detach once, before any wait/transaction; values are exact and immutable.
        for name in ('work', 'context'):
            source = getattr(self, name)
            if not isinstance(source, Mapping):
                raise GateError('cost component must be a mapping')
            copied = dict(source)
            if any(not isinstance(a, str) or not a for a in copied):
                raise GateError('cost axes must be nonempty strings')
            object.__setattr__(self, name, MappingProxyType({a: amount(v) for a, v in copied.items()}))

    def total(self, axes):
        work, context = vector(self.work, axes), vector(self.context, axes)
        return {a: work[a] + context[a] for a in axes}


@dataclass(frozen=True)
class Policy:
    axes: tuple
    per_request: object
    cumulative: object
    max_calls: int
    identity: str

    def __init__(self, *, per_request, cumulative, max_calls):
        if not isinstance(per_request, dict) or not per_request or any(
                not isinstance(a, str) or not a for a in per_request):
            raise GateError('declare nonempty unit-qualified axes')
        if type(max_calls) is not int or max_calls < 0:
            raise GateError('invalid call ceiling')
        object.__setattr__(self, 'axes', tuple(sorted(per_request)))
        object.__setattr__(self, 'per_request', MappingProxyType(vector(per_request, self.axes)))
        object.__setattr__(self, 'cumulative', MappingProxyType(vector(cumulative, self.axes)))
        object.__setattr__(self, 'max_calls', max_calls)
        object.__setattr__(self, 'identity', digest(dict(per_request=packed(self.per_request),
                                    cumulative=packed(self.cumulative), max_calls=max_calls)))

    def check(self, charges):
        charges = list(charges)
        totals = {a: Fraction(0) for a in self.axes}
        reasons = []
        if len(charges) > self.max_calls:
            reasons.append('call_ceiling')
        for charge in charges:
            values = charge.total(self.axes)
            for a, value in values.items():
                totals[a] += value
                if value > self.per_request[a]:
                    reasons.append('per_request:' + a)
        for a in self.axes:
            if totals[a] > self.cumulative[a]:
                reasons.append('cumulative:' + a)
        return dict(feasible=not reasons, reasons=sorted(set(reasons)),
                    calls=len(charges), reserved=packed(totals))


def frontier(plans, discovery, *, policy, reuse, prior=()):
    """All search/failed-candidate attempts charge every deployment comparison.

    plans[name] = {'deployment': [Charge,...], 'per_use': [Charge,...]}.
    discovery[name] is the complete proposal/train/validation attempt list for
    that candidate. Shared setup can be assigned once to one inventory entry.
    No observed success ranking is performed on infeasible candidates.
    """
    if type(reuse) is not int or reuse < 1 or not plans or set(plans) != set(discovery):
        raise GateError('positive reuse and complete candidate discovery inventory required')
    prior = list(prior)
    prior_totals = policy.check(prior)['reserved']
    search = []
    for name in sorted(plans):
        if not discovery[name]:
            raise GateError('missing candidate discovery, including failed candidates')
        search.extend(discovery[name])
    output = {}
    for name, plan in plans.items():
        if set(plan) != {'deployment', 'per_use'} or not plan['per_use']:
            raise GateError('deployment and recurring attempts required')
        fixed = policy.check(prior + search + list(plan['deployment']))
        recurring = policy.check(plan['per_use'])
        totals = {a: Fraction(fixed['reserved'][a]) + reuse * Fraction(recurring['reserved'][a])
                  for a in policy.axes}
        calls = fixed['calls'] + reuse * recurring['calls']
        reasons = set(fixed['reasons'] + recurring['reasons'])
        if calls > policy.max_calls:
            reasons.add('call_ceiling')
        reasons.update('cumulative:' + a for a in policy.axes if totals[a] > policy.cumulative[a])
        output[name] = dict(feasible=not reasons, reasons=sorted(reasons), calls=calls,
                            reserved=packed(totals), prior_reserved=prior_totals,
                            average_reserved=packed({a: (v - Fraction(prior_totals[a])) / reuse for a, v in totals.items()}))
    return output


@dataclass(frozen=True)
class Outcome:
    value: object
    actual: dict
    success: bool  # supplied strict scorer result; this module never changes scoring


class Ledger:
    """Single-flight SQLite controller, durable before any callback is entered.

    All calls (proposal, onboarding, retries, failures, evaluation) use this path.
    The path and policy must remain fixed for the approved run. No refunds/reset API.
    Unfinished attempts or unbounded actual usage halt subsequent dispatch.
    """
    def __init__(self, path, policy):
        self.policy = policy
        self.db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS policy (identity TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, phase TEXT, '
                        'candidate TEXT, binding TEXT, reserved TEXT, actual TEXT, status TEXT, components TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS inventories (binding TEXT PRIMARY KEY, artifact TEXT)')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows = self.db.execute('SELECT identity FROM policy').fetchall()
            if not rows:
                self.db.execute('INSERT INTO policy VALUES (?)', (policy.identity,))
            elif rows != [(policy.identity,)]:
                raise GateError('ledger policy mismatch; cannot reset budget')
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def register_inventory(self, binding, artifact):
        artifact = json.loads(json.dumps(artifact, allow_nan=False))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.inventory_for(binding)
            if old is not None and old != artifact:
                raise GateError('inventory already sealed')
            if old is None:
                if any(r['binding'] == binding and r['phase'] in ('validation', 'execution') for r in self.snapshot()):
                    raise GateError('cannot propose inventory after validation')
                self.db.execute('INSERT INTO inventories VALUES (?,?)', (binding, json.dumps(artifact)))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def inventory_for(self, binding):
        row = self.db.execute('SELECT artifact FROM inventories WHERE binding=?', (binding,)).fetchone()
        return json.loads(row[0]) if row else None

    def snapshot(self):
        rows = self.db.execute('SELECT id,phase,candidate,binding,reserved,actual,status,components FROM events ORDER BY rowid').fetchall()
        return [dict(id=r[0], phase=r[1], candidate=r[2], binding=r[3],
                     reserved=json.loads(r[4]), actual=json.loads(r[5]) if r[5] else None,
                     status=r[6], components=json.loads(r[7])) for r in rows]

    def run(self, event_id, *, phase, candidate, binding, charge, operation, expected_execution_index=None, expected_ledger_hash=None):
        if any(not isinstance(v, str) or not v for v in (event_id, candidate, binding)):
            raise GateError('event/candidate/binding identity required')
        if phase not in ('proposal', 'train', 'validation', 'deployment', 'execution', 'retry', 'scoring'):
            raise GateError('unknown event phase')
        policy = self.policy  # pin the same immutable policy across lock acquisition/receipt
        charge = Charge(charge.work, charge.context)
        values = charge.total(policy.axes)  # one canonical immutable request snapshot
        components = dict(work=packed(charge.work), context=packed(charge.context))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT identity FROM policy').fetchall() != [(policy.identity,)]:
                raise GateError('active policy differs from ledger')
            rows = self.snapshot()
            if expected_ledger_hash is not None and digest(rows) != expected_ledger_hash:
                raise GateError('ledger changed concurrently; recheck preflight')
            if any(r['status'] in ('inflight', 'halted') for r in rows):
                raise GateError('unresolved or halted attempt; no further dispatch')
            if any(r['id'] == event_id for r in rows):
                raise GateError('duplicate attempt ID')
            if expected_execution_index is not None:
                count = sum(r['phase'] == 'execution' and r['candidate'] == candidate
                            and r['binding'] == binding for r in rows)
                if count != expected_execution_index:
                    raise GateError('execution schedule changed concurrently')
            zero = {a: 0 for a in policy.axes}
            old = [Charge(r['reserved'], zero) for r in rows]
            result = policy.check(old + [charge])
            if not result['feasible']:
                raise GateError('budget denied: ' + ','.join(result['reasons']))
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?)',
                            (event_id, phase, candidate, binding, json.dumps(packed(values)), None, 'inflight',
                             json.dumps(components)))
            self.db.execute('COMMIT')  # reservation survives crash before callback/receipt
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        try:
            outcome = operation()
            if not isinstance(outcome, Outcome) or type(outcome.success) is not bool:
                raise GateError('missing strict outcome/cost receipt')
            actual = vector(outcome.actual, policy.axes)
            self.db.execute('UPDATE events SET actual=? WHERE id=?', (json.dumps(packed(actual)), event_id))
            if any(actual[a] > values[a] for a in policy.axes):
                raise GateError('adapter violated reserved bound')
            self.db.execute('UPDATE events SET actual=?,status=? WHERE id=?',
                            (json.dumps(packed(actual)), 'success' if outcome.success else 'failed', event_id))
            return outcome.value
        except BaseException:
            self.db.execute('UPDATE events SET status=? WHERE id=?', ('halted', event_id))
            raise
