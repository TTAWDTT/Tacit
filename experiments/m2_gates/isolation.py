"""Split-bound development input and protocol artifacts; no test ledger reader.

Trusted setup supplies split and role-manifest hashes. Hashes detect accidental
misbinding/tampering, not dishonest provenance or an operator bypassing this API.
"""
import json
from pathlib import Path
import hashlib
from .budget import Charge, GateError, digest, frontier, packed, vector
from experiments.emergent_ood_v0_4.split import validate_split


def sealed(body):
    return dict(body, sha256=digest(body))


def verified(record):
    body = dict(record)
    claimed = body.pop('sha256', None)
    if digest(body) != claimed:
        raise GateError('artifact digest mismatch')
    return body


class Scope:
    def __init__(self, split, *, validation_ids, test_ids, manifest, receiver):
        validate_split(split)
        if not isinstance(receiver, str) or not receiver:
            raise GateError('frozen receiver/config identity required')
        train, val, test = set(split['train_meaning_ids']), set(validation_ids), set(test_ids)
        if not val or not test or val & test or val | test != set(split['held_out_meaning_ids']):
            raise GateError('validation/test must partition held-out support')
        if manifest.get('split_sha256') != split['split_sha256']:
            raise GateError('manifest split mismatch')
        self.binding = digest(dict(split_sha256=split['split_sha256'], manifest_sha256=digest(manifest),
                                   receiver=receiver, train=sorted(train), validation=sorted(val), test=sorted(test)))
        self.support = dict(train=train, validation=val)
        self.attributes = tuple(split['attributes'])
        self.meanings = {tuple(r['values']): r['meaning_id'] for r in split['meanings']}
        # Defensive snapshots: later caller mutation cannot retarget the scope.
        self.files = json.loads(json.dumps(manifest['files']))
        self.receiver = receiver

    def read_development(self, root, *, stage):
        if stage not in self.support:
            raise GateError('test/gold/unknown stage unavailable to discovery')
        entry = self.files['sender'][stage]
        root = Path(root).resolve()
        path = (root / entry['file']).resolve()
        if not path.is_relative_to(root) or path.name != 'sender_' + stage + '.jsonl':
            raise GateError('role path escape or incorrect role filename')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise GateError('role file hash mismatch')
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != entry['records'] or not rows:
            raise GateError('role row count mismatch')
        clean, ids = [], []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'private_meaning'}:
                raise GateError('metadata/answer/receiver fields forbidden')
            meaning = row['private_meaning']
            if not isinstance(meaning, dict) or set(meaning) != set(self.attributes):
                raise GateError('sender meaning schema mismatch')
            values = tuple(meaning[a] for a in self.attributes)
            try:
                meaning_id = self.meanings.get(values)
            except TypeError as exc:
                raise GateError('non-atomic meaning') from exc
            if meaning_id not in self.support[stage]:
                raise GateError('meaning outside this split/stage support')
            ids.append(meaning_id)
            clean.append({'private_meaning': dict(meaning)})
        provenance = sealed(dict(binding=self.binding, stage=stage, role='sender',
                                 file_sha256=entry['sha256'], meaning_ids=ids))
        return clean, provenance

    def check_source(self, source, *, stage):
        source = verified(source)
        if source.get('binding') != self.binding or source.get('stage') != stage or source.get('role') != 'sender':
            raise GateError('cross-split, stage, receiver or role provenance misuse')
        if source.get('file_sha256') != self.files['sender'][stage]['sha256']:
            raise GateError('unregistered discovery source')
        if not source.get('meaning_ids') or not set(source['meaning_ids']) <= self.support[stage]:
            raise GateError('discovery support leakage')

    def inventory(self, candidates, *, training_source, ledger):
        self.check_source(training_source, stage='train')
        if not isinstance(candidates, dict) or not candidates or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in candidates.items()):
            raise GateError('nonempty fixed candidate-card inventory required')
        result = sealed(dict(binding=self.binding, candidates=dict(candidates), training=training_source))
        ledger.register_inventory(self.binding, result)
        return result

    def freeze(self, inventory, *, validation_source, successes, plans, discovery, reuse, ledger):
        inv = verified(inventory)
        if ledger.inventory_for(self.binding) != inventory:
            raise GateError('inventory was not sealed before validation')
        if inv.get('binding') != self.binding:
            raise GateError('inventory from another split or receiver')
        self.check_source(inv['training'], stage='train')
        self.check_source(validation_source, stage='validation')
        names = set(inv['candidates'])
        all_rows = ledger.snapshot()
        if any(r['status'] in ('inflight', 'halted') for r in all_rows):
            raise GateError('unresolved cost or in-flight attempt blocks freeze')
        if any(r['binding'] == self.binding and r['phase'] in ('deployment', 'execution') for r in all_rows):
            raise GateError('freeze must precede deployment and execution')
        history = self._discovery_history(ledger)
        history_ids = {r['id'] for r in history}
        zero = {a: 0 for a in ledger.policy.axes}
        prior = [Charge(r['reserved'], zero) for r in all_rows if r['id'] not in history_ids]
        feasible = frontier(plans, discovery, policy=ledger.policy, reuse=reuse, prior=prior)
        for name in names:
            rows = [r for r in history if r['candidate'] == name]
            expected = [packed(c.total(ledger.policy.axes)) for c in discovery.get(name, [])]
            if [r['reserved'] for r in rows] != expected or any(
                    r['status'] not in ('success', 'failed') for r in rows):
                raise GateError('missing or uncharged discovery/failed candidate')
            for phase, source in [('train', inv['training']), ('validation', validation_source)]:
                expected_ids = [self.binding + '/' + phase + '/' + name + '/' + str(i)
                                for i in range(len(source['meaning_ids']))]
                if [r['id'] for r in rows if r['phase'] == phase] != expected_ids:
                    raise GateError('discovery row provenance incomplete or out of order')
            observed = [r['status'] == 'success' for r in rows if r['phase'] == 'validation']
            if observed != successes.get(name):
                raise GateError('validation observations differ from charged ledger')
        if {r['candidate'] for r in history} != names:
            raise GateError('unlisted search candidate')
        if set(successes) != names or set(feasible) != names:
            raise GateError('missing failed candidate or feasibility evidence')
        for values in successes.values():
            if not isinstance(values, list) or not values or any(type(v) is not bool for v in values):
                raise GateError('complete paired strict success rows required')
        if {len(v) for v in successes.values()} != {len(validation_source['meaning_ids'])}:
            raise GateError('unpaired candidate observations')
        eligible = [n for n in names if feasible[n]['feasible'] is True]
        if not eligible:
            raise GateError('no budget-feasible candidate')
        chosen = min(eligible, key=lambda n: (-sum(successes[n]), n))
        schedule = [dict(work=packed(vector(c.work, ledger.policy.axes)),
                         context=packed(vector(c.context, ledger.policy.axes)))
                    for c in plans[chosen]['per_use']]
        deployment = [packed(c.total(ledger.policy.axes)) for c in plans[chosen]['deployment']]
        return sealed(dict(binding=self.binding, receiver=self.receiver, candidate=chosen,
                           policy=ledger.policy.identity, discovery_hash=digest(history),
                           deployment=deployment, schedule=schedule, reuse=reuse,
                           card=inv['candidates'][chosen], inventory=inventory,
                           validation=validation_source, successes=successes, feasibility=feasible,
                           evidence='offline extension; model execution not authorized'))

    def check_artifact(self, artifact):
        body = verified(artifact)
        if body.get('binding') != self.binding or body.get('receiver') != self.receiver:
            raise GateError('protocol belongs to another split/manifest/receiver')
        inv = verified(body['inventory'])
        if inv['binding'] != self.binding:
            raise GateError('cross-split discovery inventory')
        self.check_source(inv['training'], stage='train')
        self.check_source(body['validation'], stage='validation')
        if inv['candidates'].get(body['candidate']) != body['card']:
            raise GateError('selected card differs from frozen inventory')
        return body

    def _discovery_history(self, ledger):
        return [r for r in ledger.snapshot() if r['binding'] == self.binding and
                r['phase'] in ('proposal', 'train', 'validation', 'retry')]

    def development_call(self, ledger, source, *, row_index, candidate, phase, charge, operation):
        if phase not in ('train', 'validation'):
            raise GateError('development phase unavailable')
        self.check_source(source, stage=phase)
        inv = ledger.inventory_for(self.binding)
        if inv is None or candidate not in inv['candidates']:
            raise GateError('candidate not in pre-validation inventory')
        rows = ledger.snapshot()
        count = sum(r['binding'] == self.binding and r['candidate'] == candidate and r['phase'] == phase for r in rows)
        if type(row_index) is not int or row_index != count or not 0 <= row_index < len(source['meaning_ids']):
            raise GateError('duplicate, skipped or exhausted development row index')
        if phase == 'validation' and any(
                sum(r['binding'] == self.binding and r['candidate'] == n and r['phase'] == 'train' for r in rows)
                != len(inv['training']['meaning_ids']) for n in inv['candidates']):
            raise GateError('all candidate training rows must precede validation')
        event_id = self.binding + '/' + phase + '/' + candidate + '/' + str(row_index)
        if phase == 'train' and any(r['binding'] == self.binding and r['phase'] == 'validation' for r in rows):
            raise GateError('training after validation forbidden')
        return ledger.run(event_id, phase=phase, candidate=candidate, binding=self.binding,
                          charge=charge, operation=operation, expected_ledger_hash=digest(rows))

    def dispatch(self, ledger, artifact, *, event_id, operation):
        body = self.check_artifact(artifact)  # before reserving or calling anything
        if ledger.policy.identity != body['policy']:
            raise GateError('execution policy differs from freeze')
        if digest(self._discovery_history(ledger)) != body['discovery_hash']:
            raise GateError('discovery ledger missing, changed or extended after freeze')
        rows = ledger.snapshot()
        deployment = [r for r in rows if r['phase'] == 'deployment' and
                      r['binding'] == self.binding and r['candidate'] == body['candidate']]
        if [r['reserved'] for r in deployment] != body['deployment'] or any(
                r['status'] != 'success' for r in deployment):
            raise GateError('deployment costs not charged')
        count = sum(r['phase'] == 'execution' and r['binding'] == self.binding and
                    r['candidate'] == body['candidate'] for r in rows)
        if count >= body['reuse'] * len(body['schedule']):
            raise GateError('frozen reuse schedule exhausted')
        charge = Charge(**body['schedule'][count % len(body['schedule'])])
        return ledger.run(event_id, phase='execution', candidate=body['candidate'],
                          binding=self.binding, charge=charge, operation=operation,
                          expected_execution_index=count, expected_ledger_hash=digest(rows))
