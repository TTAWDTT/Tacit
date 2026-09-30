"""Executable offline entry: public fixtures + fake callbacks only, no model path."""
import hashlib
import json
from pathlib import Path
import tempfile
from .budget import Charge, GateError, Ledger, Outcome, Policy
from .isolation import Scope
from experiments.emergent_ood_v0_4.split import build_split


def run_demo():
    # One row/stage is enough to test the gate, never a model experiment.
    with tempfile.TemporaryDirectory(prefix='tacit-m2-offline-') as directory:
        root = Path(directory)
        split = build_split(seed=17)
        held = split['held_out_meaning_ids']
        values = {r['meaning_id']: r['values'] for r in split['meanings']}
        files = {'sender': {}}
        for stage, mid in [('train', split['train_meaning_ids'][0]), ('validation', held[0])]:
            row = {'private_meaning': dict(zip(split['attributes'], values[mid]))}
            raw = (json.dumps(row, sort_keys=True) + '\n').encode()
            filename = 'sender_' + stage + '.jsonl'
            (root / filename).write_bytes(raw)
            files['sender'][stage] = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(), records=1)
        scope = Scope(split, validation_ids=held[:16], test_ids=held[16:],
                      manifest=dict(split_sha256=split['split_sha256'], files=files), receiver='fake@v1')
        _, train = scope.read_development(root, stage='train')
        _, validation = scope.read_development(root, stage='validation')
        policy = Policy(per_request={'synthetic_units': 10}, cumulative={'synthetic_units': 10}, max_calls=7)
        charge = lambda work, context=0: Charge({'synthetic_units': work}, {'synthetic_units': context})
        ledger = Ledger(root / 'ledger.sqlite', policy)
        try:
            inventory = scope.inventory({'costly': 'public card A', 'cheap': 'public card B'},
                                        training_source=train, ledger=ledger)
            for phase, source in [('train', train), ('validation', validation)]:
                for name in ('costly', 'cheap'):
                    scope.development_call(ledger, source, row_index=0, candidate=name, phase=phase,
                                            charge=charge(1), operation=lambda name=name:
                                            Outcome(None, {'synthetic_units': 1}, name == 'costly'))
            plans = dict(costly=dict(deployment=[], per_use=[charge(11)]),
                         cheap=dict(deployment=[charge(2)], per_use=[charge(1, 1)]))
            artifact = scope.freeze(inventory, validation_source=validation,
                                    successes={'costly': [True], 'cheap': [False]}, plans=plans,
                                    discovery={n: [charge(1), charge(1)] for n in plans}, reuse=2, ledger=ledger)
            ledger.run('setup', phase='deployment', candidate='cheap', binding=scope.binding,
                        charge=charge(2), operation=lambda: Outcome(None, {'synthetic_units': 2}, True))
            for i in range(2):
                scope.dispatch(ledger, artifact, event_id=f'execution-{i}',
                               operation=lambda: Outcome('fake-result', {'synthetic_units': 2}, True))
            denied = False
            try:
                scope.dispatch(ledger, artifact, event_id='one-too-many',
                               operation=lambda: (_ for _ in ()).throw(AssertionError('must not enter')))
            except GateError:
                denied = True
            return dict(kind='offline-gates-not-model-evidence', model_calls=0,
                        selected=artifact['candidate'], feasible=artifact['feasibility'],
                        exhausted_schedule_denied=denied, events=ledger.snapshot())
        finally:
            ledger.close()


if __name__ == '__main__':
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
