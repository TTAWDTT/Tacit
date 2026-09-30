"""No models, endpoints, weights or private ledgers; temporary public fixtures only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.m2_gates.budget import Charge, GateError, Ledger, Outcome, Policy, frontier
from experiments.m2_gates.isolation import Scope
from experiments.m2_gates.demo import run_demo
from experiments.emergent_ood_v0_4.split import build_split


def cost(work=1, context=0):
    return Charge({'bytes': work}, {'bytes': context})


def policy(total=100, per=10, calls=100):
    return Policy(per_request={'bytes': per}, cumulative={'bytes': total}, max_calls=calls)


def fixture(root, seed=17, receiver='R@config', extra=None, train_override=None):
    split = build_split(seed=seed)
    by_id = {r['meaning_id']: dict(zip(split['attributes'], r['values'])) for r in split['meanings']}
    held = split['held_out_meaning_ids']
    val, test = held[:16], held[16:]
    files = {'sender': {}}
    for stage, mid in [('train', split['train_meaning_ids'][0]), ('validation', val[0])]:
        row = {'private_meaning': by_id[mid]}
        if stage == 'train' and train_override is not None:
            row['private_meaning'] = train_override
        if extra:
            row.update(extra)
        raw = (json.dumps(row) + '\n').encode()
        filename = 'sender_' + stage + '.jsonl'
        (root / filename).write_bytes(raw)
        files['sender'][stage] = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(), records=1)
    manifest = dict(split_sha256=split['split_sha256'], files=files)
    return Scope(split, validation_ids=val, test_ids=test, manifest=manifest, receiver=receiver), manifest


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / 'budget.sqlite', policy())
        self.addCleanup(self.ledger.close)

    def call(self, charge=cost(), success=True, event='e', ledger=None, actual=1):
        return (ledger or self.ledger).run(event, phase='train', candidate='a', binding='scope',
                                          charge=charge, operation=lambda: Outcome('ok', {'bytes': actual}, success))

    def test_boundary_reopen_and_failed_attempt_charged(self):
        ledger = Ledger(self.root / 'small.sqlite', policy(total=3, calls=2))
        self.call(cost(2), False, ledger=ledger)
        ledger.close()
        ledger = Ledger(self.root / 'small.sqlite', policy(total=3, calls=2))
        self.addCleanup(ledger.close)
        self.call(cost(1), event='second', ledger=ledger)
        self.assertEqual([r['status'] for r in ledger.snapshot()], ['failed', 'success'])
        with self.assertRaises(GateError): self.call(cost(0), event='third', ledger=ledger, actual=0)
        self.assertEqual(sum(int(r['reserved']['bytes']) for r in ledger.snapshot()), 3)

    def test_unknown_negative_bool_float_missing_extra_axis_no_callback(self):
        bad = [lambda: cost(None), lambda: cost(-1), lambda: cost(True), lambda: cost(1.1),
               lambda: Charge({}, {'bytes': 0}), lambda: Charge({'bytes': 1, 'tokens': 0}, {'bytes': 0})]
        for charge in bad:
            with self.subTest(charge=charge), self.assertRaises(GateError):
                self.ledger.run('x', phase='train', candidate='a', binding='s', charge=charge(),
                                operation=lambda: self.fail('callback entered'))
        self.assertEqual(self.ledger.snapshot(), [])

    def test_per_request_and_cumulative_context_reject_before_callback(self):
        self.call(cost(9, 1), actual=10)
        with self.assertRaises(GateError): self.call(cost(9, 2), event='over')
        small = Ledger(self.root / 'total.sqlite', policy(total=3))
        self.addCleanup(small.close)
        self.call(cost(1, 1), ledger=small, actual=2)
        with self.assertRaises(GateError): self.call(cost(1, 1), event='again', ledger=small, actual=2)
        self.assertEqual(len(small.snapshot()), 1)

    def test_policy_change_and_duplicate_ids_cannot_reset(self):
        self.call()
        with self.assertRaises(GateError): self.call()
        with self.assertRaises(GateError): Ledger(self.root / 'budget.sqlite', policy(total=1000))

    def test_policy_snapshot_and_in_memory_replacement_cannot_bypass(self):
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError): self.ledger.policy.max_calls = 1000
        with self.assertRaises(TypeError): self.ledger.policy.cumulative['bytes'] = 1000
        self.ledger.policy = policy(total=1000)
        with self.assertRaises(GateError): self.call()

    def test_zero_call_authorization(self):
        zero = Ledger(self.root / 'zero.sqlite', policy(calls=0))
        self.addCleanup(zero.close)
        with self.assertRaises(GateError): self.call(ledger=zero)

    def test_overrun_unknown_receipt_and_exception_halt(self):
        for index, operation in enumerate((lambda: Outcome(None, {'bytes': 2}, True),
                                            lambda: Outcome(None, {'bytes': None}, False),
                                            lambda: (_ for _ in ()).throw(RuntimeError('fake')))):
            ledger = Ledger(self.root / f'bad{index}.sqlite', policy())
            self.addCleanup(ledger.close)
            with self.assertRaises((GateError, RuntimeError)):
                ledger.run('bad', phase='train', candidate='a', binding='s', charge=cost(), operation=operation)
            self.assertEqual(ledger.snapshot()[0]['status'], 'halted')
            self.assertEqual(ledger.snapshot()[0]['reserved'], {'bytes': '1'})
            if index == 0:
                self.assertEqual(ledger.snapshot()[0]['actual'], {'bytes': '2'})
            with self.assertRaises(GateError): self.call(event='next', ledger=ledger)

    def test_reservation_visible_before_callback_and_concurrent_entry_blocked(self):
        second = Ledger(self.root / 'budget.sqlite', policy())
        self.addCleanup(second.close)
        def operation():
            self.assertEqual(second.snapshot()[0]['status'], 'inflight')
            with self.assertRaises(GateError): self.call(event='concurrent', ledger=second)
            return Outcome(None, {'bytes': 1}, True)
        self.ledger.run('x', phase='train', candidate='a', binding='s', charge=cost(), operation=operation)

    def test_crash_reservation_blocks_resume(self):
        self.ledger.db.execute("INSERT INTO events VALUES ('crash','train','a','s','{\"bytes\":\"1\"}',NULL,'inflight','{\"work\":{\"bytes\":\"1\"},\"context\":{\"bytes\":\"0\"}}')")
        second = Ledger(self.root / 'budget.sqlite', policy())
        self.addCleanup(second.close)
        with self.assertRaises(GateError): self.call(ledger=second)

    def test_frontier_charges_all_candidates_context_reuse_and_unknown(self):
        plans = {'a': {'deployment': [cost(1)], 'per_use': [cost(1, 2)]},
                 'b': {'deployment': [], 'per_use': [cost(2)]}}
        discovery = {'a': [cost(2)], 'b': [cost(4)]}  # b search charged even if it failed
        result = frontier(plans, discovery, policy=policy(total=11), reuse=2)
        self.assertFalse(result['a']['feasible'])  # 6+1+2*3 = 13
        self.assertEqual(result['a']['reserved']['bytes'], '13')
        self.assertEqual(result['b']['reserved']['bytes'], '10')
        self.assertTrue(result['b']['feasible'])
        with self.assertRaises(GateError): frontier(plans, {'a': [cost(2)]}, policy=policy(), reuse=2)
        with self.assertRaises(GateError):
            discovery['b'] = [cost(None)]
            frontier(plans, discovery, policy=policy(), reuse=2)

    def test_large_reuse_is_bounded_arithmetic_and_demo_is_offline(self):
        result = frontier({'a': dict(deployment=[], per_use=[cost()])}, {'a': [cost()]},
                          policy=policy(), reuse=10**12)
        self.assertFalse(result['a']['feasible'])
        self.assertEqual(result['a']['calls'], 10**12 + 1)
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            demo = run_demo()
        self.assertEqual(demo['model_calls'], 0)
        self.assertEqual(demo['selected'], 'cheap')
        self.assertTrue(demo['exhausted_schedule_denied'])
        self.assertEqual(demo['events'][-1]['components']['context'], {'synthetic_units': '1'})

    def test_stale_preflight_snapshot_rejected(self):
        from experiments.m2_gates.budget import digest
        before = digest(self.ledger.snapshot())
        self.call()
        with self.assertRaises(GateError):
            self.ledger.run('stale', phase='train', candidate='a', binding='scope', charge=cost(),
                            expected_ledger_hash=before, operation=lambda: self.fail())

    def test_fractional_axes_not_summed_or_amortized_past_request_cap(self):
        p = Policy(per_request={'usd': '0.2', 'tokens:R@hash': 10},
                   cumulative={'usd': '1', 'tokens:R@hash': 100}, max_calls=10)
        c = Charge({'usd': '0.1', 'tokens:R@hash': 11}, {'usd': '0', 'tokens:R@hash': 0})
        r = p.check([c])
        self.assertEqual(r['reasons'], ['per_request:tokens:R@hash'])


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scope, self.manifest = fixture(self.root)
        self.ledger = Ledger(self.root / 'budget.sqlite', policy())
        self.addCleanup(self.ledger.close)
        self.train_rows, self.train = self.scope.read_development(self.root, stage='train')
        _, self.val = self.scope.read_development(self.root, stage='validation')
        self.inv = self.scope.inventory({'a': 'card A', 'b': 'card B'}, training_source=self.train, ledger=self.ledger)
        self.plans = {'a': dict(deployment=[], per_use=[cost(11)]),
                      'b': dict(deployment=[cost(2)], per_use=[cost(1, 1)])}
        self.discovery = {name: [cost(), cost()] for name in ('a', 'b')}
        self.successes = {'a': [True], 'b': [False]}

    def develop(self):
        for stage, source in [('train', self.train), ('validation', self.val)]:
            for name in ('a', 'b'):
                self.scope.development_call(self.ledger, source, row_index=0, candidate=name,
                                            phase=stage, charge=cost(),
                                            operation=lambda name=name: Outcome(None, {'bytes': 1}, name == 'a'))

    def freeze(self):
        return self.scope.freeze(self.inv, validation_source=self.val, successes=self.successes,
                                 plans=self.plans, discovery=self.discovery, reuse=2, ledger=self.ledger)

    def test_feasible_selection_failed_search_deployment_and_reuse_enforced(self):
        self.develop()
        artifact = self.freeze()
        self.assertEqual(artifact['candidate'], 'b')  # successful a is infeasible
        self.assertEqual(artifact['feasibility']['b']['reserved']['bytes'], '10')
        with self.assertRaises(GateError): self.scope.dispatch(self.ledger, artifact, event_id='before', operation=lambda: self.fail())
        self.ledger.run('setup', phase='deployment', candidate='b', binding=self.scope.binding,
                        charge=cost(2), operation=lambda: Outcome(None, {'bytes': 2}, True))
        for index in range(2):
            self.scope.dispatch(self.ledger, artifact, event_id=f'execute{index}',
                                operation=lambda: Outcome('answer', {'bytes': 2}, True))
        with self.assertRaises(GateError): self.scope.dispatch(self.ledger, artifact, event_id='third', operation=lambda: self.fail())
        self.assertEqual(sum(int(r['reserved']['bytes']) for r in self.ledger.snapshot()), 10)

    def test_failed_deployment_charged_but_cannot_authorize_execution(self):
        self.develop(); artifact = self.freeze()
        self.ledger.run('bad-setup', phase='deployment', candidate='b', binding=self.scope.binding,
                        charge=cost(2), operation=lambda: Outcome(None, {'bytes': 2}, False))
        with self.assertRaises(GateError):
            self.scope.dispatch(self.ledger, artifact, event_id='blocked', operation=lambda: self.fail())
        self.assertEqual(self.ledger.snapshot()[-1]['reserved'], {'bytes': '2'})

    def test_unpaid_or_omitted_failed_candidate_rejected(self):
        with self.assertRaises(GateError): self.freeze()
        self.develop()
        self.discovery['b'] = [cost()]
        with self.assertRaises(GateError): self.freeze()

    def test_relabelled_validation_scores_rejected(self):
        self.develop()
        self.successes['b'] = [True]
        with self.assertRaises(GateError): self.freeze()

    def test_inventory_cannot_change_or_train_after_validation(self):
        self.develop()
        with self.assertRaises(GateError): self.scope.inventory({'new': 'new'}, training_source=self.train, ledger=self.ledger)
        with self.assertRaises(GateError): self.scope.development_call(self.ledger, self.train, row_index=0, candidate='a', phase='train', charge=cost(), operation=lambda: self.fail())

    def test_development_row_order_duplicate_and_validation_before_training(self):
        for phase, source, index in [('train', self.train, 1), ('validation', self.val, 0)]:
            with self.assertRaises(GateError):
                self.scope.development_call(self.ledger, source, row_index=index, candidate='a',
                                            phase=phase, charge=cost(), operation=lambda: self.fail())
        self.develop()
        with self.assertRaises(GateError):
            self.scope.development_call(self.ledger, self.val, row_index=0, candidate='a',
                                        phase='validation', charge=cost(), operation=lambda: self.fail())

    def test_cross_split_artifact_and_source_rejected_before_dispatch(self):
        self.develop()
        artifact = self.freeze()
        other_dir = self.root / 'other'; other_dir.mkdir()
        other, _ = fixture(other_dir, seed=23)
        before = len(self.ledger.snapshot())
        with self.assertRaises(GateError): other.check_source(self.train, stage='train')
        with self.assertRaises(GateError): other.dispatch(self.ledger, artifact, event_id='wrong', operation=lambda: self.fail())
        self.assertEqual(len(self.ledger.snapshot()), before)

    def test_receiver_change_and_artifact_tamper_rejected(self):
        self.develop(); artifact = self.freeze()
        other, _ = fixture(self.root, receiver='R2@config')
        with self.assertRaises(GateError): other.check_artifact(artifact)
        artifact['card'] = 'tampered'
        with self.assertRaises(GateError): self.scope.check_artifact(artifact)

    def test_new_ledger_cannot_skip_paid_search(self):
        self.develop(); artifact = self.freeze()
        empty = Ledger(self.root / 'empty.sqlite', policy())
        self.addCleanup(empty.close)
        with self.assertRaises(GateError): self.scope.dispatch(empty, artifact, event_id='skip', operation=lambda: self.fail())

    def test_test_gold_file_not_opened_and_sender_metadata_rejected(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read')):
            for stage in ('test', 'gold', 'receiver'):
                with self.assertRaises(GateError): self.scope.read_development(self.root, stage=stage)
        bad, _ = fixture(self.root, extra={'candidate_id': 'secret'})
        with self.assertRaises(GateError): bad.read_development(self.root, stage='train')

    def test_file_hash_and_path_escape_rejected(self):
        (self.root / 'sender_train.jsonl').write_text('changed')
        with self.assertRaises(GateError): self.scope.read_development(self.root, stage='train')
        self.scope.files['sender']['train']['file'] = '../outside'
        with self.assertRaises(GateError): self.scope.read_development(self.root, stage='train')

    def test_52_tuple_overlap_and_relabelled_support_blocked(self):
        a, b = build_split(seed=17), build_split(seed=23)
        overlap = set(a['train_meaning_ids']) & set(b['held_out_meaning_ids'])
        self.assertEqual(len(overlap), 52)
        target = next(r for r in b['meanings'] if r['meaning_id'] in overlap)
        meaning = dict(zip(b['attributes'], target['values']))
        # Even a new matching file hash cannot make a held-out tuple into seed-23 training data.
        bad, _ = fixture(self.root, seed=23, train_override=meaning)
        with self.assertRaises(GateError): bad.read_development(self.root, stage='train')

    def test_other_split_spending_reduces_remaining_global_budget(self):
        self.develop()
        for i in range(10):
            self.ledger.run('other' + str(i), phase='train', candidate='other', binding='other-split',
                            charge=cost(9), operation=lambda: Outcome(None, {'bytes': 9}, True))
        # Current split needs 10, other scope already reserved 90: exact equality is feasible.
        artifact = self.freeze()
        self.assertEqual(artifact['feasibility']['b']['reserved']['bytes'], '100')
        self.assertEqual(artifact['feasibility']['b']['prior_reserved']['bytes'], '90')
        self.ledger.run('extra', phase='train', candidate='other', binding='other-split',
                        charge=cost(), operation=lambda: Outcome(None, {'bytes': 1}, True))
        with self.assertRaises(GateError): self.freeze()

    def test_no_feasible_candidates_and_unknown_deployment_fail_closed(self):
        self.develop()
        with self.assertRaises(GateError):
            self.plans['b']['deployment'] = [cost(None)]
            self.freeze()
        self.plans['b']['deployment'] = [cost(101)]
        with self.assertRaises(GateError): self.freeze()


if __name__ == '__main__':
    unittest.main()
