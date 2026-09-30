"""Deterministic two-connection regressions for PR7 review #5914001217."""
import sqlite3
import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from experiments.m2_gates.budget import Charge, GateError, Ledger, Outcome, Policy, digest
from tests import test_m2_gates as fixtures


class ChargeSnapshotTests(unittest.TestCase):
    def race(self, work, context, mutate, actual, *, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.sqlite'
            policy = Policy(per_request={'bytes': 10}, cumulative={'bytes': 10}, max_calls=2)
            blocker = Ledger(path, policy)
            charge = Charge(work, context)
            ready, go, entering = (threading.Event() for _ in range(3))
            called, results, errors = [], [], []
            def worker():
                ledger = None
                try:
                    ledger = Ledger(path, policy)
                    ledger.db.set_trace_callback(lambda sql: entering.set() if sql == 'BEGIN IMMEDIATE' else None)
                    ready.set()
                    if not go.wait(5): raise TimeoutError('go')
                    def callback():
                        called.append(True)
                        if fail: raise RuntimeError('fake failure')
                        return Outcome('ok', {'bytes': actual}, True)
                    results.append(ledger.run('race', phase='train', candidate='a', binding='s',
                                              charge=charge, operation=callback))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    if ledger: ledger.close()
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                self.assertTrue(ready.wait(5))
                blocker.db.execute('BEGIN IMMEDIATE')
                go.set()
                self.assertTrue(entering.wait(5))
                mutate()
                blocker.db.execute('COMMIT')
                thread.join(5)
                self.assertFalse(thread.is_alive())
                rows = blocker.snapshot()
                for row in rows:
                    self.assertEqual(Fraction(row['reserved']['bytes']),
                                     Fraction(row['components']['work']['bytes']) +
                                     Fraction(row['components']['context']['bytes']))
                return called, results, errors, rows
            finally:
                if blocker.db.in_transaction: blocker.db.execute('ROLLBACK')
                go.set()
                thread.join(5)
                blocker.close()

    def test_original_over_cap_cannot_shrink_while_waiting_for_lock(self):
        work, context = {'bytes': 11}, {'bytes': 0}
        called, results, errors, rows = self.race(work, context, lambda: work.update(bytes=1), 11)
        self.assertEqual(called, [])
        self.assertEqual(results, [])
        self.assertEqual(rows, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], GateError)

    def test_original_valid_charge_components_and_receipt_stay_together(self):
        work, context = {'bytes': 3}, {'bytes': 2}
        def mutate():
            work['bytes'] = 9
            context['bytes'] = 9
        called, results, errors, rows = self.race(work, context, mutate, 5)
        self.assertEqual(errors, [])
        self.assertEqual((called, results), ([True], ['ok']))
        self.assertEqual(rows[0]['reserved'], {'bytes': '5'})
        self.assertEqual(rows[0]['actual'], {'bytes': '5'})
        self.assertEqual(rows[0]['components'], {'work': {'bytes': '3'}, 'context': {'bytes': '2'}})

    def test_failure_retains_original_components_after_external_mutation(self):
        work, context = {'bytes': 3}, {'bytes': 2}
        called, _, errors, rows = self.race(work, context, lambda: context.update(bytes=0), 5, fail=True)
        self.assertEqual(called, [True])
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(rows[0]['status'], 'halted')
        self.assertEqual(rows[0]['reserved'], {'bytes': '5'})
        self.assertEqual(rows[0]['components']['context'], {'bytes': '2'})

    def test_charge_dictionaries_are_immutable_exact_copies(self):
        work = {'bytes': '1.25'}
        charge = Charge(work, {'bytes': 0})
        work['bytes'] = 100
        self.assertEqual(charge.total(('bytes',)), {'bytes': Fraction(5, 4)})
        with self.assertRaises(TypeError): charge.work['bytes'] = 2
        with self.assertRaises(TypeError): charge.context['bytes'] = 2

    def test_lock_timeout_leaves_no_reservation_and_connection_remains_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.sqlite'
            policy = Policy(per_request={'bytes': 10}, cumulative={'bytes': 10}, max_calls=2)
            first, second = Ledger(path, policy), Ledger(path, policy)
            try:
                first.db.execute('BEGIN IMMEDIATE')
                second.db.execute('PRAGMA busy_timeout=10')
                with self.assertRaises(sqlite3.OperationalError):
                    second.run('timeout', phase='train', candidate='a', binding='s',
                               charge=fixtures.cost(), operation=lambda: self.fail('callback'))
                self.assertFalse(second.db.in_transaction)
                self.assertEqual(second.snapshot(), [])
                first.db.execute('COMMIT')
                second.run('after', phase='train', candidate='a', binding='s', charge=fixtures.cost(),
                           operation=lambda: Outcome(None, {'bytes': 1}, True))
                self.assertEqual(second.snapshot()[0]['status'], 'success')
            finally:
                if first.db.in_transaction: first.db.execute('ROLLBACK')
                first.close()
                second.close()


class LedgerSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.IsolationTests('test_feasible_selection_failed_search_deployment_and_reuse_enforced')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.develop()
        self.artifact = self.fixture.freeze()
        self.other = Ledger(self.fixture.root / 'budget.sqlite', self.fixture.ledger.policy)
        self.addCleanup(self.other.close)

    def write_late(self, event='late'):
        f = self.fixture
        self.other.run(event, phase='proposal', candidate='b', binding=f.scope.binding,
                       charge=fixtures.cost(), operation=lambda: Outcome(None, {'bytes': 1}, True))

    def deploy(self):
        f = self.fixture
        f.ledger.run('setup', phase='deployment', candidate='b', binding=f.scope.binding,
                     charge=fixtures.cost(2), operation=lambda: Outcome(None, {'bytes': 2}, True))

    def test_discovery_interleaving_cannot_replace_validated_snapshot(self):
        f = self.fixture
        self.deploy()
        original = f.scope._discovery_history
        def interleaved(snapshot):
            history = original(snapshot)
            self.write_late()
            return history
        with patch.object(f.scope, '_discovery_history', interleaved):
            with self.assertRaises(GateError):
                f.scope.dispatch(f.ledger, self.artifact, event_id='race', operation=lambda: self.fail('callback'))
        self.assertFalse(any(r['id'] == 'race' for r in f.ledger.snapshot()))

    def test_write_after_single_read_rejected_in_reservation_transaction(self):
        f = self.fixture
        self.deploy()
        original = f.ledger.snapshot
        calls = []
        def interleaved():
            rows = original()
            calls.append(True)
            if len(calls) == 1: self.write_late()
            return rows
        with patch.object(f.ledger, 'snapshot', interleaved):
            with self.assertRaises(GateError):
                f.scope.dispatch(f.ledger, self.artifact, event_id='race', operation=lambda: self.fail('callback'))
        self.assertEqual(len(calls), 2)  # one preflight read; one transaction-time CAS
        self.assertFalse(f.ledger.db.in_transaction)
        self.assertFalse(any(r['id'] == 'race' for r in original()))

    def test_freeze_uses_one_snapshot_and_rejects_interleaved_writer(self):
        f = self.fixture
        original = f.scope._discovery_history
        def interleaved(snapshot):
            history = original(snapshot)
            self.write_late()
            return history
        with patch.object(f.scope, '_discovery_history', interleaved):
            with self.assertRaises(GateError): f.freeze()

    def test_verified_artifact_is_detached_from_caller_schedule(self):
        f = self.fixture
        self.deploy()
        original = f.scope.check_artifact
        def mutate_after_verification(artifact):
            body = original(artifact)
            artifact['schedule'][0]['work']['bytes'] = '100'
            return body
        with patch.object(f.scope, 'check_artifact', mutate_after_verification):
            f.scope.dispatch(f.ledger, self.artifact, event_id='stable',
                             operation=lambda: Outcome(None, {'bytes': 2}, True))
        self.assertEqual(f.ledger.snapshot()[-1]['reserved'], {'bytes': '2'})
        with self.assertRaises(GateError): f.scope.check_artifact(self.artifact)

    def test_freeze_cost_plan_is_detached_from_caller_after_feasibility(self):
        f = self.fixture
        from experiments.m2_gates import isolation
        original = isolation.frontier
        def mutate_after_check(*args, **kwargs):
            result = original(*args, **kwargs)
            f.plans['b']['per_use'] = [fixtures.cost(100)]
            return result
        with patch.object(isolation, 'frontier', mutate_after_check):
            artifact = f.freeze()
        self.assertEqual(artifact['schedule'][0], {'work': {'bytes': '1'}, 'context': {'bytes': '1'}})
        self.assertEqual(artifact['feasibility']['b']['reserved']['bytes'], '10')


if __name__ == '__main__':
    unittest.main()
