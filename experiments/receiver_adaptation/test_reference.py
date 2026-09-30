import unittest
from experiments.receiver_adaptation.reference import (
    Candidate, Observation, Cost, break_even, select, verify_freeze, shuffled_mapping, decoder_corruption_control,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cards = [Candidate('plain', 'nl', 'red or blue', (('red', 'red'), ('blue', 'blue'))),
                      Candidate('code', 'codebook', 'red=a blue=b', (('red', 'a'), ('blue', 'b')))]
        self.rows = [Observation(c.name, 'R1', stage, eid, c.name == 'code', 2)
                     for c in self.cards for stage, eid in [('train', 't'), ('validation', 'v')]]

    def freeze(self, rows=None):
        return select(self.cards, self.rows if rows is None else rows, receiver='R1',
                      train_ids=['t'], validation_ids=['v'], unit='synthetic_units',
                      proposal_cost=3, deployment_cost=7, max_candidates=2)

    def test_all_failed_candidates_charged(self):
        f = self.freeze()
        self.assertEqual(f['discovery_cost'], '11')
        self.assertEqual(f['deployment_cost'], '7')
        self.assertEqual(f['candidate']['name'], 'code')

    def test_reproducible_and_order_invariant(self):
        self.assertEqual(self.freeze(), self.freeze(list(reversed(self.rows))))

    def test_generator_feedback(self):
        self.assertEqual(self.freeze(), self.freeze(iter(self.rows)))

    def test_unequal_meaning_coverage(self):
        self.cards[1] = Candidate('code', 'codebook', 'wrong domain', (('green', 'a'), ('blue', 'b')))
        with self.assertRaises(ValueError): self.freeze()

    def test_unbalanced_encoder_shuffle_changes_payload_cost(self):
        c = Candidate('unequal', 'codebook', 'fixture', (('red', 'a'), ('blue', 'bbbb')))
        meanings = ('red', 'red', 'red', 'blue')
        original, shuffled = dict(c.mapping), dict(shuffled_mapping(c))
        self.assertEqual(sum(len(original[m].encode()) for m in meanings), 7)
        self.assertEqual(sum(len(shuffled[m].encode()) for m in meanings), 13)

    def test_decoder_control_preserves_actual_messages(self):
        c = Candidate('unequal', 'codebook', 'fixture', (('red', 'a'), ('blue', 'bbbb')))
        meanings = ('red', 'red', 'red', 'blue')
        control = decoder_corruption_control(c, meanings)
        self.assertEqual(control['correct_messages'], control['corrupted_messages'])
        self.assertEqual(sum(len(m.encode()) for m in control['corrupted_messages']), 7)
        self.assertEqual(tuple(control['correct_decoder'][m] for m in control['correct_messages']), meanings)
        self.assertTrue(all(control['corrupted_decoder'][m] != truth
                            for m, truth in zip(control['corrupted_messages'], meanings)))

    def test_ranking_does_not_claim_budget_feasibility(self):
        rows = [Observation(r.candidate, r.receiver, r.stage, r.episode,
                            r.candidate == 'code', 100 if r.candidate == 'code' else 1)
                for r in self.rows]
        f = self.freeze(rows)
        self.assertEqual(f['candidate']['name'], 'code')
        self.assertEqual(f['discovery_cost'], '205')
        self.assertEqual(f['selection_scope'], 'unconstrained_inventory_ranking')
        self.assertEqual(f['budget_feasibility'], 'not_evaluated')
        self.assertFalse(f['deployment_authorized'])
        verify_freeze(f)

    def test_freeze_tamper(self):
        f = self.freeze()
        verify_freeze(f)
        f['candidate']['card'] = 'changed'
        with self.assertRaises(ValueError): verify_freeze(f)

    def test_test_feedback_rejected(self):
        with self.assertRaises(ValueError):
            self.freeze(self.rows + [Observation('code', 'R1', 'test', 'x', True, 2)])

    def test_missing_failed_candidate_rejected(self):
        with self.assertRaises(ValueError): self.freeze(self.rows[1:])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): self.freeze(self.rows + self.rows[:1])

    def test_other_receiver_rejected(self):
        with self.assertRaises(ValueError):
            self.freeze([Observation(r.candidate, 'R2', r.stage, r.episode, r.strict_success, r.cost) for r in self.rows])

    def test_correct_and_shuffled_mapping_control(self):
        # Mechanical property, not a synthetic claim that adaptation works.
        c = self.cards[1]
        decoder = {label: meaning for meaning, label in c.mapping}
        self.assertTrue(all(decoder[label] == meaning for meaning, label in c.mapping))
        shuffled = shuffled_mapping(c)
        self.assertTrue(all(decoder[label] != meaning for meaning, label in shuffled))
        self.assertEqual(sorted(v for _, v in shuffled), sorted(v for _, v in c.mapping))

    def test_receiver_change_failure_counterexample(self):
        c = self.cards[1]
        new_decoder = {label: meaning for meaning, label in shuffled_mapping(c)}
        self.assertEqual(sum(new_decoder[label] == meaning for meaning, label in c.mapping), 0)
        # No inference about real receivers follows from this constructed decoder.

    def test_no_adaptation_gain_counterexample(self):
        rows = [Observation(r.candidate, r.receiver, r.stage, r.episode, True, r.cost) for r in self.rows]
        f = self.freeze(rows)
        self.assertEqual(f['candidate']['name'], 'code')  # arbitrary deterministic tie
        self.assertEqual(break_even(Cost('u', 11, 7, 2), Cost('u', 0, 0, 2))['status'], 'never')

    def test_break_even_30_and_31(self):
        a, b = Cost('u', 100, 20, 6), Cost('u', 0, 0, 10)
        self.assertEqual(break_even(a, b)['first'], 30)
        self.assertEqual(a.average(30), b.average(30))
        self.assertLess(a.average(31), b.average(31))
        self.assertEqual((a.average(100) - 6) * 100, 120)

    def test_repeated_context_eliminates_saving(self):
        self.assertEqual(break_even(Cost('u', 100, 20, 11), Cost('u', 0, 0, 10))['status'], 'never')

    def test_temporary_advantage(self):
        self.assertEqual(break_even(Cost('u', 0, 0, 12), Cost('u', 20, 0, 10)),
                         {'status': 'temporary', 'first': 1, 'last': 10})

    def test_unknown_and_axis_mismatch(self):
        self.assertEqual(break_even(Cost('u', None, 0, 1), Cost('u', 0, 0, 2))['status'], 'unknown')
        with self.assertRaises(ValueError): break_even(Cost('R1tokens', 0, 0, 1), Cost('R2tokens', 0, 0, 1))

    def test_invalid_cost_and_reuse(self):
        for v in (-1, True, 'nan', 'inf'):
            with self.assertRaises((ValueError, OverflowError)): Cost('u', v, 0, 1)
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): Cost('u', 0, 0, 1).average(n)

    def test_exhaustive_integer_interval_against_direct_cost(self):
        for fa in range(5):
            for fb in range(5):
                for ca in range(4):
                    for cb in range(4):
                        a, b = Cost('u', fa, 0, ca), Cost('u', fb, 0, cb)
                        interval = break_even(a, b)
                        for n in range(1, 21):
                            inside = interval['first'] is not None and n >= interval['first'] and (interval['last'] is None or n <= interval['last'])
                            self.assertEqual(inside, a.average(n) <= b.average(n))


if __name__ == '__main__':
    unittest.main()
