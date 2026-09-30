"""Offline properties and falsifiers, including unchanged repository baselines."""
import itertools
import unittest

from experiments.compositional_protocol.codec import TypedFields
from experiments.emergent_ood_v0_4.compact_fields import encode_fields, audit_fields
from experiments.emergent_ood_v0_4.split import build_split
from experiments.private_match_v0_3.generate_tasks import generate_episode_for_target, oracle_answer, score_answer
from experiments.private_match_v0_3.protocols import encode_coordinate_message


class TypedFieldsTests(unittest.TestCase):
    def test_all_meanings_and_heldout_match_compact(self):
        for seed in (17, 23, 41):
            split = build_split(seed=seed)
            schema = TypedFields(split['values_by_attribute'])
            scope = split['attributes']
            train = [r for r in split['meanings'] if r['split'] == 'train']
            held = [r for r in split['meanings'] if r['split'] == 'held_out']
            self.assertEqual((len(train), len(held)), (192, 64))
            self.assertFalse({tuple(r['values']) for r in train} & {tuple(r['values']) for r in held})
            for i, key in enumerate(scope):
                self.assertEqual({r['values'][i] for r in train}, set(split['values_by_attribute'][key]))
            for row in split['meanings']:
                meaning = dict(zip(scope, row['values']))
                message = schema.encode(meaning, scope=scope)
                self.assertEqual(message, encode_fields(meaning))
                self.assertEqual(schema.decode(message, scope=scope), meaning)
                self.assertTrue(audit_fields(message, meaning)['canonical_label_fidelity'])
                parts = [([k], schema.encode({k: v}, scope=[k])) for k, v in meaning.items()]
                self.assertEqual(schema.compose(parts, scope=scope), meaning)

    def test_private_match_exhaustive_role_separation(self):
        # Public fixture key only: never reuse it in model experiments.
        q = 4
        schema = TypedFields({k: [f'{k}{i:04d}' for i in range(q)] for k in ('x', 'y')})
        seen_x = {}
        for x, y in itertools.product(range(q), repeat=2):
            sx, sy, receiver, gold = generate_episode_for_target(
                episode_id='offline', seed=901, q=q, task_key=b'\x13' * 32, x_index=x, y_index=y)
            parts = []
            for sender in (sx, sy):
                k, value = sender['coordinate'], sender['private_value']
                wire = schema.encode({k: value}, scope=[k])
                self.assertEqual(wire, encode_coordinate_message('compact_kv', q=q, sender=sender['role'], value=value))
                parts.append(([k], wire))
            if x in seen_x:
                self.assertEqual(seen_x[x], parts[0][1])  # hidden y cannot affect x message
            seen_x[x] = parts[0][1]
            joined = schema.compose(parts, scope=['x', 'y'])
            self.assertTrue(score_answer(receiver, gold, oracle_answer(receiver, joined['x'], joined['y'])))

    def test_invalid_messages_rejected(self):
        schema = TypedFields({'a': ['red', 'blue'], 'b': ['one', 'two']})
        bad = [None, '', 'a=red', 'a=red;a=red;b=one', 'a=red;a=blue;b=one',
               'b=one;a=red', 'a=red;b=one;', 'a=red;b=one=c', 'a=red;b=one;z=x',
               'a=one;b=red', 'a= red;b=one', 'a=green;b=one', 'a=red\n;b=one']
        for message in bad:
            with self.subTest(message=message), self.assertRaises(ValueError):
                schema.decode(message, scope=['a', 'b'])

    def test_ambiguous_and_unauthorized_composition_rejected(self):
        schema = TypedFields({'a': ['0', '1'], 'b': ['0', '1']})
        for parts in [[], [(['a'], 'a=0')], [(['a'], 'a=0'), (['a'], 'a=0')],
                      [(['a'], 'a=0'), (['a'], 'a=1')], [(['a'], 'b=0')]]:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                schema.compose(parts, scope=['a', 'b'])
        # Erasing BOTH labels and order loses bindings; order alone can retain them.
        first, second = {'a': '0', 'b': '1'}, {'a': '1', 'b': '0'}
        self.assertEqual(sorted(first.values()), sorted(second.values()))
        self.assertNotEqual(schema.encode(first, scope=['a', 'b']), schema.encode(second, scope=['a', 'b']))

    def test_valid_wrong_value_is_not_detectable_as_unfaithful(self):
        schema = TypedFields({'a': ['red', 'blue'], 'b': ['one', 'two']})
        target = {'a': 'red', 'b': 'one'}
        message = 'a=blue;b=one'
        self.assertNotEqual(schema.decode(message, scope=['a', 'b']), target)
        self.assertFalse(audit_fields(message, target)['canonical_label_fidelity'])

    def test_schema_and_input_rejections(self):
        for domains in [{}, {'a': []}, {'a': 'red'}, {'a': ['red', 'red']}, {'a;': ['red']},
                        {'a': ['\ud800']}, {'a': [True]}]:
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                TypedFields(domains)
        schema = TypedFields({'a': ['red'], 'b': ['one']})
        for scope in [[], ['a', 'a'], ['z'], 'a', [False]]:
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                schema.encode({'a': 'red'}, scope=scope)
        for record in [{'a': 'red', 'gold': 'r0'}, {'a': True}, {'a': 'red;gold=r0'}]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                schema.encode(record, scope=['a'])

    def test_unicode_and_schema_snapshot(self):
        domains = {'颜色': ['红色'], 'shape': ['circle']}
        schema = TypedFields(domains)
        domains['颜色'].append('blue')
        meaning = {'颜色': '红色', 'shape': 'circle'}
        self.assertEqual(schema.decode(schema.encode(meaning, scope=list(meaning)), scope=list(meaning)), meaning)
        with self.assertRaises(ValueError):
            schema.encode({'颜色': 'blue'}, scope=['颜色'])

    def test_non_decomposable_missing_relation_counterexample(self):
        schema = TypedFields({'color': ['red'], 'shape': ['circle']})
        # Worlds differ in an unrepresented relation; no decoder of these fields can distinguish them.
        worlds = [(dict(color='red', shape='circle'), answer) for answer in ('left', 'right')]
        messages = [schema.encode(record, scope=['color', 'shape']) for record, _ in worlds]
        self.assertEqual(messages[0], messages[1])
        self.assertNotEqual(worlds[0][1], worlds[1][1])


if __name__ == '__main__':
    unittest.main()
