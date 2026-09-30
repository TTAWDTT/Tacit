"""Minimal offline falsification of a new expressivity/wire-saving claim.

Run from repository root: python -m experiments.compositional_protocol.falsify
This exhausts public synthetic meanings, not a sealed model evaluation set.
"""
import hashlib
import json
from pathlib import Path

from .codec import TypedFields
from experiments.emergent_ood_v0_4.compact_fields import encode_fields
from experiments.emergent_ood_v0_4.split import build_split
from experiments.private_match_v0_3.protocols import encode_coordinate_message


def report():
    splits = []
    for seed in (17, 23, 41):
        split = build_split(seed=seed)
        codec = TypedFields(split['values_by_attribute'])
        equal = roundtrips = heldout = new_bytes = old_bytes = 0
        for row in split['meanings']:
            record = dict(zip(split['attributes'], row['values']))
            new = codec.encode(record, scope=split['attributes'])
            old = encode_fields(record)
            equal += new == old
            roundtrips += codec.decode(new, scope=split['attributes']) == record
            heldout += row['split'] == 'held_out'
            new_bytes += len(new.encode('utf-8'))
            old_bytes += len(old.encode('utf-8'))
        if equal != 256 or roundtrips != 256 or new_bytes != old_bytes:
            raise AssertionError('equivalence or preservation falsified')
        splits.append(dict(seed=seed, split_sha256=split['split_sha256'], meanings=256,
                           heldout_meanings=heldout, exact_wire_matches=equal,
                           roundtrips=roundtrips, candidate_payload_bytes=new_bytes,
                           compact_payload_bytes=old_bytes))
    private_equal = 0
    codec = TypedFields({k: [f'{k}{i:04d}' for i in range(4)] for k in ('x', 'y')})
    for k in ('x', 'y'):
        for i in range(4):
            value = f'{k}{i:04d}'
            private_equal += codec.encode({k: value}, scope=[k]) == encode_coordinate_message(
                'compact_kv', q=4, sender=f'sender_{k}', value=value)
    if private_equal != 8:
        raise AssertionError('private compact equivalence falsified')
    root = Path(__file__).resolve().parents[2]
    sources = ['experiments/compositional_protocol/codec.py',
               'experiments/emergent_ood_v0_4/compact_fields.py',
               'experiments/emergent_ood_v0_4/split.py',
               'experiments/private_match_v0_3/protocols.py',
               'examples/compact_labeled_fields_v3.json']
    return dict(kind='offline-mechanism-falsification', model_calls=0,
                conclusion='No additional product expressivity or payload saving over compact fields',
                scope='Public fixture enumeration; no LLM success, tokens, latency or full-cost frontier measured',
                splits=splits, private_match_distinct_messages_equal=private_equal,
                source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources})


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, sort_keys=True))
