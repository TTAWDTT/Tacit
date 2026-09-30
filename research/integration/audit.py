"""Model-free fixed-source audit. Run with --source-root pointing to the five-pack tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = '5dd260c5824b280b5323a1707ab132524afa8e4c'
HEADS = {
    '1': '6de34bfd821aa8656d9063378087afc362aa9252',
    '2': '05d2921ead5ac8f555b7f776974f907de5bbc5a4',
    '3': 'ab06295aa5e0331c414f002b077dd71293c9c6f9',
    '4_original': 'c6e0e3e80b43523b78aa273e0a6c6f9005827042',
    '4_revised': '04129b8dcefedbaf1738f0851f9cd940ba863eff',
    '5_original': 'b438b8d07aae5e148b223141203599ccd6eef54c',
    '5_revised': 'e0dc30806886caacdcd44dcbc33a151c17c8af92',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    root = parser.parse_args().source_root.resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('diff', '--name-only', 'HEAD'):
        raise AssertionError('source tree has tracked edits')
    # Additions are permitted; every baseline blob, including frozen sources, must survive.
    revisions = dict(HEADS, integration=git('rev-parse', 'HEAD'))
    for revision in revisions.values():
        assert not git('diff', '--name-only', '--diff-filter=DMRTUXB', BASE, revision)
    for key in ('1', '2', '3', '4_revised', '5_revised'):
        subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', HEADS[key], 'HEAD'], check=True)
    sys.path.insert(0, str(root))
    from experiments.emergent_ood_v0_4.split import build_split
    from experiments.compositional_protocol.codec import TypedFields
    from experiments.receiver_adaptation.reference import (
        Candidate, Observation, select, shuffled_mapping, decoder_corruption_control,
    )
    # Public fixture only. Different split seeds are not globally disjoint meaning supports.
    first, second = build_split(seed=17), build_split(seed=23)
    train = {tuple(r['values']) for r in first['meanings'] if r['split'] == 'train'}
    held = {tuple(r['values']) for r in second['meanings'] if r['split'] == 'held_out'}
    overlap = len(train & held)
    assert overlap > 0
    candidate = Candidate('fixture', 'code', 'fixture', (('red', 'a'), ('blue', 'bbbb')))
    meanings = ('red', 'red', 'red', 'blue')
    lengths = [sum(len(dict(mapping)[m].encode()) for m in meanings)
               for mapping in (candidate.mapping, shuffled_mapping(candidate))]
    assert lengths == [7, 13]
    control = decoder_corruption_control(candidate, meanings)
    assert control['correct_messages'] == control['corrupted_messages']
    cheap = Candidate('cheap', 'code', 'fixture', candidate.mapping)
    costly = Candidate('costly', 'code', 'fixture', candidate.mapping)
    rows = [Observation(c.name, 'R', stage, eid, c.name == 'costly',
                        100 if c.name == 'costly' else 1)
            for c in (cheap, costly) for stage, eid in (('train', 't'), ('validation', 'v'))]
    freeze = select([cheap, costly], rows, receiver='R', train_ids=['t'], validation_ids=['v'],
                    unit='invented', proposal_cost=0, deployment_cost=0, max_candidates=2)
    assert freeze['candidate']['name'] == 'costly'
    assert freeze['budget_feasibility'] == 'not_evaluated' and not freeze['deployment_authorized']
    # A typed validator accepts a legal false value; caller-supplied scope isn't authentication.
    codec = TypedFields({'x': ['0', '1'], 'y': ['0', '1']})
    assert codec.decode('x=1', scope=['x']) != {'x': '0'}
    assert codec.compose([(['x'], 'x=0'), (['y'], 'y=1')], scope=['x', 'y']) == {'x': '0', 'y': '1'}
    rejected_controls = 0
    for cp in (*range(0x20), *range(0x7f, 0xa0)):
        try:
            TypedFields({'x': ['a' + chr(cp) + 'b']})
        except ValueError:
            rejected_controls += 1
    assert rejected_controls == 65
    sources = ['experiments/compositional_protocol/codec.py',
               'experiments/receiver_adaptation/reference.py',
               'experiments/emergent_ood_v0_4/split.py']
    result = dict(evidence='public offline fixtures; no LLM evidence', model_calls=0,
                  base=BASE, reviewed_heads=HEADS, all_baseline_blobs_unchanged=True,
                  seed17_train_overlap_seed23_heldout=overlap,
                  sender_shuffle_bytes=lengths, revised_control_payload_identical=True, rejected_unicode_cc_controls=rejected_controls,
                  unconstrained_winner=freeze['candidate']['name'], budget_feasibility='not_evaluated',
                  legal_false_values_and_untrusted_scope_not_detected=True,
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
