"""Development-only connection to the frozen v0.4 runner, with durable results.

Receives already validated role episodes/provenance; does not read sealed test files.
Accounting receipt status and strict task outcome deliberately remain separate.
"""
import json

from experiments.emergent_ood_v0_4.runner import run_condition
from experiments.m2_gates.budget import GateError, digest
from experiments.emergent_ood_v0_4.split import validate_split
from experiments.runner_cost_adapter.adapter import GuardedChatModel, receipts


def episode_prefix(scope, *, stage, condition, row_index, protocol_card=None, wire_budget_bytes=4096):
    intervention = digest(dict(condition=condition, protocol_card=protocol_card,
                               wire_budget_bytes=wire_budget_bytes))
    return scope.binding + '/' + stage + '/' + condition + '/' + intervention + '/' + str(row_index)


def run_development_episode(*, ledger, scope, source, row_index, episode, condition,
                            split, task_seed, sender, receiver, protocol_card=None,
                            wire_budget_bytes=4096):
    stage = source.get('stage')
    if stage not in ('train', 'validation'):
        raise GateError('sealed test unavailable to this bridge')
    scope.check_source(source, stage=stage)
    if type(row_index) is not int or not 0 <= row_index < len(source['meaning_ids']):
        raise GateError('development row index invalid')
    target = episode['sender']['private_meaning']
    mid = scope.meanings.get(tuple(target[a] for a in scope.attributes))
    if mid != source['meaning_ids'][row_index]:
        raise GateError('episode differs from split-bound development row')
    # Scope stores meanings, not the original split object; check full mapping/order.
    validate_split(split)
    if set(split['train_meaning_ids']) != scope.support['train']:
        raise GateError('split support mismatch')
    expected = {tuple(r['values']): r['meaning_id'] for r in split['meanings']}
    if tuple(split['attributes']) != scope.attributes or expected != scope.meanings:
        raise GateError('public ontology mismatch')
    clients = [receiver] + ([sender] if sender is not None else [])
    prefix = episode_prefix(scope,stage=stage,condition=condition,row_index=row_index,
                            protocol_card=protocol_card,wire_budget_bytes=wire_budget_bytes)
    for c in clients:
        if not isinstance(c, GuardedChatModel) or c.ledger is not ledger or c.binding != scope.binding or c.phase != stage or c.prefix != prefix:
            raise GateError('model client ledger/config/stage/episode misbinding')
    if receiver.contract.identity != scope.receiver or receiver.role != 'receiver' or (sender and sender.role != 'sender'):
        raise GateError('receiver contract or role mismatch')
    if any(c.index != 0 for c in clients):
        raise GateError('one fresh client per role and episode; no reuse/retry')
    ledger.db.execute('CREATE TABLE IF NOT EXISTS episode_results ('
                      'id TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT, error TEXT)')
    # A duplicate (including a failed partial episode) cannot silently rerun.
    ledger.db.execute('INSERT INTO episode_results VALUES (?,?,?,?)', (prefix, 'started', None, None))
    try:
        result = run_condition(episode=episode, condition=condition, stage=stage,
            sender_model=sender, receiver_model=receiver, protocol_card=protocol_card,
            sender_tokenizer_id=sender.contract.tokenizer_sha256 if sender else None,
            receiver_tokenizer_id=receiver.contract.tokenizer_sha256,
            attributes=split['attributes'], values=split['values_by_attribute'],
            split_seed=split['seed'], task_seed=task_seed, model_population_id=scope.receiver,
            wire_budget_bytes=wire_budget_bytes)
        result['adapter'] = dict(binding=scope.binding, source_sha256=source['sha256'],
            contract_sha256=receiver.contract.identity, evidence='development plumbing; not model superiority',
            receipt_event_ids=[r['event_id'] for r in receipts(ledger) if r['event_id'].startswith(prefix + '/')],
            setup_cost_status='discovery/deployment must be separately charged; not inferred zero',
            ledger_status_semantics='receipt validity; use outcome for strict task success')
        ledger.db.execute('UPDATE episode_results SET status=?,result=? WHERE id=?',
                          ('completed', json.dumps(result, allow_nan=False), prefix))
        return result
    except BaseException as exc:
        ledger.db.execute('UPDATE episode_results SET status=?,error=? WHERE id=?', ('error', type(exc).__name__, prefix))
        raise


def export_evidence(ledger):
    """Evaluator-private all-attempt evidence, including partial/error episodes."""
    table = ledger.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episode_results'").fetchone()
    rows = ledger.db.execute('SELECT id,status,result,error FROM episode_results ORDER BY rowid').fetchall() if table else []
    return dict(schema='tacit.runner-cost-adapter.evidence.v1', ledger=ledger.snapshot(),
        model_receipts=receipts(ledger),
        episodes=[dict(id=r[0],status=r[1],result=json.loads(r[2]) if r[2] else None,error=r[3]) for r in rows],
        limitations='receipt status is not task success; unknown setup/compute/reasoning costs remain unknown')
