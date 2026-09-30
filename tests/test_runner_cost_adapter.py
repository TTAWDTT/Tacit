"""No models: actual loopback HTTP fixture + frozen runner + SQLite receipts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import sqlite3
import base64
import http.client
import io
from unittest.mock import patch
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest

from experiments.runner_cost_adapter.adapter import Contract, GuardedChatModel, LoopbackHTTP, raw_json, receipts, NativeCounters, PartialHTTPRead
from experiments.runner_cost_adapter.runner_bridge import run_development_episode, episode_prefix, export_evidence
from experiments.m2_gates.budget import GateError, Ledger, Policy
from experiments.m2_gates.isolation import Scope
from experiments.emergent_ood_v0_4.split import build_split
from experiments.emergent_ood_v0_4.episodes import generate_ledgers
from experiments.emergent_ood_v0_4.runner import select_candidate_sets


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = Contract('http://127.0.0.1:9876/v1/chat/completions', 'fixture-model',
            'a'*64, 'b'*64, 48, 8192, 2, True, True, True, True)
        self.requests = []
        self.response_text = 'invalid answer'
        self.modify = lambda p: p
        self.transport_calls = 0
        self.split = build_split(seed=17)
        self.bundle = generate_ledgers(split=self.split, task_key=bytes(range(32)), task_seed=9, k=4, sets_per_stage=1)
        # File writer restricts paths to the project, so construct the development
        # sender role files from this public in-memory fixture, without test access.
        files = {'sender': {}}
        for stage in ('train', 'validation'):
            raw = b''.join(raw_json(r)+b'\n' for r in self.bundle['sender'][stage])
            name='sender_'+stage+'.jsonl'; (self.root/name).write_bytes(raw)
            files['sender'][stage]=dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),records=len(self.bundle['sender'][stage]))
        self.manifest=dict(split_sha256=self.split['split_sha256'],files=files)
        self.train_ids = set(self.split['train_meaning_ids'])
        self.val_ids = [r['meaning_id'] for r in self.bundle['gold']['validation']]
        self.scope = Scope(self.split,validation_ids=self.val_ids,
            test_ids=set(self.split['held_out_meaning_ids'])-set(self.val_ids),manifest=self.manifest,receiver=self.contract.identity)
        _,self.source=self.scope.read_development(self.root,stage='validation')
        self.episode=select_candidate_sets(self.bundle,'validation',1)[0]
        self.prefix=self.scope.binding+'/validation/shared_protocol_card/0'
        # Explicitly fake counters: full request encoding length, NOT a real tokenizer.
        self.counter=lambda request:len(raw_json(request))
        self.text_counter=lambda text:(len(text.encode())+3)//4
        self.ledger=self.new_ledger()
        self.addCleanup(self.ledger.close)

    def new_ledger(self, max_calls=12, input_cap=100000):
        axes={}
        for role in ('sender','receiver'):
            p=role+':'+self.contract.identity+':'
            axes[p+'input_tokens']=input_cap
            axes[p+'output_tokens']=48
        axes.update(request_body_utf8_bytes=100000,response_body_utf8_bytes=8192,model_requests=1)
        cumulative={a:v*12 for a,v in axes.items()}
        return Ledger(self.root/('ledger-'+str(max_calls)+'-'+str(input_cap)+'.sqlite'),Policy(per_request=axes,cumulative=cumulative,max_calls=max_calls))

    def transport(self, body, contract):
        self.transport_calls += 1
        request=json.loads(body);self.requests.append(request)
        payload=dict(id='fake',model='fixture-model',choices=[dict(message=dict(content=self.response_text),finish_reason='stop')],
                     usage=dict(prompt_tokens=len(body),completion_tokens=(len(self.response_text.encode())+3)//4,completion_tokens_details=dict(reasoning_tokens=0)))
        return 200,raw_json(self.modify(payload))

    def client(self, role='receiver', transport=None, prefix=None):
        return GuardedChatModel(ledger=self.ledger,contract=self.contract,role=role,
            event_prefix=prefix or self.prefix,binding=self.scope.binding,phase='validation',
            count_prompt=self.counter,count_text=self.text_counter,transport=transport or self.transport)

    def run_episode(self, text, card=None, wire_budget=4096):
        target=self.episode['sender']['private_meaning']
        self.response_text=json.dumps(target,separators=(',',':'))
        card=card or dict(protocol_id='fixture-json',sender_instruction='Encode every field as JSON.',receiver_instruction='Decode every field from JSON.')
        prefix=episode_prefix(self.scope,stage='validation',condition='shared_protocol_card',row_index=0,protocol_card=card,wire_budget_bytes=wire_budget)
        sender=self.client('sender',prefix=prefix)
        def receiver_transport(body,contract):
            self.response_text=text
            return self.transport(body,contract)
        receiver=self.client('receiver',receiver_transport,prefix=prefix)
        return run_development_episode(ledger=self.ledger,scope=self.scope,source=self.source,row_index=0,
            episode=self.episode,condition='shared_protocol_card',split=self.split,task_seed=9,sender=sender,receiver=receiver,
            protocol_card=card,wire_budget_bytes=wire_budget)

    def test_unsupported_budget_axes_rejected_before_post(self):
        for name in ('provider_generation_seconds','local_compute_cost','unknown_tokens'):
            with self.subTest(axis=name):
                axes=dict(self.ledger.policy.per_request)
                axes[name]=0
                ledger=Ledger(self.root/(name+'.sqlite'),Policy(per_request=axes,cumulative=axes,max_calls=1))
                try:
                    c=GuardedChatModel(ledger=ledger,contract=self.contract,role='receiver',
                        event_prefix=name,binding=self.scope.binding,phase='validation',
                        count_prompt=self.counter,count_text=self.text_counter,transport=self.transport)
                    self.modify=lambda p:dict(p,usage=dict(p['usage'],generation_seconds=5))
                    with self.assertRaisesRegex(GateError,'unsupported'):
                        c.complete([dict(role='user',content='x')])
                    self.assertEqual(ledger.snapshot(),[])
                finally:ledger.close()
        self.assertEqual(self.transport_calls,0)

    def test_incomplete_chunked_response_retains_partial_status_and_truncation(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(inner):
                inner.rfile.read(int(inner.headers['Content-Length']))
                owner.transport_calls += 1
                inner.send_response(200)
                inner.send_header('Transfer-Encoding','chunked')
                inner.end_headers()
                inner.wfile.write(b'3\r\nabc\r\n')
                inner.wfile.flush()
                inner.close_connection=True  # no terminal zero chunk
        server=HTTPServer(('127.0.0.1',0),Handler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            contract=replace(self.contract,endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
            axes={a:100000 for a in ('request_body_utf8_bytes','response_body_utf8_bytes','model_requests')}
            for a in ('input_tokens','output_tokens'):axes['receiver:'+contract.identity+':'+a]=100000
            ledger=Ledger(self.root/'partial.sqlite',Policy(per_request=axes,cumulative=axes,max_calls=2))
            try:
                c=GuardedChatModel(ledger=ledger,contract=contract,role='receiver',event_prefix='partial',binding='fixture',phase='train',count_prompt=self.counter,count_text=self.text_counter)
                with self.assertRaises(PartialHTTPRead):c.complete([dict(role='user',content='x')])
                row=receipts(ledger)[0]
                self.assertEqual(base64.b64decode(row['response_base64']),b'abc')
                self.assertEqual(row['http_status'],200)
                self.assertTrue(row['response_truncated'])
                self.assertEqual(ledger.snapshot()[0]['status'],'halted')
                self.assertIsNone(ledger.snapshot()[0]['actual'])
                with self.assertRaises(GateError):c.complete([dict(role='user',content='x')])
                self.assertEqual(owner.transport_calls,1)
            finally:ledger.close()
        finally:server.shutdown();server.server_close();t.join()

    def check_wire_response(self, make_wire, *, expected_partial=None, response_limit=8192):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(inner):
                body=inner.rfile.read(int(inner.headers['Content-Length']))
                _,valid=owner.transport(body,owner.contract)
                headers,wire=make_wire(valid)
                inner.send_response(200)
                for k,v in headers.items():inner.send_header(k,str(v))
                inner.end_headers();inner.wfile.write(wire);inner.wfile.flush()
                inner.close_connection=True
        server=HTTPServer(('127.0.0.1',0),Handler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            contract=replace(self.contract,endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',max_response_bytes=response_limit)
            axes={a:100000 for a in ('request_body_utf8_bytes','response_body_utf8_bytes','model_requests')}
            for a in ('input_tokens','output_tokens'):axes['receiver:'+contract.identity+':'+a]=100000
            ledger=Ledger(self.root/('wire-'+str(server.server_port)+'.sqlite'),Policy(per_request=axes,cumulative=axes,max_calls=2))
            try:
                c=GuardedChatModel(ledger=ledger,contract=contract,role='receiver',event_prefix='wire',binding='fixture',phase='train',count_prompt=self.counter,count_text=self.text_counter)
                if expected_partial is None:
                    self.assertEqual(c.complete([dict(role='user',content='x')]).text,self.response_text)
                    self.assertEqual(ledger.snapshot()[0]['status'],'success')
                    self.assertFalse(receipts(ledger)[0]['response_truncated'])
                else:
                    with self.assertRaises(GateError):c.complete([dict(role='user',content='x')])
                    evidence=receipts(ledger)[0]
                    raw=base64.b64decode(evidence['response_base64'])
                    if callable(expected_partial):self.assertTrue(expected_partial(raw))
                    else:self.assertEqual(raw,expected_partial)
                    self.assertTrue(evidence['response_truncated'])
                    self.assertEqual(evidence['http_status'],200)
                    self.assertEqual(ledger.snapshot()[0]['status'],'halted')
                    self.assertIsNone(ledger.snapshot()[0]['actual'])
                    before=owner.transport_calls
                    with self.assertRaises(GateError):c.complete([dict(role='user',content='x')])
                    self.assertEqual(owner.transport_calls,before)
            finally:ledger.close()
        finally:server.shutdown();server.server_close();t.join()

    def test_short_first_chunk_keeps_received_bytes(self):
        self.check_wire_response(lambda valid:({'Transfer-Encoding':'chunked'},b'5\r\nabc'),expected_partial=b'abc')

    def test_short_later_chunk_keeps_all_returned_body_bytes(self):
        self.check_wire_response(lambda valid:({'Transfer-Encoding':'chunked'},b'3\r\nabc\r\n5\r\ndef'),expected_partial=b'abcdef')

    def test_short_content_length_valid_json_cannot_succeed(self):
        self.check_wire_response(lambda valid:({'Content-Length':len(valid)+10},valid),expected_partial=lambda raw:json.loads(raw)['model']=='fixture-model')

    def test_complete_length_and_complete_chunked_controls(self):
        self.check_wire_response(lambda valid:({'Content-Length':len(valid)},valid))
        self.check_wire_response(lambda valid:({'Transfer-Encoding':'chunked'},f'{len(valid):x}\r\n'.encode()+valid+b'\r\n0\r\n\r\n'))

    def test_response_size_limit_uses_bounded_evidence(self):
        self.check_wire_response(lambda valid:({'Content-Length':300},b'x'*300),expected_partial=lambda raw:len(raw)==33,response_limit=32)

    def test_missing_final_trailer_terminator_cannot_succeed(self):
        self.check_wire_response(lambda valid:({'Transfer-Encoding':'chunked'},f'{len(valid):x}\r\n'.encode()+valid+b'\r\n0\r\n'),expected_partial=lambda raw:json.loads(raw)['model']=='fixture-model')

    def test_explicit_peer_contract_axes_only(self):
        peer=replace(self.contract,endpoint='http://127.0.0.1:9877/v1/chat/completions')
        axes={a:100000 for a in ('request_body_utf8_bytes','response_body_utf8_bytes','model_requests')}
        for role,contract in (('sender',peer),('receiver',self.contract)):
            for a in ('input_tokens','output_tokens'):axes[role+':'+contract.identity+':'+a]=100000
        ledger=Ledger(self.root/'peer.sqlite',Policy(per_request=axes,cumulative=axes,max_calls=1))
        try:
            kwargs=dict(ledger=ledger,contract=self.contract,role='receiver',event_prefix='peer',binding='fixture',phase='train',count_prompt=self.counter,count_text=self.text_counter,transport=self.transport)
            with self.assertRaisesRegex(GateError,'unsupported'):
                GuardedChatModel(**kwargs).complete([dict(role='user',content='x')])
            self.assertEqual(self.transport_calls,0)
            GuardedChatModel(**kwargs,peer_contract=peer).complete([dict(role='user',content='x')])
            actual=ledger.snapshot()[0]['actual']
            self.assertEqual(actual['sender:'+peer.identity+':input_tokens'],'0')
            self.assertEqual(actual['sender:'+peer.identity+':output_tokens'],'0')
            self.assertEqual(self.transport_calls,1)
        finally:ledger.close()

    def test_partial_evidence_is_bounded_and_has_no_headers(self):
        class Response:
            status=200
            fp=io.BufferedReader(io.BytesIO(b'x'*10000))
            def getheader(inner,name):return '10010' if name=='Content-Length' else None
            def close(inner):inner.fp.close()
        class Connection:
            def request(inner,*args,**kwargs):pass
            def getresponse(inner):return Response()
            def close(inner):pass
        with patch('experiments.runner_cost_adapter.adapter.http.client.HTTPConnection',return_value=Connection()):
            with self.assertRaises(GateError):
                self.client(transport=LoopbackHTTP()).complete([dict(role='user',content='x')])
        r=receipts(self.ledger)[0]
        self.assertEqual(len(base64.b64decode(r['response_base64'])),self.contract.max_response_bytes+1)
        self.assertEqual(r['http_status'],200)
        self.assertTrue(r['response_truncated'])
        self.assertIsNone(self.ledger.snapshot()[0]['actual'])
        self.assertNotIn('headers',r)

    def test_legacy_receipt_columns_migrate_without_reset(self):
        self.ledger.db.execute('CREATE TABLE model_receipts (event_id TEXT PRIMARY KEY, request BLOB NOT NULL, request_sha256 TEXT NOT NULL, contract TEXT NOT NULL, response BLOB, receipt TEXT, error TEXT)')
        self.ledger.db.execute('INSERT INTO model_receipts VALUES (?,?,?,?,?,?,?)',
            ('old',b'{}','old-hash','{}',b'old',None,'old-error'))
        c=self.client()
        old=receipts(self.ledger)[0]
        self.assertEqual(old['response_base64'],base64.b64encode(b'old').decode())
        self.assertEqual(old['error'],'old-error')
        self.assertIsNone(old['http_status'])
        self.assertIsNone(old['response_truncated'])
        c.complete([dict(role='user',content='x')])
        self.assertEqual(len(receipts(self.ledger)),2)
        self.assertEqual(receipts(self.ledger)[1]['http_status'],200)
        self.assertFalse(receipts(self.ledger)[1]['response_truncated'])

    def test_native_counter_uses_verified_template_options(self):
        class Tokenizer:
            chat_template='local fixture template'
            def apply_chat_template(inner,messages,**kwargs):
                inner.options=kwargs
                return [1,2,3]
            def encode(inner,text,**kwargs):return [1,2]
        tokenizer=Tokenizer()
        contract=replace(self.contract,template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
        native=NativeCounters(tokenizer,contract)
        self.assertEqual(native.prompt(dict(messages=[dict(role='user',content='x')],chat_template_kwargs=dict(enable_thinking=False))),3)
        self.assertEqual(tokenizer.options,dict(tokenize=True,add_generation_prompt=True,enable_thinking=False))
        self.assertEqual(native.text('x'),2)
        with self.assertRaises(GateError):NativeCounters(tokenizer,self.contract)

    def test_unconfirmed_or_remote_contract_denied(self):
        for changes in ({'free_local_confirmed':False},{'stateless_verified':False},
                        {'reasoning_disabled_verified':False},{'completion_cap_verified':False},
                        {'endpoint':'https://example.com/v1/chat/completions'}, {'tokenizer_sha256':'unknown'}):
            with self.assertRaises(GateError): replace(self.contract,**changes)
        self.assertEqual(self.transport_calls,0)

    def test_full_request_context_reserved_before_transport(self):
        c=self.client()
        def transport(body,contract):
            rows=self.ledger.snapshot()
            self.assertEqual(rows[0]['status'],'inflight')
            self.assertEqual(int(rows[0]['components']['context'][c.token_axes[0]]),len(body))
            self.assertEqual(receipts(self.ledger)[0]['request'],json.loads(body))
            return self.transport(body,contract)
        c.transport=transport
        c.complete([dict(role='system',content='card '+ '例'*200),dict(role='user',content='message')])
        self.assertEqual(len(self.ledger.snapshot()),1)
        self.assertEqual(self.requests[0]['max_tokens'],48)
        self.assertFalse(self.requests[0]['chat_template_kwargs']['enable_thinking'])

    def test_prompt_and_repeated_onboarding_can_exceed_cap_before_post(self):
        self.ledger.close();self.ledger=self.new_ledger(input_cap=10)
        with self.assertRaises(GateError): self.client().complete([dict(role='user',content='example '*100)])
        self.assertEqual(self.transport_calls,0)
        self.assertEqual(self.ledger.snapshot(),[])
        self.assertIsNotNone(receipts(self.ledger)[0]['request'])

    def test_invalid_format_and_valid_wrong_id_preserved_as_failures(self):
        row=self.run_episode('Answer: '+self.episode['gold']['candidate_id'])
        self.assertFalse(row['outcome']['answer_format_valid'])
        self.assertFalse(row['outcome']['exact_selection'])
        self.assertEqual(row['trace']['answer'],'Answer: '+self.episode['gold']['candidate_id'])
        # Ledger success is explicitly receipt validity, not task success.
        self.assertEqual([r['status'] for r in self.ledger.snapshot()],['success','success'])
        stored=json.loads(self.ledger.db.execute('SELECT result FROM episode_results').fetchone()[0])
        self.assertFalse(stored['outcome']['exact_selection'])
        self.assertEqual(len(row['adapter']['receipt_event_ids']),2)

    def test_valid_id_unchanged_scorer_and_full_card_context(self):
        row=self.run_episode(self.episode['gold']['candidate_id'])
        self.assertTrue(row['outcome']['exact_selection'])
        self.assertIn('Shared protocol card',self.requests[-1]['messages'][0]['content'])
        self.assertEqual(row['costs']['complete_input_tokens'],sum(len(raw_json(r)) for r in self.requests))
        self.assertEqual(row['adapter']['setup_cost_status'],'discovery/deployment must be separately charged; not inferred zero')

    def test_valid_wrong_candidate_remains_failure(self):
        wrong=next(c['candidate_id'] for c in self.episode['receiver']['candidates']
                   if c['candidate_id'] != self.episode['gold']['candidate_id'])
        row=self.run_episode(wrong)
        self.assertTrue(row['outcome']['answer_format_valid'])
        self.assertFalse(row['outcome']['exact_selection'])

    def test_raw_reasoning_and_redirect_halt_without_retry(self):
        self.modify=lambda p:dict(p,choices=[dict(message=dict(content='x',reasoning_content='hidden'),finish_reason='stop')])
        with self.assertRaises(GateError):self.client().complete([dict(role='user',content='x')])
        self.assertEqual(self.transport_calls,1)
        self.assertEqual(self.ledger.snapshot()[0]['status'],'halted')

    def test_role_visibility_keeps_evaluator_metadata_out(self):
        self.run_episode('invalid')
        sender=json.loads(self.requests[0]['messages'][-1]['content'])
        receiver=json.loads(self.requests[1]['messages'][-1]['content'])
        self.assertEqual(set(json.loads(sender['private_context'])),{'private_meaning'})
        self.assertEqual(set(json.loads(receiver['private_context'])),{'candidates'})
        for request in self.requests:
            rendered=json.dumps(request)
            self.assertNotIn(self.episode['gold']['episode_id'],rendered)
            self.assertNotIn(self.episode['gold']['meaning_id'],rendered)

    def test_http_redirect_denied_and_body_cap_recorded(self):
        def redirect(body,contract):return 302,b'location: unknown'
        with self.assertRaises(GateError):self.client(transport=redirect).complete([dict(role='user',content='x')])
        self.assertEqual(self.ledger.snapshot()[0]['status'],'halted')
        self.assertIsNotNone(receipts(self.ledger)[0]['response_base64'])

    def test_paired_cards_on_same_row_have_distinct_ids_and_charge_all_context(self):
        a=dict(protocol_id='nl-fixture',sender_instruction='Use concise NL.',receiver_instruction='Interpret concise NL.')
        b=dict(protocol_id='compact-fixture',sender_instruction='Use compact fields.',receiver_instruction='Interpret compact fields; '+'public domain explanation '*100)
        first=self.run_episode('invalid',card=a)
        second=self.run_episode('invalid',card=b)
        self.assertEqual(first['episode_id'],second['episode_id'])
        self.assertNotEqual(first['adapter']['receipt_event_ids'],second['adapter']['receipt_event_ids'])
        self.assertGreater(second['costs']['complete_input_tokens'],first['costs']['complete_input_tokens'])
        self.assertEqual(len(self.ledger.snapshot()),4)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM episode_results').fetchone()[0],2)
        with self.assertRaises(sqlite3.IntegrityError):self.run_episode('invalid',card=a)
        self.assertEqual(self.transport_calls,4)

    def test_wire_rejection_still_charges_sender_and_receiver(self):
        from tacit.channel import LocalTCPMessageChannel
        minimum=LocalTCPMessageChannel(lambda _:None).measure('',protocol_id='fixture-json',round_number=1,sender='sender',recipient='receiver').total_application_bytes
        row=self.run_episode('invalid',wire_budget=minimum)
        self.assertFalse(row['costs']['message_delivered'])
        self.assertGreater(row['costs']['generated_message_bytes'],0)
        self.assertEqual(len(self.ledger.snapshot()),2)
        self.assertFalse(row['outcome']['exact_selection'])

    def test_missing_usage_halts_retains_raw_and_reservation(self):
        self.modify=lambda p:dict(p,usage={})
        with self.assertRaises(GateError):self.client().complete([dict(role='user',content='x')])
        self.assertEqual(self.ledger.snapshot()[0]['status'],'halted')
        self.assertIsNotNone(receipts(self.ledger)[0]['response_base64'])
        with self.assertRaises(GateError):self.client(prefix='later').complete([dict(role='user',content='x')])
        self.assertEqual(self.transport_calls,1)

    def test_template_mismatch_and_reasoning_rejected(self):
        self.modify=lambda p:dict(p,usage=dict(p['usage'],prompt_tokens=1))
        with self.assertRaises(GateError):self.client().complete([dict(role='user',content='x')])
        self.assertEqual(self.ledger.snapshot()[0]['status'],'halted')

    def test_output_overrun_retains_observed_actual(self):
        self.response_text='x'*196
        with self.assertRaises(GateError):self.client().complete([dict(role='user',content='x')])
        row=self.ledger.snapshot()[0]
        self.assertEqual(row['status'],'halted')
        self.assertEqual(row['actual'][self.client().token_axes[1]],'49')

    def test_durable_transport_error_and_no_retry(self):
        def fail(*args):raise TimeoutError('fixture')
        with self.assertRaises(TimeoutError):self.client(transport=fail).complete([dict(role='user',content='x')])
        self.assertEqual(self.ledger.snapshot()[0]['status'],'halted')
        self.assertEqual(receipts(self.ledger)[0]['error'],'TimeoutError')
        self.assertIsNone(self.ledger.snapshot()[0]['actual'])
        exported=export_evidence(self.ledger)
        self.assertEqual(exported['ledger'][0]['status'],'halted')
        self.assertEqual(exported['model_receipts'][0]['error'],'TimeoutError')

    def test_test_source_and_cross_split_rejected_before_transport(self):
        source=dict(self.source,stage='test')
        with self.assertRaises(GateError):run_development_episode(ledger=self.ledger,scope=self.scope,source=source,row_index=0,
            episode=self.episode,condition='shared_protocol_card',split=self.split,task_seed=9,sender=self.client('sender'),receiver=self.client())
        with self.assertRaises(GateError):run_development_episode(ledger=self.ledger,scope=self.scope,source=self.source,row_index=0,
            episode=self.episode,condition='shared_protocol_card',split=build_split(seed=23),task_seed=9,sender=self.client('sender'),receiver=self.client())
        self.assertEqual(self.transport_calls,0)

    def test_actual_loopback_http_body_and_response(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=self.rfile.read(int(self.headers['Content-Length']))
                status,raw=owner.transport(body,owner.contract)
                self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=HTTPServer(('127.0.0.1',0),Handler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        try:
            contract=replace(self.contract,endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
            # Endpoint identity is a cost axis; allocate fresh policy for this fixture.
            axes={a:100000 for a in ('request_body_utf8_bytes','response_body_utf8_bytes','model_requests')}
            for a in ('input_tokens','output_tokens'):axes['receiver:'+contract.identity+':'+a]=100000
            ledger=Ledger(self.root/'http.sqlite',Policy(per_request=axes,cumulative=axes,max_calls=1))
            try:
                c=GuardedChatModel(ledger=ledger,contract=contract,role='receiver',event_prefix='http',binding='fixture',phase='train',count_prompt=self.counter,count_text=self.text_counter)
                completion=c.complete([dict(role='user',content='实际 HTTP 假服务')])
                self.assertEqual(completion.text,'invalid answer')
                self.assertEqual(ledger.snapshot()[0]['status'],'success')
                self.assertEqual(receipts(ledger)[0]['request'],self.requests[-1])
            finally:ledger.close()
        finally:server.shutdown();server.server_close();t.join()


if __name__=='__main__':unittest.main()
