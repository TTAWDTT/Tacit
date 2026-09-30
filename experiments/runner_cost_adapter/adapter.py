"""Additive ChatModel bridge to PR7's ledger and the unchanged OOD runner.

No execution CLI, server startup, model download, retry or remote API support.
Native template counting and endpoint guarantees are trusted reviewed inputs.
"""
from dataclasses import dataclass
import base64
import hashlib
import http.client
import json
import time
from urllib.parse import urlsplit

from experiments.m2_gates.budget import Charge, GateError, Outcome, digest
from tacit.runtime import ChatCompletion


def raw_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def count(value):
    if type(value) is not int or value < 0:
        raise GateError('missing or invalid native token usage')
    return value


@dataclass(frozen=True)
class Contract:
    endpoint: str
    model: str
    tokenizer_sha256: str
    template_sha256: str
    max_output_tokens: int
    max_response_bytes: int
    timeout_seconds: int
    # These claims must be independently checked on the named server, not guessed.
    free_local_confirmed: bool = False
    stateless_verified: bool = False
    reasoning_disabled_verified: bool = False
    completion_cap_verified: bool = False

    def __post_init__(self):
        u = urlsplit(self.endpoint)
        if (u.scheme != 'http' or u.hostname not in ('127.0.0.1', '::1') or
                u.username or u.password or u.query or u.fragment or
                u.path != '/v1/chat/completions' or u.port is None):
            raise GateError('explicit literal-loopback chat endpoint required; no redirects/auth/remote')
        if not self.model or not isinstance(self.model, str):
            raise GateError('pinned model required')
        for h in (self.tokenizer_sha256, self.template_sha256):
            if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise GateError('tokenizer and full chat-template SHA256 required')
        for v in (self.max_output_tokens, self.max_response_bytes, self.timeout_seconds):
            if type(v) is not int or v < 1:
                raise GateError('positive integer output/response/socket-timeout bounds required')
        if not all(v is True for v in (self.free_local_confirmed, self.stateless_verified,
                                       self.reasoning_disabled_verified, self.completion_cap_verified)):
            raise GateError('unknown endpoint pricing/state/reasoning/output guarantees block execution')

    @property
    def identity(self):
        return digest(self.__dict__)


class PartialHTTPRead(GateError):
    """Bounded response-body evidence only; never carries headers/credentials."""
    def __init__(self, status, partial, reason='incomplete_http_body'):
        super().__init__('incomplete HTTP response body')
        self.status, self.partial, self.reason = status, partial, reason


class LoopbackHTTP:
    """One stateless HTTP POST, no proxy, redirects, retry or connection reuse.

    Socket timeout bounds each I/O wait, NOT server GPU time or overall duration.
    Read cap protects client memory, NOT provider generation. Server completion
    cap must be separately verified. A timed-out model may continue server-side.
    """
    def __call__(self, body, contract):
        u = urlsplit(contract.endpoint)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=contract.timeout_seconds)
        try:
            conn.request('POST', u.path, body=body, headers={'Content-Type': 'application/json'})
            response = conn.getresponse()
            try:
                return response.status, self.read_body(response, contract.max_response_bytes)
            finally:
                response.close()
        finally:
            conn.close()


    @staticmethod
    def read_body(response, limit):
        """Incremental decoded-body evidence; framing/headers are NOT saved.

        HTTPResponse.read(amount) can discard short chunk payloads or silently
        accept short Content-Length. Read the buffered stream incrementally and
        verify declared framing ourselves. Never infer bytes not returned by fp.
        At most limit+1 body bytes are held (one over-limit sentinel).
        """
        body = bytearray()
        fp = response.fp

        def fail(reason):
            raise PartialHTTPRead(response.status, bytes(body[:limit]), reason)

        def line():
            value = fp.readline(8193)
            if len(value) > 8192 or not value.endswith(b'\r\n'):
                fail('incomplete_or_invalid_chunk_framing')
            return value

        def framing_bytes(n):
            value = bytearray()
            while len(value) < n:
                part = fp.read1(n - len(value))
                if not part:
                    fail('incomplete_chunk_delimiter')
                value.extend(part)
            return bytes(value)

        def payload(n=None):
            while n is None or n > 0:
                want = min(4096, limit + 1 - len(body))
                if n is not None:
                    want = min(want, n)
                part = fp.read1(want)
                if not part:
                    if n is not None:
                        fail('incomplete_declared_body')
                    return False
                body.extend(part)
                if len(body) > limit:
                    return True
                if n is not None:
                    n -= len(part)
            return False

        try:
            transfer = response.getheader('Transfer-Encoding')
            length = response.getheader('Content-Length')
            if transfer is not None:
                if transfer.lower().strip() != 'chunked' or length is not None:
                    fail('unsupported_or_ambiguous_transfer_framing')
                while True:
                    size_text = line()[:-2].split(b';', 1)[0]
                    if not size_text or any(c not in b'0123456789abcdefABCDEF' for c in size_text):
                        fail('invalid_chunk_size')
                    size = int(size_text, 16)
                    if size == 0:
                        trailer_size = 0
                        while True:
                            trailer = line()
                            trailer_size += len(trailer)
                            if trailer_size > 8192:
                                fail('trailer_limit_exceeded')
                            if trailer == b'\r\n':
                                return bytes(body)
                    if payload(size):
                        return bytes(body)
                    if framing_bytes(2) != b'\r\n':
                        fail('invalid_chunk_delimiter')
            if length is not None:
                if not length.isascii() or not length.isdigit():
                    fail('invalid_content_length')
                payload(int(length))
            else:
                payload()  # Valid EOF-delimited HTTP body; no length claim.
            return bytes(body)
        except PartialHTTPRead:
            raise
        except (OSError, http.client.HTTPException) as exc:
            # Only accumulated, actually-returned payload is evidence. Exception
            # partial may be framing data; never append or fabricate it.
            raise PartialHTTPRead(response.status, bytes(body[:limit]), 'body_read_error') from exc


class GuardedChatModel:
    """One ledger event per complete(), full prompts persisted before any POST.

    count_prompt receives the exact detached JSON request (including template
    options) and must count the actual server chat template + generation prefix.
    It is NOT len(text), a surrogate tokenizer, or provider usage after dispatch.
    count_text uses that endpoint's native tokenizer for the returned text.
    """
    def __init__(self, *, ledger, contract, role, event_prefix, binding, phase,
                 count_prompt, count_text, transport=None, peer_contract=None):
        if role not in ('sender', 'receiver') or phase not in ('train', 'validation'):
            raise GateError('development-only named role/phase required')
        if not event_prefix or not binding:
            raise GateError('split/config-bound identities required')
        self.ledger, self.contract, self.role = ledger, contract, role
        self.model = contract.model
        self.prefix, self.binding, self.phase = event_prefix, binding, phase
        self.count_prompt, self.count_text = count_prompt, count_text
        self.transport = LoopbackHTTP() if transport is None else transport
        self.index = 0
        peer = contract if peer_contract is None else peer_contract
        if not isinstance(peer, Contract):
            raise GateError('peer token axes require an explicit verified endpoint Contract')
        other_role = 'sender' if role == 'receiver' else 'receiver'
        peer_prefix = other_role + ':' + peer.identity + ':'
        self.supported_axes = frozenset((*self.axes(), peer_prefix + 'input_tokens', peer_prefix + 'output_tokens'))
        ledger.db.execute('CREATE TABLE IF NOT EXISTS model_receipts ('
            'event_id TEXT PRIMARY KEY, request BLOB NOT NULL, request_sha256 TEXT NOT NULL, '
            'contract TEXT NOT NULL, response BLOB, receipt TEXT, error TEXT, '
            'http_status INTEGER, response_truncated INTEGER)')
        # Add observational fields to existing receipts without resetting ledger history.
        columns = {r[1] for r in ledger.db.execute('PRAGMA table_info(model_receipts)')}
        for name in ('http_status', 'response_truncated'):
            if name not in columns:
                ledger.db.execute('ALTER TABLE model_receipts ADD COLUMN ' + name + ' INTEGER')

    @property
    def token_axes(self):
        # Separate endpoint/model/template/role axes even when names happen to match.
        p = self.role + ':' + self.contract.identity + ':'
        return p + 'input_tokens', p + 'output_tokens'

    def axes(self):
        return (*self.token_axes, 'request_body_utf8_bytes', 'response_body_utf8_bytes', 'model_requests')

    def complete(self, messages):
        detached = json.loads(raw_json(list(messages)))
        if not detached or any(not isinstance(m, dict) or set(m) != {'role', 'content'} or
                               m['role'] not in ('system', 'user', 'assistant') or
                               not isinstance(m['content'], str) for m in detached):
            raise GateError('plain role/content chat only; tools or hidden state unsupported')
        request = dict(model=self.model, messages=detached, max_tokens=self.contract.max_output_tokens,
                       temperature=0, stream=False, chat_template_kwargs={'enable_thinking': False})
        body = raw_json(request)
        # Pass a different copy to the tokenizer; mutation cannot change the wire.
        input_tokens = count(self.count_prompt(json.loads(body)))
        event_id = self.prefix + '/' + self.role + '/' + str(self.index)
        self.index += 1
        axes = self.ledger.policy.axes
        if set(axes) - self.supported_axes:
            raise GateError('unsupported budget axis; unknown cost cannot be reserved/measured as zero')
        if not set(self.axes()) <= set(axes):
            raise GateError('ledger lacks exact endpoint/role/native token or serialization axes')
        work, context = {a: 0 for a in axes}, {a: 0 for a in axes}
        ia, oa = self.token_axes
        context[ia] = input_tokens
        context['request_body_utf8_bytes'] = len(body)
        work[oa] = self.contract.max_output_tokens
        work['response_body_utf8_bytes'] = self.contract.max_response_bytes
        work['model_requests'] = 1
        charge = Charge(work, context)
        self.ledger.db.execute('INSERT INTO model_receipts(event_id,request,request_sha256,contract) VALUES (?,?,?,?)',
            (event_id, body, hashlib.sha256(body).hexdigest(), json.dumps(self.contract.__dict__, sort_keys=True)))

        def operation():
            started = time.perf_counter()
            try:
                status, raw = self.transport(body, self.contract)
                elapsed = time.perf_counter() - started
                if not isinstance(raw, bytes):
                    raise GateError('transport must return raw response bytes')
                self.ledger.db.execute('UPDATE model_receipts SET response=?,http_status=?,response_truncated=? WHERE event_id=?',
                                       (raw, status, int(len(raw) > self.contract.max_response_bytes), event_id))
                if len(raw) > self.contract.max_response_bytes or status != 200:
                    raise GateError('HTTP error/redirect or response read cap exceeded')
                payload = json.loads(raw)
                choice = payload['choices'][0]
                message = choice['message']
                text, usage = message['content'], payload['usage']
                if (not isinstance(text, str) or payload.get('model') != self.model or
                        message.get('tool_calls') or message.get('function_call') or
                        message.get('reasoning_content') not in (None, '')):
                    raise GateError('wrong model, tools or hidden reasoning unsupported')
                used_input = count(usage.get('prompt_tokens'))
                used_output = count(usage.get('completion_tokens'))
                details = usage.get('completion_tokens_details') or {}
                if details.get('reasoning_tokens') not in (None, 0):
                    raise GateError('reasoning tokens contradict disabled reasoning contract')
                if used_input != input_tokens or count(self.count_text(text)) > used_output:
                    raise GateError('native token/template receipt mismatch')
                actual = {a: 0 for a in axes}
                actual.update({ia: used_input, oa: used_output,
                               'request_body_utf8_bytes': len(body),
                               'response_body_utf8_bytes': len(raw), 'model_requests': 1})
                receipt = dict(role=self.role, phase=self.phase, request_sha256=hashlib.sha256(body).hexdigest(),
                    model=self.model, tokenizer_sha256=self.contract.tokenizer_sha256,
                    template_sha256=self.contract.template_sha256, input_tokens=used_input,
                    output_tokens=used_output, finish_reason=choice.get('finish_reason'),
                    reasoning_tokens_observed=details.get('reasoning_tokens'),
                    reasoning_contract='disabled; server verification required',
                    elapsed_seconds=elapsed, provider_generation_seconds=usage.get('generation_seconds'),
                    external_api_fee='0: separately confirmed free loopback', local_compute_cost=None,
                    transport_boundary='HTTP JSON body only; headers/TCP excluded', retries=0,
                    cache_hit=None, session='new HTTP connection; supplied messages only',
                    accounting_success='valid receipt, NOT strict task success')
                self.ledger.db.execute('UPDATE model_receipts SET receipt=? WHERE event_id=?',
                                       (json.dumps(receipt, allow_nan=False), event_id))
                completion = ChatCompletion(text, self.model, used_input, used_output, elapsed,
                    payload.get('id'), choice.get('finish_reason'), False)
                # Task success is determined by unchanged run_condition and stored separately.
                return Outcome(completion, actual, True)
            except PartialHTTPRead as exc:
                evidence = dict(http_status=exc.status, response_truncated=True,
                    partial_body_bytes=len(exc.partial), actual_usage=None,
                    transport_boundary='HTTP response body only; headers excluded', http_read_error=exc.reason)
                self.ledger.db.execute('UPDATE model_receipts SET response=?,http_status=?,response_truncated=1,receipt=?,error=? WHERE event_id=?',
                    (exc.partial, exc.status, json.dumps(evidence), type(exc).__name__, event_id))
                raise
            except BaseException as exc:
                self.ledger.db.execute('UPDATE model_receipts SET error=? WHERE event_id=?',
                                       (type(exc).__name__, event_id))
                raise
        try:
            return self.ledger.run(event_id, phase=self.phase, candidate=self.prefix,
                binding=self.binding, charge=charge, operation=operation)
        except BaseException as exc:
            self.ledger.db.execute('UPDATE model_receipts SET error=COALESCE(error,?) WHERE event_id=?',
                                   (type(exc).__name__, event_id))
            raise


def receipts(ledger):
    rows = ledger.db.execute('SELECT event_id,request,request_sha256,contract,response,receipt,error,http_status,response_truncated FROM model_receipts ORDER BY rowid').fetchall()
    return [dict(event_id=r[0], request=json.loads(r[1]), request_sha256=r[2], contract=json.loads(r[3]),
                 response_base64=base64.b64encode(r[4]).decode() if r[4] is not None else None,
                 receipt=json.loads(r[5]) if r[5] else None, error=r[6], http_status=r[7],
                 response_truncated=bool(r[8]) if r[8] is not None else None) for r in rows]


class NativeCounters:
    """Count with an already-loaded, pinned local tokenizer; never loads/downloads.

    Server must use the same chat template, generation prefix and options. The
    request/receipt equality check detects a discrepancy after the first call;
    it cannot retroactively prevent that first call's input overrun.
    """
    def __init__(self, tokenizer, contract):
        template = tokenizer.chat_template
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != contract.template_sha256:
            raise GateError('loaded template differs from frozen endpoint template')
        self.tokenizer = tokenizer

    def prompt(self, request):
        tokens = self.tokenizer.apply_chat_template(request['messages'], tokenize=True,
            add_generation_prompt=True, **request['chat_template_kwargs'])
        if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
            raise GateError('native tokenizer must return a single token ID sequence')
        return len(tokens)

    def text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
