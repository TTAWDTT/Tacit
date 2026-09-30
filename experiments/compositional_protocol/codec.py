"""Offline typed partial records. Same wire syntax as compact fields, no LLM calls.

The schema is public, fixed before evaluation and independent of any target.
A scope is supplied by the trusted channel, never inferred from message text.
"""
from collections.abc import Mapping
from types import MappingProxyType


def _atom(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(c in value for c in ';=')
            or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise ValueError('expected delimiter-free Unicode text without edge whitespace or controls')
    return value


class TypedFields:
    """Finite product schema; composition is disjoint union, not inference."""

    def __init__(self, domains):
        if not isinstance(domains, Mapping) or not domains:
            raise ValueError('nonempty domain mapping required')
        checked = {}
        for key, values in domains.items():
            _atom(key)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError('each domain must be a nonempty list or tuple')
            values = tuple(_atom(v) for v in values)
            if len(set(values)) != len(values):
                raise ValueError('duplicate domain value')
            checked[key] = frozenset(values)
        self.domains = MappingProxyType(checked)

    def _scope(self, scope):
        if not isinstance(scope, (list, tuple)) or not scope:
            raise ValueError('explicit nonempty scope required')
        if any(not isinstance(k, str) for k in scope):
            raise ValueError('scope keys must be strings')
        if len(set(scope)) != len(scope) or not set(scope) <= self.domains.keys():
            raise ValueError('duplicate or unknown scope key')
        return set(scope)

    def validate(self, record, *, scope):
        keys = self._scope(scope)
        if not isinstance(record, Mapping) or set(record) != keys:
            raise ValueError('record keys must equal authorized scope')
        for key, value in record.items():
            _atom(value)
            if value not in self.domains[key]:
                raise ValueError('value outside field domain')
        return dict(record)

    def encode(self, record, *, scope):
        record = self.validate(record, scope=scope)
        return ';'.join(f'{k}={record[k]}' for k in sorted(record))

    def decode(self, message, *, scope):
        if not isinstance(message, str):
            raise ValueError('message must be text')
        record = {}
        for field in message.split(';'):
            if field.count('=') != 1:
                raise ValueError('each field requires exactly one equals sign')
            key, value = field.split('=')
            if key in record:
                raise ValueError('duplicate field')
            record[key] = value
        record = self.validate(record, scope=scope)
        if self.encode(record, scope=scope) != message:
            raise ValueError('noncanonical field order')
        return record

    def compose(self, scoped_messages, *, scope):
        """Combine independently authorized fragments; reject any overlap/gap.

        scoped_messages contains (trusted_scope, message) pairs, not sender IDs
        asserted by untrusted content. Caller owns channel identity enforcement.
        """
        result = {}
        for part_scope, message in scoped_messages:
            part = self.decode(message, scope=part_scope)
            if result.keys() & part.keys():
                raise ValueError('overlapping scopes')
            result.update(part)
        return self.validate(result, scope=scope)
