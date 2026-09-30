# Quoted labeled-fields counter-baseline v0.1

**Status:** model-free robustness and byte-accounting audit; not a prompt card, LLM result, or protocol-superiority claim.

## Question

The frozen compact labeled-fields v0.3 serializer saves 18 UTF-8 payload bytes against compact JSON on the current three synthetic ontologies. It rejects keys and values containing `;` or `=`, however. Is most of the measured saving a general property of labeled records, or a consequence of the fixture alphabet?

## Standard-based challenger

The challenger keeps the sorted `field=field;...` framing but encodes each key and value as a JSON string. Quoting and escaping use RFC 8259's JSON string syntax; messages are UTF-8 and reject unpaired surrogate code points for interoperable Unicode text. The deterministic implementation is [`quoted_labeled_fields.py`](quoted_labeled_fields.py). It handles delimiter characters inside fields, quotation marks, reverse solidus, control characters, and non-ASCII Unicode without a custom escape convention.

This borrows a mature serialization primitive; it does not establish a new language. The reference point is [RFC 8259, sections 7 and 8](https://www.rfc-editor.org/rfc/rfc8259.html).

## Exact byte result

For a mapping of `n ≥ 1` string pairs, sort keys identically and serialize every key/value using the same canonical JSON-string function. Compare:

- compact JSON object: `{` + each `JSON(key):JSON(value)` pair separated by `,` + `}`;
- quoted labels: each `JSON(key)=JSON(value)` pair separated by `;`.

The per-pair JSON quotes and contents are identical. `:` and `=` each occupy one ASCII byte; `,` and `;` each occupy one ASCII byte. Therefore:

\[
|J| - |Q| = 2,
\]

because the only removed bytes are the JSON object's opening and closing braces. This remains true when keys or values contain delimiters or require JSON escapes. It is independent of record width and payload contents under the stated canonicalization.

The finite held-out support check covers all 64 held-out tuples in each of the default, robotics, and music fixtures (192 tuples total). On every row, v0.3 saves 18 payload bytes versus compact JSON, while the standard-string-safe challenger saves exactly 2. These are exact UTF-8 serialization lengths, not tokenizer costs, network framing, total request cost, or task success.

## Interpretation and decision

The 16-byte difference between the two field formats comes from omitting quotes around ordinary labels and values under a restricted alphabet. Quoting restores general string safety but leaves only a two-byte punctuation advantage over the standard JSON object. The two-byte difference can be erased by tokenizer segmentation, envelope framing, or any one extra prompt token; only a matched model/tokenizer and end-to-end run can determine that.

Keep v0.3 as a restricted-alphabet baseline for the existing fixtures and use JSON when arbitrary string content or standard interoperability is required. Do not promote the quoted wrapper as a more efficient language: it is a robustness challenger whose byte advantage over JSON is structurally limited to two braces. A future protocol claim must show semantic success, full input/output token and wire costs, receiver transfer, and setup cost on actual communication-dependent tasks.

## Verification and limits

Six focused tests cover special characters, Unicode, canonical and valid-noncanonical escapes, duplicate labels, malformed strings, unpaired surrogates, and exhaustive round trips/byte comparisons over the 192 held-out meanings. No models, services, private evaluator keys, or endpoints were read or contacted. The ontologies are hand-authored synthetic fixtures, and bytes do not predict model-token use.
