# Response contract diagnostics v0.1 (work package B)

## Scope and baseline

Baseline HEAD: `5dd260c5824b280b5323a1707ab132524afa8e4c`
(2026-09-30 13:10:32 UTC), repository `TTAWDTT/Tacit`. The cloud
checkout had branch `work`, no local `main` reference, and a clean worktree.
Work branch: `codex/tacit-response-contract`. No remote freshness check was
performed. No repository AGENTS.md or applicable .agents/skills were present.

This implements the offline follow-up to
[the Qwen3 audit](LOCAL_QWEN3_CAPABILITY_SCREEN_AUDIT_V0_1.md).
It changes neither the runtime adapter, launcher, runner nor scorer.
The existing `_parse_choice` accepts surrounding whitespace via `strip()`;
otherwise the complete content must equal a candidate ID. The new diagnostic
calls that same parser on the unmodified string. It never removes wrappers,
extracts a trailing answer or substitutes reasoning for content.

## Offline entry point

From the repository root, with Python >=3.10 and no extra dependencies:

```sh
python -m unittest discover -s tests -p test_response_diagnostics.py -v
python -m experiments.emergent_ood_v0_4.response_diagnostics tests/fixtures/response_contract/offline.json
python -m unittest discover -s tests -p test_emergent_ood_v04_runner.py -v
python -m unittest discover -s tests -p test_tacit_runtime.py -v
```

The CLI reads a local JSON object with `configuration`, `candidate_ids`, and
`response` (an OpenAI-compatible API envelope), and writes only diagnostic JSON
to stdout. It contains no client, service launch, model read or network operation.
The fixture is entirely synthetic, including all identifiers and candidate IDs.
The diagnostic is not a runner ledger and cannot satisfy the capability gate.

Input configuration requires model ID, model artifact ID, template ID, service
build ID, service configuration ID, reasoning mode and reasoning format. These
are operator-supplied identifiers, explicitly marked unverified. Use immutable
revisions/digests referencing a separate local configuration manifest; include
binary/version, effective template, endpoint, token cap, context, threads and
server flags in that manifest. Do not put credentials or raw templates in IDs.
The returned model is compared with the configured model without echoing arbitrary
provider text. This tool does not verify artifact hashes or actual server flags.

Diagnostics distinguish candidate-ID content, empty/whitespace content, think
markers (including an unfinished opening tag), candidate ID with extra text,
unrecognized text, missing/non-string content and malformed envelopes.
`extra_text` is a heuristic observation, not a semantic answer extraction.
`strict_format_valid` is format validity only: a wrong but listed ID is valid.
`truncated` is independently true only when `finish_reason` is `length`.
A syntactically valid ID with `length` remains format-valid under the unchanged
parser; the diagnostic still flags truncation. Missing finish reason does not
establish that generation completed. No token-count heuristic claims truncation.

For `reasoning_content` and `reasoning`, only key presence and JSON type are
emitted (missing, null, empty string and other JSON types are distinguished).
No values, excerpts, lengths or hashes of reasoning are recorded. Content is also
never emitted. Malformed reasoning types remain visible diagnostically, even
though the existing endpoint adapter may reject them. Keep raw API inputs private
and local (e.g. ignored `.cache/`); publish only reviewed metadata, never raw
responses. Input error messages suppress source text. Configuration identifiers
are caller-supplied public metadata and must be reviewed before publication.

## Later acceptance: reasoning off / deepseek (NOT EXECUTED)

The combination `--reasoning off --reasoning-format deepseek` remains unverified.
This offline fixture does not establish behavior for any Qwen3/llama.cpp build.
For a separately authorized future run:

1. Obtain a fresh eligible preflight under the frozen resource policy. A rejected
   preflight stops before model artifact access, service start or requests.
2. Record actual model/artifact, effective template, service build/configuration
   identifiers and the two reasoning controls before requests. Keep task, prompt,
   endpoint, 48-token cap and strict scorer fixed. Do not assume upstream option
   names prove behavior of the installed binary.
3. Use only the frozen independent training calibration block: k=4, task_seed=0,
   calibration split seed 17, three balanced sets / 12 calls. Evaluation seed 23
   stays separate. Do not inspect held-out answers to configure the service.
4. At the API boundary, feed each raw envelope into `diagnose_response` before
   adapter validation, using that request's candidate IDs and configuration.
   Store only the returned metadata in a separate diagnostic sidecar, joined by
   an externally supplied calibration episode ID. This module intentionally has
   no automatic runner/launcher hook; the future capture integration is still
   required. Do not write raw reasoning to the sidecar or change content passed
   to the runner. The offline CLI can inspect already-authorized local captures.
5. Stop and investigate if content is empty/malformed/wrapped or truncation is
   reported. An absent finish reason or unverifiable configuration requires
   further interface verification, not a claim of success. Do not fix failed
   answers after generation. If adapting the endpoint becomes necessary, freeze
   a separate contract before new observations.
6. The existing runner capability validator must accept the exact original
   12/12 format-valid and correct training ledger and its manifest before any
   communication arm is eligible. Diagnostic flags alone never open that gate.

No preflight, actual model/template/service validation, inference, external GPU,
paid API, model download, push, PR or merge occurred in this work package.
These are software tests and engineering diagnostics, not scientific results.

## Verification evidence (2026-09-30)

The commands above completed successfully on this branch: 7 diagnostic tests,
26 existing v0.4 runner tests and 24 existing runtime tests (57 total). The CLI
fixture returned `candidate_id`, `strict_format_valid=true`, `truncated=false`,
and reasoning_content presence/type `true/string`; all identifiers were synthetic.
The CLI test forbids socket creation. Negative fixtures retain strict rejection
of think wrappers and extra text; privacy sentinels in content/reasoning/provider
metadata are absent from diagnostic output. `git diff --check` passed.

These checks cover the new module and the existing affected contracts; the entire
repository suite was not run. No actual response capture or server configuration
was tested. Historical 0/12 observations in the source audit are unchanged.
