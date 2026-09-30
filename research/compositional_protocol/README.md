# Route A: typed composition after the compact-fields equivalence check

Disposition: **revise / narrow**. On the existing finite-product tasks, explicit
labels and product composition are already present in compact fields. Reject the
strong candidate claim that this route adds a new compositional communication
mechanism or automatically compresses messages. Keep only a minimal, offline,
target-independent schema/scope validator and a falsification experiment. No LLM
advantage, qualification pass, or complete-cost frontier is established.

Plan: https://chatgpt.com/space/page_bd3d1298bd5c81919e4101089077ae2e

## Evidence and provenance

Actual fetched main and branch base: `5dd260c5824b280b5323a1707ab132524afa8e4c`.
Branch: `codex/tacit-compositional-research`. The exact publication head is in the
PR body (a file cannot self-contain its own commit hash). No AGENTS.md or repository
`.agents/skills` files existed in the checked workspace/base. Read the experiment
plan, readiness report, v0.4 split/episodes/compact card and v0.3 Private Match
protocols/preregistration before implementing this independent extension.

PRs #1/#2/#3 were open and unmerged at inspection, respectively at
`6de34bfd821aa8656d9063378087afc362aa9252`,
`05d2921ead5ac8f555b7f776974f907de5bbc5a4`, and
`ab06295aa5e0331c414f002b077dd71293c9c6f9`.
There is **no dependency** on those branches; no files were copied from them.
All additions are inside `experiments/compositional_protocol/` and
`research/compositional_protocol/`. Existing runners, scorers, cards, protocols,
preregistrations and conditions remain unchanged. This is not a registered arm.

## Actual task and baseline comparison

| Existing artifact | Already supplies | Increment left for this reference |
| --- | --- | --- |
| `experiments/emergent_ood_v0_4/compact_fields.py`, generic-v3 card | Canonically sorted labeled facts, unambiguous delimiters, duplicate rejection, exact key-set and evaluator-target fidelity audit | Public finite-domain validation without consulting gold; checked partial-record composition |
| `experiments/private_match_v0_3/protocols.py`, compact_kv | Coordinate labels, domain checks and sender-role binding; one coordinate per sender | A reusable disjoint-union check; no new message or task capability |
| v0.4 symbolic condition / induced compositional cards | Explicit positional/factorized codes | No claim that factorization is new |
| `research/quoted_labeled_fields.py` | JSON-string quoting for delimiter/Unicode safety | None; this reference intentionally rejects delimiter-containing inputs instead of inventing another escape scheme |

The new encoder emits **the same bytes** as the old compact encoder on every valid
full v0.4 tuple; for one-coordinate Private Match it emits the same compact_kv
bytes. It adds no information selection, abbreviation, learned vocabulary,
receiver adaptation, relation calculus, or new ontology. This choice makes the
null identifiable instead of disguising serialization as a research mechanism.
A schema validator is engineering, not a new language. An instruction explaining
finite types could still affect a bounded receiver, but that is an empirical
instruction/validation effect; compact fields plus the same instruction is its
mandatory control. The scope comes from trusted channel metadata; the code does
not authenticate senders or sandbox role ledgers.

## Definitions and properties

Let the public schema be finite domains D_k for field keys k in K. A meaning is a
record m in the product of D_k. A sender with authorized scope S subset K sees
only m restricted to S; its encoder E_S sorts keys and joins `k=v` fields by `;`.
Keys and values must be nonempty Unicode scalar strings, contain neither `;` nor
`=`, have no surrounding whitespace or Unicode control characters (category Cc: C0, DEL and C1). The legal language
L_S is exactly the canonical encodings of records with key set S and v in D_k.
D_S is strict parsing followed by domain and scope validation. Semantics are the
conjunction of field equalities, not an answer ID. A receiver matches those
facts against its private candidate table and must have a unique exact match.

1. **Round trip and injectivity.** Under the delimiter restrictions and unique
   keys, each field boundary and binding has one parse. Sorting gives one
   canonical order. Thus D_S(E_S(m))=m and E_S(m)=E_S(n) implies m=n. This covers
   unseen combinations as long as each value is in the public schema. It also
   holds for existing compact fields; it is not an advantage over them.
2. **Composition.** If authorized scopes form a disjoint partition of K,
   decoding each fragment and taking their union yields m. Any grouping/order
   of this union has the same semantics. Overlap (even identical values), missing
   fields, out-of-scope fields and illegal values fail explicitly. This is a
   partial operation on records, not raw-string concatenation, and it cannot
   recover information that no sender observed.
3. **Information bound.** For full product support and a receiver context in
   which every pair of meanings may need distinction, any zero-error encoder
   needs at least product_k |D_k| distinct messages, or ceil(log2(product_k
   |D_k|)) fixed-width bits. For Private Match's independent uniform q-valued
   sources, fixing the other source proves each source must distinguish q
   values, giving 2 log2(q) total fixed-width bits. Types do not beat these
   bounds. Finite candidate co-occurrence/support restrictions may weaken them;
   the existing held-out-rank codec already exploits such restrictions. Do not
   claim the full-product bound for a restricted split or measured wire bytes.

All properties concern deterministic reference code. A type-correct false value
is still false; neither the decoder nor the receiver has access to sender truth
for fidelity scoring. Evaluator fidelity and strict final-answer success remain
separate. None of the proofs implies an LLM obeys the grammar.

## Counterexamples and conclusions that can fail

- **Existing-compact null:** enumeration gives equal messages and equal payload
  lengths on every tuple. If the receiver prompt, serialized boundary and random
  state are identical, renaming this encoder cannot change its output law. Any
  deterministic contrast here is an implementation error or changed condition.
- **Redundant type labels:** a fixed positional vector already binds values to
  keys. Removing labels while retaining the public order is not removing all
  types. Test it as a strong baseline, not an intentionally ambiguous straw man.
  Erasing both order and labels does collapse `{a:0,b:1}` and `{a:1,b:0}` when
  domains overlap. Existing compact labels solve this just as well.
- **No decomposable sufficient facts:** two worlds with identical color/shape
  but opposite left/right relations encode identically here and demand different
  answers. At most one can be decoded correctly by any deterministic receiver.
  A random holistic answer map over M tuples requires its independent mapping
  information; knowing atom names does not determine unseen mapping entries.
  Such a task is outside the current protocol and is not invented as a new win.
- **Undetectable substitution:** `a=blue;b=one` is legal even if the true a was
  red. Domain validation rejects out-of-domain strings but does not establish
  truth. No silent repair/retry is supplied.
- **No amortization rescue:** with equal per-use wire cost and positive extra
  setup cost ΔF, mean extra cost is ΔF/N > 0 at every finite N. There is no finite
  wire break-even. Repeated card text in stateless prompts is per-use, not setup.
  Tokenizer cost or receiver error improvements would require new measurements.
- **Weak/new receiver:** an unfamiliar grammar or lengthy type card may increase
  errors, input tokens or refusals. A flat codebook supplied in full may decode
  perfectly; withholding its unseen entries only demonstrates missing coverage,
  not a fair advantage over an equal-information baseline.

## Reproduction and observed scope

From the repository root, Python 3.10+ standard library only:

```sh
python -m unittest experiments.compositional_protocol.test_codec -v
python -m experiments.compositional_protocol.falsify
python -m unittest tests.test_compact_labeled_fields tests.test_private_match_v03 tests.test_emergent_ood_v04_split -v
```

The first suite has 9 test methods, exhausts 256 meanings for each seed 17/23/41
(768 round trips and wire comparisons; 64 held-out per seed), checks train atom
coverage and complete-tuple exclusion, and verifies all 16 q=4 Private Match
coordinate pairs against the original strict scorer and compact_kv encoder.
Repeated tuples across seeds are **not** 768 independent statistical samples.
It also exercises illegal syntax, invalid schemas, duplicate/conflicting scopes,
Unicode, gold/extra-field rejection, hidden-y invariance of x messages, valid
false values and the missing-relation counterexample. Public fixture keys and
synthetic full enumeration are for code tests only, never a sealed model test.

`offline_report.json` is the deterministic falsifier output with source hashes.
Its byte totals are **payload only**; model tokens, inference, complete transport,
latency and actual fees are unmeasured, not zero. Tests do not certify model tool
isolation, sender authentication, prompt fairness or real transfer.

The companion [experiment specification](EXPERIMENT_SPEC.md) describes the smallest
remaining empirical question and the gates required before any model call.

Validation after the control-character correction: 9 new tests and 38 existing focused regression tests
passed; `compileall` and `git diff --check` passed. The broader repository suite
was not run. No real model inference was attempted. Remote head and PR CI status
are recorded in the PR body and Space status after publication, not inferred from
local test success.

Fixed-head review correction: DEL and C1 controls were previously accepted despite
the documented control-character exclusion. The codec now rejects Unicode category
Cc, with exhaustive C0/DEL/C1 regression coverage and permitted Unicode boundary
checks. The payload-equivalence result is unchanged; the report source hash is refreshed.
