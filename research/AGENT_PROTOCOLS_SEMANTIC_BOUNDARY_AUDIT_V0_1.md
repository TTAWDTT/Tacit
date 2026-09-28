# Agent communication protocol boundary audit v0.1

**Research date:** 2026-09-29  
**Scope:** FIPA ACL, KQML, MCP 2026-07-28, and A2A 1.0.0. This is a specification-level comparison, not an adoption or runtime-performance study.

## Question

Do existing agent protocols already answer what semantic language LLM agents should use to exchange task-relevant information, or do they primarily standardize message acts, lifecycle, and interoperability?

## Findings

### KQML

The 1993 KQML draft defines a message format and message-handling protocol for run-time knowledge sharing. Its performatives (for example `ask-one`, `tell`, `reply`, and `subscribe`) specify the kind of communicative operation. The content is explicitly carried in the sender's representation language; the `:language` and `:ontology` fields identify that representation and its assumed domain vocabulary. The original draft says formal semantics for its basic performatives and associated protocols were not yet defined, and described work in progress toward such semantics. A later account of KQML's design likewise separates content, message, and communication layers and notes that KQML implementations generally treat the content as opaque except for locating its boundaries. [1993 KQML draft](https://research.cs.umbc.edu/kqml/kqmlspec/spec.html) · [Finin et al., *KQML—A Language and Protocol for Knowledge and Information Exchange*](https://research.cs.umbc.edu/kqml/papers/kbkshtml/kbks.html)

**Implication:** KQML is a genuine agent communication language with pragmatic message acts, not a task-oriented semantic codec. It leaves domain content syntax and ontology alignment to the participants. Its layered design is a useful precedent for keeping a Tacit content code separable from its delivery envelope.

### FIPA ACL

FIPA separates the ACL message structure (SC00061) from the communicative act library (SC00037). The library gives formal semantic characterizations for acts such as `inform`, `request`, and `agree`, grounded in assumptions about the agents' beliefs, desires, and intentions. The message structure carries a `content` field and may identify the content `language` and `ontology`; the act's semantics therefore does not by itself define a compact, shared representation for arbitrary task content. [FIPA Communicative Act Library Specification, SC00037J](https://jmvidal.cse.sc.edu/library/XC00037H.pdf) · [FIPA ACL Message Structure Specification, SC00061G](https://www.yumpu.com/en/document/view/35150485/fipa-acl-message-structure-specification)

**Implication:** FIPA ACL is semantically richer at the speech-act/conversation layer than a generic RPC envelope. It still does not establish which representation minimizes task distortion or receiver effort for a given LLM pair and task distribution. The user-facing project must not claim it is inventing agent performatives or conversation acts.

### MCP (version pinned)

This audit uses the official [2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28), read on 2026-09-29. It defines JSON-RPC messages among an LLM application host, clients, and servers, and standardizes context/resources, prompts, tools, capability negotiation, and supporting operations. Its 2026-07-28 base protocol describes stateless self-contained requests and per-request capability negotiation; this differs from the stateful session and initialization model in the pinned 2025-11-25 revision. [2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28) · [2025-11-25 architecture](https://modelcontextprotocol.io/specification/2025-11-25/architecture)

**Implication:** MCP is primarily an application-to-tool/context integration contract, not a peer-agent semantic language. A future Tacit runtime could expose or consume MCP tools, but MCP compatibility would not demonstrate that the inner message representation is efficient. Any cost study must identify the version and count both the semantic payload and complete serialized request/response, including relevant protocol metadata.

### A2A 1.0.0

The pinned [A2A 1.0.0 specification](https://a2a-protocol.org/v1.0.0/specification/) standardizes communication between independent agents through agent discovery/capability descriptions, messages, tasks, status updates, artifacts, security, and transport bindings. Its content model supports typed message parts, including text and structured data; tasks track work and artifacts carry outputs. Thus A2A already supplies a meaningful inter-agent exchange and lifecycle layer, while leaving task-specific message representation and the agent's interpretation of content to the participating implementations.

**Implication:** A2A is a deployment/interoperability substrate and should be treated as such. If a protocol becomes viable, test it inside the same A2A part/container where feasible. Report inner semantic payload cost and complete serialized A2A cost separately; do not compare a bare compact payload against a full A2A message or mistake an envelope-level saving for semantic compression.

## Synthesis: protocol stack boundary

These standards are not interchangeable and none is a straw baseline:

| Layer/question | Relevant existing work | What remains open for this project |
|---|---|---|
| What act is being performed, and what response is licensed? | KQML performatives; FIPA communicative acts/conversation protocols | Whether LLMs can reliably follow such acts; not a new contribution by itself |
| How are tools, capabilities, tasks, and asynchronous results interoperably exchanged? | MCP; A2A | Runtime integration and whole-envelope cost; not semantic-code superiority |
| What task-relevant information should cross the boundary, and in what code? | Content language / ontology slots in KQML/FIPA; text/structured parts in MCP/A2A | Task-conditioned fidelity, receiver effort, bandwidth frontier, compositionality, transfer, robustness, and setup amortization |

This distinction is a working decomposition, not a claim that the layers are perfectly separable: a container, schema, interaction act, or tool interface can change model behavior and therefore must be held fixed or measured in an experiment.

## Falsifiable experimental consequences

1. **Codec comparison:** freeze agent roles, act/task lifecycle, schedule, prompt scaffold, and transport envelope. Vary only the inner content representation where possible. Score message adherence, decoded meaning, task success, and all cost dimensions separately.
2. **System comparison:** when comparing complete systems, allow different acts, schedules, and formats only as a declared policy/system comparison; count discovery, negotiation, extra calls, parsing, repair, and repeated context.
3. **Envelope accounting:** serialize every arm through a declared common envelope for a portability claim. Report payload bytes and whole-message bytes as distinct axes. No zero-cost assumption for setup, dictionaries, schema exchange, or shared runtime state.
4. **Interop is a separate success criterion:** a language may be semantically efficient but fail to negotiate or recover across heterogeneous implementations. Conversely, a standards-compliant message can be interoperable without carrying task-sufficient information.

## Limitations and next work

- The FIPA message-structure specification was accessed through an archived standards mirror because the historical FIPA URL did not resolve; verify normative details against the IEEE/FIPA archive before implementation.
- This pass did not benchmark conformance, adoption, actual wire overhead, or model behavior under these standards.
- Next: inspect current A2A/MCP SDK serialization and extension behavior only when choosing a concrete runtime; keep the 2026-07-28 MCP revision pinned in any reproducible comparison.
