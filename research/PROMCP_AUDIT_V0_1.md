# ProMCP audit v0.1

**Research date:** 2026-09-30  
**Sources:** Anjum et al., *ProMCP: Profiling Token Flows and Latency Costs in Model Context Protocol–Based LLM Agents*, Findings of ACL 2026; paper §§3–5 and Appendix A; authors' public repository.  
**Scope:** identify which ProMCP methods and findings transfer to measurement of LLM-agent communication. This is a source audit, not a replication.

## What the work measures

ProMCP instruments the MCP Host–Client–Server path as six stages: user prompt to LLM, LLM plan to client, client tool request, tool response, context update to the LLM, and final answer. It also measures session initialization and `tools/list` discovery before user tasks. The paper separates provider/local-model usage counts from a tokenizer-based footprint estimate for protocol artifacts such as schemas, tool calls, and tool results. This distinction matters: bytes crossing an agent/runtime boundary, text inserted into an LLM context, and provider-billed prompt tokens are related but not interchangeable measures.

The evaluation uses MCP-Bench (30 single-server tasks) and MCP-Universe (125 longer tool workflows), spanning 20 servers and 169 tools. Three deployment arrangements differ in more than wire format: local Ollama models with a custom client; Claude API with a custom client; and Claude Desktop as an off-the-shelf client. The local models include Mistral Small 3.2 24B and Llama 3.2. The paper's own evaluation card records different decoding, retry, streaming, and observability settings across configurations.

## Reported observations and their limits

- Discovery can create a pre-task cost. Tool schema serialization and transport dominate discovery; the reported connection latency depends on whether the server is a persistent HTTP/SSE process or a newly spawned STDIO subprocess.
- In the customized clients, model planning/schema context is a large input-side cost. The abstract reports 56–72% of tokens and 60–67% of latency in planning/schema injection. This is not the marginal cost of a semantic message codec: it includes client choices and the supplied tool inventory.
- Deferred tool loading in Claude Desktop lowers its planning share, but the workflow shifts cost into retained tool results and final synthesis. In MCP-Universe, the paper attributes a large result-context cost to retaining full web-search JSON and replaying it in later turns. This is an orchestration/context policy effect, not a message grammar effect.
- Tool execution itself is a small share in these light-to-moderate workloads. The paper explicitly says heavy production I/O can change this regime.
- The paper reports a separate 100-task quality audit (85–100% tool-call accuracy and execution success in its listed configurations, with human-rated answer quality 4.06–4.91/5). It does not present a task-success-versus-equal-channel-budget frontier for alternative semantic languages.

The C-OTS trace is reconstructed after the fact from exported conversation logs, so hidden retries and internal timing detail are unavailable. All measurements use one Windows 11 / RTX 4090 / i9-13900K workstation. Tool inventories, models, client access, streaming, and retry controls differ between topologies. Therefore neither the reported token/latency shares nor the quality figures can be transferred as a causal estimate for Tacit's protocol comparisons.

## Relevance to Tacit

ProMCP is strong prior art for cost attribution and a weak direct baseline for this project's central question. MCP standardizes application-to-tool/context exchange; the paper does not compare how two agents encode complementary private task information. Its six-stage event model also differs from Tacit's current one-task/one-or-more-agent episode record.

Tacit's `tlu.costs.v3` already records complete model input/output usage separately from the serialized agent-boundary payload and its framing, and charges setup artifacts where declared. This means a bare two-agent experiment should continue to count protocol instructions in model input cost without relabeling them as transmitted semantic payload. If a future deployment includes MCP discovery or context orchestration, record those as a distinct setup/runtime stage and count schemas/results at both their actual transport boundary and their actual LLM-context boundary. Do not silently fold tool schemas into the language's message bytes, or assume that a prompt footprint estimate equals provider usage.

**Decision:** no benchmark, task, or protocol-arm change is licensed by this paper. For an MCP-integrated artifact, borrow the stage timestamps and initialization ledger as an optional runtime extension after the core communication frontier has been established. Keep codec, context-selection policy, and MCP deployment topology as separate factors.

## Primary sources

- [ACL Anthology paper and PDF](https://aclanthology.org/2026.findings-acl.1967/)
- [Authors' public implementation](https://github.com/ResponsibleAILab/ProMCP)
- Method detail: paper §§3.1–3.5 and 4.1–4.3; limitations and configuration differences: §5 and Appendix A.

No repository was cloned, no code/dependencies/data were downloaded, and no model or benchmark was run for this audit.
