# Task-grounded claim audit

**Status:** `tlu.claim-audit.v1` is a descriptive label and aggregation contract. It does not extract claims from language, judge truth, estimate causal utility, or establish that one representation is better.

## Purpose and boundary

`tools/claim_audit.py` summarizes atomic message claims against a task's exact state and receiver knowledge. The tool does not call an LLM or inspect raw message text. A benchmark-specific extractor or blinded annotation process must first map every communicative claim to a stable `fact_id` and supply the labels below. Freeze that procedure, report its coverage and adjudication, and keep its identifier in `claim_extractor_id`; changing extraction changes the measurement condition.

A claim counts toward the theory's (q) only when all are true: the sender had observed supporting evidence, the fact was true at send time, the receiver did not already know it, and the fact was relevant to the scored task. This stricter conjunction avoids calling a repeated true fact informative or treating a plausible but unobserved statement as grounded. Component rates remain visible so a low conjunction can be diagnosed.

## Input record

One JSON object per episode in UTF-8 JSONL. A silent episode is represented by `"messages": []`; a sent message with no extracted atomic claims has `"claims": []`.

```json
{
  "schema_version": "tlu.claim-audit.v1",
  "episode_id": "task-004-seed-2",
  "protocol_id": "format-json-v1",
  "claim_extractor_id": "task-fact-map-v1",
  "stratum": {
    "experiment_id": "study-v1",
    "task_id": "private-facts-v1",
    "split": "heldout",
    "task_parameters": {"fact_count": 3},
    "model_population_id": "receiver-pair-v1",
    "sender_model": "sender-revision",
    "receiver_model": "receiver-revision",
    "scorer_id": "exact-task-v1"
  },
  "messages": [{
    "message_id": "round-1-A-to-B",
    "sender_id": "agent-A",
    "recipient_id": "agent-B",
    "claims": [{
      "claim_instance_id": "round-1-claim-1",
      "fact_id": "room-3-has-red-key",
      "sender_observed": true,
      "fact_true": true,
      "receiver_knows_before": false,
      "task_relevant": true
    }]
  }]
}
```

Each message record represents one directed delivery to one recipient; expand a broadcast into one record per receiver when receiver knowledge differs. `sender_id` and `recipient_id` identify the endpoints. Each claim instance is one atomic proposition as emitted, including repetitions. `fact_id` names the canonical task fact; it is not a substitute for the claim instance ID. All four labels are required booleans:

- `sender_observed`: the sender had direct task-grounded evidence for the proposition at send time.
- `fact_true`: the proposition matches ground truth at send time.
- `receiver_knows_before`: the receiver already had this fact before receiving this message, from its initial view or earlier valid evidence.
- `task_relevant`: the frozen task/scorer says this fact can affect the target outcome.

The full experimental stratum is mandatory, including task parameters and both endpoint model identities; results never pool across sender/receiver models, task settings, split, scorer, protocol ID, or extractor ID. Raw message contents and personal data are intentionally outside this record; retain a separately governed annotation audit trail if needed.

## Report and use

Run:

```powershell
python tools/claim_audit.py path/to/claim-labels.jsonl --output path/to/claim-report.json
```

The report gives pooled claim-level component rates and a macro mean of per-episode grounded-novel-relevant rates. It reports silent/no-claim episodes separately and uses `null` when a rate has no denominator. The pooled rate weights every claim equally; the macro rate weights each episode with one or more claims equally. Neither is a task-success estimate.

The tool does not measure extraction recall: unextracted claims are invisible. It cannot establish that a claim caused a receiver action, estimate (g), (h), or (c) in the threshold in [`THEORY.md`](THEORY.md), or replace exact task success and paired causal controls. A claim may be true, novel, and relevant yet still be misunderstood or unused. Log receiver application and terminal task outcome separately. Do not compare protocol rates unless extraction/annotation procedures are equivalent and held fixed.

Treat this ledger as only the semantic-grounding layer of a broader communication audit. A separate causal-use layer can test whether a claim changes the receiver's next-action distribution by replacing it with a pragmatic-matched proposition at the same decision point, holding all other incoming context fixed, and subtracting same-input sampling variability. Calibrate any action-sensitivity threshold on held-out identity/noise controls. Then measure whether that action change improves terminal task utility. Sensitivity is not correctness: a message can alter an action and harm the task. This distinction follows the one-step scope and calibration sensitivity documented for [Zhao et al. (2026)](../research/RELATED_WORK.md#zhao-et-al-2026-are-agents-listening-to-each-other-measuring-what-drives-agent-actions-in-llm-based-multi-agent-systems).

Validation is standard-library-only and local. Regression checks are in [`tests/test_claim_audit.py`](../tests/test_claim_audit.py); use the repository's full test command in the root README when verifying changes.
