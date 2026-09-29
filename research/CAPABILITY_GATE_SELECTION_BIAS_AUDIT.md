# Capability gates must not select evaluation episodes

**Status:** correction identified and implemented in the unrun Private Match v0.2 and emergent OOD v0.3 runners on 2026-09-29. No inference data was collected under the affected preregistrations.

## Finding

Both runners required a 100%-correct full-information receiver ledger on the *same episode IDs* later used for protocol comparisons. This makes full-information correctness an inclusion rule for the evaluation sample. The resulting protocol estimate is conditional on the receiver having solved those particular gold-labeled episodes when shown the answer; it is not the predeclared task-distribution success rate. Because task difficulty affects both full-information and message-condition outcomes, the screen can preferentially retain easy cases. The condition is a possible, narrower capability-conditional estimand, but neither preregistration defined that estimand nor the runner reported it as such.

This is a form of outcome-dependent sample selection. More generally, finite-sample model or procedure selection can bias reported performance; Cawley and Talbot analyze this selection bias in model evaluation ([JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). For paired binary comparisons, the exact McNemar test conditions on discordant pairs, and power depends strongly on their rate; matched-pair design should examine power over plausible discordance values rather than choose a sample size from one unverified point ([Lennox, Statistics in Medicine 2009](https://doi.org/10.1002/sim.3683)).

## Correction

- Use a calibration seed block that is disjoint from every evaluation episode. Require the calibration receiver to pass the registered full-information criterion, then freeze the receiver/model population before opening evaluation outputs.
- Compare all preregistered evaluation episodes, regardless of their full-information outcomes. A full-information run on those evaluation episodes may be reported as an upper/control arm, but it must not decide which episodes enter protocol analysis.
- Keep the calibration ledger out of paired evaluation reports. The runner validates task family/parameters and receiver model/tokenizer/population while rejecting overlapping episode IDs.
- Treat the four/five episode screen as an execution-eligibility check only; it does not establish a stable capability rate. Use a larger independently generated evaluation set for any ranking or generalization claim.

Private Match v0.2 preregistration revision 7 makes this change before any model data. Emergent OOD v0.3 has no model execution and its stopping rule and runner now use the same disjoint-calibration rule. Older capability-screen results remain valid as descriptive outcomes for their observed prompts/tasks; they cannot be presented as independent evidence about a task population or used to filter a later confirmatory set without a new independent evaluation.
