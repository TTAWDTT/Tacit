# Interlat audit v0.1: a learned latent channel, not a portable language

**Status:** static review of the ACL 2026 paper and the authors' public implementation pinned at commit `66a89cb4d4097b2f86cbe48ed9851d6e8578f821` (shallow clone under this project's ignored `.cache/research/`). The source was not executed; no model weights were downloaded and no training or inference was run.

This audit evaluates what Interlat establishes, what it leaves open for a fair communication comparison, and how it should enter Tacit's baseline plan. It does not reproduce or dispute the paper's reported results.

## What Interlat sends and what the receiver must learn

The sender exports one final-layer hidden vector per generated plan token. The receiver inserts this sequence between learned boundary markers, runs it through a trainable attention/projection adapter, and conditions its response on the result. Training uses a supervised task loss, a matched-versus-mismatched latent separation objective, a plan-alignment term, and a stochastic curriculum that mixes natural-language plan embeddings with latents. Thus the channel carries task-specific information in a receiver-trained representation; it is not a syntax an arbitrary LLM can interpret without adaptation.

Compression is a second learned component: an autoregressive model produces a shorter latent sequence while a frozen actor and adapter provide downstream supervision. That can be a useful machine-to-machine representation when both endpoints expose hidden states and the receiver can be trained. It is not available through ordinary text-only model APIs, and “no parameter sharing” does not mean “no receiver-side training or compatibility assumptions.” Cross-family use is demonstrated for a Qwen sender and a trained LLaMA receiver, rather than for arbitrary unseen model pairs.

## What the experiments support

On ALFWorld, the paper reports gains over its Text condition for the three main backbones on seen and unseen tasks. For example, Qwen2.5-7B reports 70.48/65.42 success versus 64.29/62.44 for Text; LLaMA3.1-8B reports 70.71/70.90 versus 62.86/60.82. It also includes matched three-agent chain and tree comparisons on ALFWorld with Qwen2.5-0.5B. These are meaningful task results and make Interlat a serious latent-channel baseline, not merely an activation-transfer proposal.

The result is not uniform across tasks. On MATH, Interlat scores 36.88 overall versus 38.35 for CoT(full); its reported Level-5 score is 15.80 versus 15.05 for CoT(full), while it trails on Levels 3 and 4. This supports a task- and difficulty-dependent trade-off, not general superiority. The main paper describes results averaged over three independent runs, but the headline main tables do not provide uncertainty intervals; Table 4's ablations do. The ALFWorld/Text comparison should therefore be independently replicated with paired tasks and uncertainty before it is treated as stable.

The authors test mismatched-task latents and several perturbations, and the adapter ablation sharply reduces success. This is useful evidence that the trained receiver is sensitive to the learned channel. It does not by itself establish an advantage at equal wire bytes or isolate representation from the full training recipe: the actor, adapter, curriculum, contrastive objective, and task supervision jointly define the communicating system. The cross-family row also lacks a matched cross-family text-message condition in Table 1, so that row establishes latent transfer to a trained cross-family receiver, not a latent-over-text cross-family win.

## Efficiency claim and the missing wire accounting

Table 3 reports the “end-to-end latency ... of the message generation process.” Its untrained full-length case takes 9.19 seconds; the untrained 8-step case takes 0.39 seconds, yielding the paper's roughly 24x ratio. Trained compression to 8 steps reports 0.20 seconds and 66.43/60.45 ALFWorld success, compared with 70.48/65.42 for the full-length reference. This is a promising generation-latency/quality trade-off under the paper's setup. The table is not a measurement of serialized network bytes, transport time, all agents' full inference, or amortized training and artifact distribution.

The public collector stores hidden states as float32 and estimates their raw size at four bytes per scalar (`number_of_vectors × hidden_dimension × 4`), before framing and metadata. This gives a concrete accounting baseline, but the experiment tables do not report the tensor payload bytes or an inter-process/network transfer boundary. Tacit should separately record raw tensor bytes, chosen transport encoding and framing, receiver execution, compression and adapter artifacts, their training cost, and the reuse horizon. A shared-device tensor handoff is a different system condition from a portable transmitted message.

## Theory and research implications

Interlat motivates a task-utility bottleneck: preserve the downstream actor's behavior while reducing latent length. That is a functional compression objective relative to a specific trained receiver, not a model-independent semantic code or an information-theoretic proof that latents beat text at equal communication budget. Its learned adapter, task conditioning, and receiver access are assumptions of the result. The natural falsifiable prediction for Tacit is conditional: when both endpoints expose internals and the same receiver can be trained, learned latents may improve the quality/latency frontier on tasks where a text plan discards decision-relevant alternatives; this advantage should shrink or disappear once serialized bytes, sender plus receiver compute, and training amortization are included. For black-box/API agents, Interlat is not an eligible protocol.

For a fair comparison, keep Interlat as a separate latent-channel stratum. Match the task episodes, roles, receiver training data and budget, agent calls, and task supervision for latent and text messages; include true, same-shape other-example, neutral, and no-message controls; add a cross-family text receiver condition; sweep latent length and compare at equal serialized-byte budgets; and report paired uncertainty plus full train, inference, storage, and transport cost. Do not merge these results into the black-box text-language frontier.

## Reproduction status

The paper and source were reviewed, but the public implementation and benchmark were not run. Mark Interlat **paper-supported, locally unreplicated**. This is a resource and evidence boundary, not a code defect finding. The pinned implementation's collection path explicitly stores float32 hidden states and has documented training/evaluation scripts; no claim is made here about whether those scripts complete successfully. Revisit an actual reproduction only if the fixed host resource gate permits model inference and the project can account for the binary transport boundary.

## Sources

- Du et al. (2026), [ACL Anthology paper](https://aclanthology.org/2026.acl-long.1248/) and [PDF](https://aclanthology.org/2026.acl-long.1248.pdf).
- Authors' [official implementation](https://github.com/XiaoDu-flying/Interlat/tree/66a89cb4d4097b2f86cbe48ed9851d6e8578f821), including [hidden-state collection and float32 size estimation](https://github.com/XiaoDu-flying/Interlat/blob/66a89cb4d4097b2f86cbe48ed9851d6e8578f821/data_collection/base_data_collector.py#L125-L155), [receiver adapter](https://github.com/XiaoDu-flying/Interlat/blob/66a89cb4d4097b2f86cbe48ed9851d6e8578f821/core_training/hidden_model/custom_model.py#L45-L180), and [README evaluation and compression claims](https://github.com/XiaoDu-flying/Interlat/blob/66a89cb4d4097b2f86cbe48ed9851d6e8578f821/README.md).
