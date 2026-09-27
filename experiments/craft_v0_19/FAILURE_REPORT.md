# v0.19 interrupted by prompt-prefill stall

The frozen Qwen3-8B CRAFT run was stopped before its first Director response completed. The llama.cpp slot remained at a 2,048-token prompt batch with zero reported prompt tokens processed for more than four minutes; the GPU stayed near 98% utilization. The server emitted no error, but produced no response, parsed message, Builder receipt, or task outcome.

This is a local inference-throughput failure, not evidence about the parser, task, or communication protocol. The exact Qwen3-8B artifact passed the preregistered size/hash checks and the endpoint served the right model alias. The server had defaulted to four parallel slots, logical batch 2,048, and physical micro-batch 512. A new feasibility revision will reduce the server to one slot and smaller batches, then rerun separately. No data from this incomplete generation are scored.
